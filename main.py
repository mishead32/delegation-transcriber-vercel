"""
Delegation Voice-Note Transcriber - Vercel version
----------------------------------------------------
Runs on Vercel Functions (Fluid Compute gives the Hobby/free plan up to
300 seconds per request by default -- plenty of headroom for Gemini's
~45-90s audio calls).

Difference from the Render version: Vercel's Python runtime has no ffmpeg
available, so this sends Slack's original audio file straight to Gemini
(no format normalization, no duration lookup).

Serverless-critical details (do not undo):
- App(process_before_response=True): on Vercel, background threads are frozen
  the instant the HTTP response is sent. Bolt's default ack-first behavior
  therefore silently killed the files.slack.com download. Everything must run
  before the response.
- /slack/events drops Slack's automatic retries (X-Slack-Retry-Num header),
  otherwise the >3s ack time would cause duplicate processing.
"""

import os
import json
import logging
import ssl
import tempfile
import time
import urllib.request
import urllib.error
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import gspread
from flask import Flask, request
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials as UserCredentials
from google.oauth2.service_account import Credentials
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("delegation-transcriber")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
DELEGATION_CHANNEL_ID = os.environ["DELEGATION_CHANNEL_ID"]

# Vercel servers run in UTC. Voice notes say things like "kal"/"tomorrow",
# so resolve the reference date in YOUR timezone, not the server's.
LOCAL_TIMEZONE = os.environ.get("LOCAL_TIMEZONE", "Asia/Kolkata")

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

ANALYSIS_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "transcript": types.Schema(
            type=types.Type.STRING,
            description="Exact word-for-word transcript in the original spoken language/script.",
        ),
        "task": types.Schema(
            type=types.Type.STRING,
            description="A clear, concise statement of the actual task/instruction being assigned, "
                        "with filler words removed, in the same language as the transcript.",
        ),
        "doers": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Name(s) of the person/people responsible for doing the task, exactly as "
                        "mentioned. Empty list if no name is mentioned.",
        ),
        "planned_date_time": types.Schema(
            type=types.Type.STRING,
            description="The planned date/time to complete the task, if mentioned, resolved to an "
                        "actual date using the provided reference date (e.g. 'aaj'/'today' -> that "
                        "date, 'kal'/'tomorrow' -> next day). Format as DD-MM-YYYY, optionally with "
                        "a time. Empty string if no date/time is mentioned at all.",
        ),
    },
    required=["transcript", "task", "doers", "planned_date_time"],
)

GOOGLE_SHEET_ID = os.environ["GOOGLE_SHEET_ID"]
GOOGLE_SHEET_TAB = os.environ.get("GOOGLE_SHEET_TAB", "Delegation Log")
GOOGLE_CREDENTIALS_JSON = os.environ["GOOGLE_CREDENTIALS_JSON"]

# Google Drive folder where audio files are archived. Create a folder in your
# Drive, share it with the service account's client_email as Editor, and put
# its ID here (the part of the folder URL after /folders/). If this env var is
# not set, the Drive upload is skipped and the sheet's link column stays empty
# -- the transcription pipeline is never affected.
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "").strip()

# Google no longer gives service accounts ANY Drive storage quota, so a
# service-account upload into a My Drive folder fails with 403
# storageQuotaExceeded even when the folder is shared with it. The fix is to
# upload as the real user via OAuth. Generate these three values once with
# get_drive_token.py (in this folder) and set them in Vercel. If they are not
# set, the code falls back to the service account (works only for Shared
# Drives on Google Workspace).
GDRIVE_OAUTH_CLIENT_ID = os.environ.get("GDRIVE_OAUTH_CLIENT_ID", "").strip()
GDRIVE_OAUTH_CLIENT_SECRET = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET", "").strip()
GDRIVE_OAUTH_REFRESH_TOKEN = os.environ.get("GDRIVE_OAUTH_REFRESH_TOKEN", "").strip()

AUDIO_SUBTYPES = {"audio", "m4a", "mp3", "mp4", "wav", "ogg", "webm", "aac"}
MIME_OVERRIDES = {
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
}

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
# process_before_response=True is REQUIRED on Vercel (and any serverless/FaaS
# runtime). Without it, Bolt returns the HTTP 200 ack to Slack immediately and
# runs listeners in a background thread -- but Vercel FREEZES the execution
# environment the moment the response is sent, suspending that thread silently
# in the middle of the files.slack.com download (no exception, no log). With
# this flag, the whole listener runs BEFORE the response is returned, so the
# invocation stays alive for the full pipeline (covered by maxDuration=300).
slack_app = App(
    token=SLACK_BOT_TOKEN,
    signing_secret=SLACK_SIGNING_SECRET,
    process_before_response=True,
)
handler = SlackRequestHandler(slack_app)
genai_client = genai.Client(api_key=GEMINI_API_KEY)
google_creds = Credentials.from_service_account_info(json.loads(GOOGLE_CREDENTIALS_JSON), scopes=GOOGLE_SCOPES)

# IMPORTANT: Vercel looks for a Flask instance named exactly "app" -- do not rename.
app = Flask(__name__)


# ---------------------------------------------------------------------------
# Google Sheets helper
# ---------------------------------------------------------------------------
_sheet = None


def get_sheet():
    # Lazy + cached: doing this at module import time means a Google Sheets
    # network call on every cold start, and an import-time crash (HTTP 500 to
    # Slack) if Sheets is momentarily unreachable. Fetch on first use instead.
    global _sheet
    if _sheet is not None:
        return _sheet
    client = gspread.authorize(google_creds)
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        _sheet = sh.worksheet(GOOGLE_SHEET_TAB)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=GOOGLE_SHEET_TAB, rows=1000, cols=10)
        ws.append_row([
            "Timestamp", "Sent By", "Channel", "Transcript", "Slack Link", "Audio Length (s)",
            "Task", "Doer(s)", "Planned Date/Time", "Audio File Link",
        ])
        _sheet = ws
    return _sheet


# ---------------------------------------------------------------------------
# Google Drive helper (fail-soft: any problem here returns "" and the main
# transcription pipeline continues untouched)
# ---------------------------------------------------------------------------
_drive_user_creds = None


def _get_drive_access_token():
    """Access token for Drive uploads. Prefers the user's OAuth credentials
    (files get owned by YOU and use YOUR storage quota); falls back to the
    service account, which only works for Shared Drives."""
    global _drive_user_creds
    if GDRIVE_OAUTH_CLIENT_ID and GDRIVE_OAUTH_CLIENT_SECRET and GDRIVE_OAUTH_REFRESH_TOKEN:
        if _drive_user_creds is None:
            _drive_user_creds = UserCredentials(
                token=None,
                refresh_token=GDRIVE_OAUTH_REFRESH_TOKEN,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=GDRIVE_OAUTH_CLIENT_ID,
                client_secret=GDRIVE_OAUTH_CLIENT_SECRET,
                scopes=["https://www.googleapis.com/auth/drive"],
            )
        if not _drive_user_creds.valid:
            _drive_user_creds.refresh(GoogleAuthRequest())
        return _drive_user_creds.token
    logger.warning(
        "upload_audio_to_drive: GDRIVE_OAUTH_* env vars not set -- falling back "
        "to the service account, which CANNOT upload to My Drive folders "
        "(no storage quota). Run get_drive_token.py and set the env vars."
    )
    if not google_creds.valid:
        google_creds.refresh(GoogleAuthRequest())
    return google_creds.token


def upload_audio_to_drive(local_path, original_name, message_dt):
    """Uploads the audio to the Drive folder, makes it link-viewable,
    and returns the webViewLink. Returns "" on any failure or if not configured.

    Uses the Drive v3 REST API directly, so no new pip dependencies are needed.
    """
    if not GDRIVE_FOLDER_ID:
        logger.info("upload_audio_to_drive: GDRIVE_FOLDER_ID not set, skipping upload.")
        return ""
    try:
        token = _get_drive_access_token()

        safe_name = original_name or "voice_note.m4a"
        drive_name = message_dt.strftime("%Y-%m-%d_%H-%M-%S") + "_" + safe_name
        ext = os.path.splitext(safe_name)[1].lower()
        mime_type = MIME_OVERRIDES.get(ext, "audio/mp4")

        metadata = {"name": drive_name, "parents": [GDRIVE_FOLDER_ID]}
        with open(local_path, "rb") as fh:
            audio_bytes = fh.read()

        resp = requests.post(
            "https://www.googleapis.com/upload/drive/v3/files"
            "?uploadType=multipart&fields=id,webViewLink&supportsAllDrives=true",
            headers={"Authorization": "Bearer " + token},
            files={
                "metadata": ("metadata", json.dumps(metadata), "application/json; charset=UTF-8"),
                "file": (drive_name, audio_bytes, mime_type),
            },
            timeout=60,
        )
        resp.raise_for_status()
        info = resp.json()
        file_id = info["id"]
        link = info.get("webViewLink") or ("https://drive.google.com/file/d/" + file_id + "/view")
        logger.info("upload_audio_to_drive: uploaded %s as file id %s", drive_name, file_id)

        # Make the file viewable by anyone with the link, so the sheet link
        # works for everyone. If this step fails we still return the link
        # (folder-level sharing may already cover the viewers).
        try:
            perm_resp = requests.post(
                "https://www.googleapis.com/drive/v3/files/" + file_id
                + "/permissions?supportsAllDrives=true",
                headers={
                    "Authorization": "Bearer " + token,
                    "Content-Type": "application/json",
                },
                data=json.dumps({"role": "reader", "type": "anyone"}),
                timeout=30,
            )
            perm_resp.raise_for_status()
        except Exception as e:
            logger.warning("upload_audio_to_drive: could not set link-sharing permission: %s", repr(e))

        return link
    except Exception as e:
        # Include the API's response body -- Google's 403s carry the real
        # reason (e.g. storageQuotaExceeded, accessNotConfigured) there.
        detail = ""
        err_resp = getattr(e, "response", None)
        if err_resp is not None:
            try:
                detail = " | body: " + err_resp.text[:500]
            except Exception:
                pass
        logger.warning("upload_audio_to_drive: upload failed (continuing without link): %s%s", repr(e), detail)
        return ""


# ---------------------------------------------------------------------------
# Slack + audio helpers
# ---------------------------------------------------------------------------
def _save_bytes(data, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def download_slack_file(file_info):
    url = file_info["url_private_download"]
    logger.info("download_slack_file: starting, url=%s", url)

    suffix = os.path.splitext(file_info.get("name", "audio.m4a"))[1].lower() or ".m4a"
    auth_headers = {
        "Authorization": "Bearer " + SLACK_BOT_TOKEN,
        "User-Agent": "Mozilla/5.0 (compatible; DelegationTranscriber/1.0)",
    }

    max_attempts = 3
    attempt = 1
    last_error = None

    while attempt <= max_attempts:
        # Method 1: urllib with default SSL context
        logger.info("download_slack_file: attempt %s/%s (urllib, default TLS)", attempt, max_attempts)
        try:
            req = urllib.request.Request(url, headers=auth_headers)
            response = urllib.request.urlopen(req, timeout=15)
            data = response.read()
            response.close()
            logger.info("download_slack_file: urllib (default TLS) succeeded, %s bytes", len(data))
            return _save_bytes(data, suffix)
        except Exception as e:
            last_error = e
            logger.warning("download_slack_file: urllib (default TLS) attempt %s failed: %s", attempt, repr(e))

        # Method 2: urllib forcing TLS 1.2 explicitly (in case the CDN drops
        # the connection during a TLS 1.3 handshake specifically).
        logger.info("download_slack_file: attempt %s/%s (urllib, forced TLS 1.2)", attempt, max_attempts)
        try:
            ctx = ssl.create_default_context()
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.maximum_version = ssl.TLSVersion.TLSv1_2
            req = urllib.request.Request(url, headers=auth_headers)
            response = urllib.request.urlopen(req, timeout=15, context=ctx)
            data = response.read()
            response.close()
            logger.info("download_slack_file: urllib (TLS 1.2) succeeded, %s bytes", len(data))
            return _save_bytes(data, suffix)
        except Exception as e:
            last_error = e
            logger.warning("download_slack_file: urllib (TLS 1.2) attempt %s failed: %s", attempt, repr(e))

        # Method 3: requests library fallback.
        logger.info("download_slack_file: attempt %s/%s (requests fallback)", attempt, max_attempts)
        try:
            resp = requests.get(url, headers=auth_headers, timeout=15)
            resp.raise_for_status()
            logger.info("download_slack_file: requests succeeded, %s bytes", len(resp.content))
            return _save_bytes(resp.content, suffix)
        except Exception as e:
            last_error = e
            logger.warning("download_slack_file: requests attempt %s failed: %s", attempt, repr(e))

        if attempt == max_attempts:
            logger.error("download_slack_file: all %s attempts (x3 methods each) failed", max_attempts)
            raise last_error

        wait = 3 * attempt
        time.sleep(wait)
        attempt = attempt + 1


def analyze_audio(local_path, reference_dt):
    """Transcribes the audio AND extracts task / doer(s) / planned date-time in one Gemini call."""
    reference_str = reference_dt.strftime("%A, %d %B %Y")
    prompt = (
        "This is a voice note from a workplace delegation/task-assignment Slack channel. "
        "The speaker may use Urdu, Hindi, or English, or a mix of these.\n\n"
        "The reference date for this message is: " + reference_str + ".\n\n"
        "Listen to the audio and return a JSON object with these fields:\n"
        "1. transcript - exact word-for-word transcript in the original spoken language/script. "
        "No translation, no commentary.\n"
        "2. task - a clear, concise statement of the actual task/instruction being assigned, in the "
        "same language as the transcript, with filler words removed.\n"
        "3. doers - a list of the name(s) of the person/people who are supposed to do the task, "
        "exactly as mentioned. Empty list if no name is mentioned.\n"
        "4. planned_date_time - the planned date/time to complete the task, if mentioned. Resolve "
        "relative terms like 'aaj'/'today', 'kal'/'tomorrow', 'parso', or day names (Monday, etc.) "
        "into an actual date using the reference date above. Format as DD-MM-YYYY, optionally with "
        "a time (e.g. '10-07-2026, 5:00 PM'). Empty string if no date/time is mentioned at all.\n\n"
        "Return ONLY the JSON object."
    )

    ext = os.path.splitext(local_path)[1].lower()
    mime_type = MIME_OVERRIDES.get(ext, "audio/mp4")
    myfile = genai_client.files.upload(file=local_path, config={"mime_type": mime_type})

    max_attempts = 4
    response = None
    attempt = 1
    while attempt <= max_attempts:
        try:
            response = genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt, myfile],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ANALYSIS_SCHEMA,
                ),
            )
            break
        except genai_errors.ServerError as e:
            if attempt == max_attempts:
                raise
            wait = 10 * attempt
            logger.warning("Gemini overloaded (attempt %s/%s), retrying in %ss: %s", attempt, max_attempts, wait, e)
            time.sleep(wait)
            attempt = attempt + 1

    data = json.loads(response.text)
    return {
        "transcript": (data.get("transcript") or "").strip(),
        "task": (data.get("task") or "").strip(),
        "doers": ", ".join(d.strip() for d in (data.get("doers") or []) if d.strip()),
        "planned_date_time": (data.get("planned_date_time") or "").strip() or "Not mentioned",
    }


def get_user_name(user_id):
    try:
        info = slack_app.client.users_info(user=user_id)
        u = info["user"]
        return u.get("real_name") or u.get("name") or user_id
    except Exception:
        return user_id


def get_permalink(channel, ts):
    try:
        r = slack_app.client.chat_getPermalink(channel=channel, message_ts=ts)
        return r["permalink"]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Bolt-level error handler
# ---------------------------------------------------------------------------
@slack_app.error
def handle_bolt_errors(error, body, logger):
    logger.exception("Bolt-level error handling event: " + str(error))


# ---------------------------------------------------------------------------
# Event handler: any audio file posted in the Delegation channel
# Runs fully synchronously so Vercel keeps the invocation alive for the
# whole pipeline (up to maxDuration 300s).
# ---------------------------------------------------------------------------
@slack_app.event("message")
def handle_message_events(event, say, logger):
    try:
        logger.info(
            "Received message event: channel=%s (expecting %s) has_files=%s subtype=%s",
            event.get("channel"), DELEGATION_CHANNEL_ID, bool(event.get("files")), event.get("subtype"),
        )
        if event.get("channel") != DELEGATION_CHANNEL_ID:
            return
        if event.get("subtype") == "message_changed":
            return

        files = event.get("files", [])
        if not files:
            return

        for f in files:
            mimetype = f.get("mimetype", "")
            filetype = f.get("filetype", "")
            logger.info("File in message: name=%s mimetype=%s filetype=%s", f.get("name"), mimetype, filetype)
            if not (mimetype.startswith("audio/") or filetype in AUDIO_SUBTYPES):
                logger.info("Skipping file -- not recognized as audio.")
                continue

            logger.info("New audio file detected: %s", f.get("name"))
            local_path = None
            try:
                local_path = download_slack_file(f)
                logger.info("Downloaded file to %s", local_path)

                message_dt = datetime.fromtimestamp(float(event["ts"]), tz=ZoneInfo(LOCAL_TIMEZONE))
                result = analyze_audio(local_path, message_dt)
                logger.info("Gemini analysis complete.")

                user_name = get_user_name(event.get("user", ""))
                permalink = get_permalink(event["channel"], event["ts"])
                timestamp = message_dt.strftime("%Y-%m-%d %H:%M:%S")

                # Archive the audio in Google Drive (fail-soft: "" on any error).
                audio_link = upload_audio_to_drive(local_path, f.get("name"), message_dt)

                get_sheet().append_row([
                    timestamp,
                    user_name,
                    event["channel"],
                    result["transcript"],
                    permalink,
                    "",
                    result["task"],
                    result["doers"],
                    result["planned_date_time"],
                    audio_link,
                ])
                logger.info("Logged transcript + task info from %s to Google Sheet.", user_name)
            except Exception as e:
                logger.exception("Failed to process audio file")
                say(text="Couldn't transcribe this audio: " + str(e), thread_ts=event["ts"])
            finally:
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)
    except BaseException:
        logger.exception("Unhandled error in handle_message_events")
        raise


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/slack/events", methods=["POST"])
def slack_events():
    # NOTE: do NOT read request.get_json()/request.data here before calling
    # handler.handle(request) -- consuming the body first breaks Bolt's own
    # ability to read it afterward in this environment, causing it to silently
    # fail to dispatch to any listener while still returning 200.
    # With process_before_response=True the original request takes 45-90s, so
    # Slack's 3-second ack deadline WILL be exceeded and Slack WILL send
    # retries while the original invocation is still working. Dropping retries
    # here is what prevents duplicate transcriptions/sheet rows -- do not
    # remove this check. X-Slack-No-Retry asks Slack to stop retrying sooner.
    if request.headers.get("X-Slack-Retry-Num"):
        return "", 200, {"X-Slack-No-Retry": "1"}
    return handler.handle(request)


@app.route("/", methods=["GET"])
def health():
    return "Delegation Transcriber (Vercel) is running.", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
