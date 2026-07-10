"""
Delegation Voice-Note Transcriber - Vercel version
----------------------------------------------------
Runs on Vercel Functions (Fluid Compute gives the Hobby/free plan up to
300 seconds per request by default -- plenty of headroom for Gemini's
~45-90s audio calls).

Difference from the Render version: Vercel's Python runtime has no ffmpeg
available, so this sends Slack's original audio file straight to Gemini
(no format normalization, no duration lookup).
"""

import os
import json
import logging
import tempfile
import time
import urllib.request
import urllib.error
from datetime import datetime

import requests
import gspread
from flask import Flask, request
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
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
]

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
slack_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
handler = SlackRequestHandler(slack_app)
genai_client = genai.Client(api_key=GEMINI_API_KEY)
google_creds = Credentials.from_service_account_info(json.loads(GOOGLE_CREDENTIALS_JSON), scopes=GOOGLE_SCOPES)

# IMPORTANT: Vercel looks for a Flask instance named exactly "app" -- do not rename.
app = Flask(__name__)


# ---------------------------------------------------------------------------
# Google Sheets helper
# ---------------------------------------------------------------------------
def get_sheet():
    client = gspread.authorize(google_creds)
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        return sh.worksheet(GOOGLE_SHEET_TAB)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=GOOGLE_SHEET_TAB, rows=1000, cols=10)
        ws.append_row([
            "Timestamp", "Sent By", "Channel", "Transcript", "Slack Link", "Audio Length (s)",
            "Task", "Doer(s)", "Planned Date/Time", "Audio File Link",
        ])
        return ws


sheet = get_sheet()


# ---------------------------------------------------------------------------
# Slack + audio helpers
# ---------------------------------------------------------------------------
def download_slack_file(file_info):
    url = file_info["url_private_download"]
    logger.info("download_slack_file: starting, url=%s", url)

    suffix = os.path.splitext(file_info.get("name", "audio.m4a"))[1].lower() or ".m4a"
    max_attempts = 4
    attempt = 1
    last_error = None

    while attempt <= max_attempts:
        logger.info("download_slack_file: attempt %s/%s (urllib)", attempt, max_attempts)
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": "Bearer " + SLACK_BOT_TOKEN,
                    "User-Agent": "Mozilla/5.0 (compatible; DelegationTranscriber/1.0)",
                },
            )
            response = urllib.request.urlopen(req, timeout=60)
            data = response.read()
            response.close()
            logger.info("download_slack_file: urllib succeeded, %s bytes", len(data))
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data)
            tmp.close()
            return tmp.name
        except Exception as e:
            last_error = e
            logger.warning("download_slack_file: urllib attempt %s failed: %s", attempt, repr(e))

        try:
            logger.info("download_slack_file: attempt %s/%s (requests fallback)", attempt, max_attempts)
            resp = requests.get(
                url,
                headers={
                    "Authorization": "Bearer " + SLACK_BOT_TOKEN,
                    "User-Agent": "Mozilla/5.0 (compatible; DelegationTranscriber/1.0)",
                },
                timeout=60,
            )
            resp.raise_for_status()
            logger.info("download_slack_file: requests succeeded, %s bytes", len(resp.content))
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(resp.content)
            tmp.close()
            return tmp.name
        except Exception as e:
            last_error = e
            logger.warning("download_slack_file: requests attempt %s failed: %s", attempt, repr(e))

        if attempt == max_attempts:
            logger.error("download_slack_file: all %s attempts failed", max_attempts)
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

                message_dt = datetime.fromtimestamp(float(event["ts"]))
                result = analyze_audio(local_path, message_dt)
                logger.info("Gemini analysis complete.")

                user_name = get_user_name(event.get("user", ""))
                permalink = get_permalink(event["channel"], event["ts"])
                timestamp = message_dt.strftime("%Y-%m-%d %H:%M:%S")

                sheet.append_row([
                    timestamp,
                    user_name,
                    event["channel"],
                    result["transcript"],
                    permalink,
                    "",
                    result["task"],
                    result["doers"],
                    result["planned_date_time"],
                ])
                logger.info("Logged transcript + task info from %s to Google Sheet.", user_name)

                say(text="Transcribed and logged to the Delegation Sheet.", thread_ts=event["ts"])
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
    retry_num = request.headers.get("X-Slack-Retry-Num")
    try:
        body = request.get_json(silent=True) or {}
        logger.info(
            "Incoming /slack/events: retry_num=%s type=%s event_type=%s",
            retry_num, body.get("type"), (body.get("event") or {}).get("type"),
        )
    except Exception:
        logger.exception("Failed to log incoming request body")

    if retry_num:
        return "", 200
    return handler.handle(request)


@app.route("/", methods=["GET"])
def health():
    return "Delegation Transcriber (Vercel) is running.", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
