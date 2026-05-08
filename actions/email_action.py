# actions/email_action.py

"""
Telegram API action layer for CRYPTO AI.

Backward compatible with the existing pipeline import:
    from actions.email_action import send_email_alert

Important:
- Telegram is the ONLY real API action channel in this version.
- The function name remains send_email_alert() because pipeline_runner.py already imports it.
- success=True ONLY when Telegram actually sends the alert.
- If Telegram credentials are missing or sending fails, the action is logged but success=False.
- No Gmail/SMTP fallback is used, to avoid Streamlit showing email-related false positives.
"""

from __future__ import annotations

import csv
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    from utils.config import ACTION_LOG_PATH
except Exception:
    ACTION_LOG_PATH = "data/outputs/action_log.csv"

load_dotenv()


# =========================================================
# Helpers
# =========================================================

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_str(value: Any, limit: int | None = None) -> str:
    text = str(value or "")
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 3)] + "..."
    return text


def action_log_path() -> Path:
    return Path(os.getenv("ACTION_LOG_PATH", ACTION_LOG_PATH))


def read_existing_header(path: Path) -> List[str] | None:
    """
    Keeps compatibility with an existing CSV log file.

    If the old action_log.csv already has a header, we write only those columns.
    If the file is new, we use the new Telegram-aware header.
    """

    try:
        if not path.exists() or path.stat().st_size == 0:
            return None

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)

        return header if header else None

    except Exception:
        return None


# =========================================================
# Logging
# =========================================================

DEFAULT_LOG_FIELDS: List[str] = [
    "timestamp",
    "action_type",
    "channel",
    "subject",
    "success",
    "message",
    "telegram_attempted",
    "telegram_sent",
    "telegram_message_count",
    "telegram_messages_sent",
    "email_attempted",
    "email_sent",
    "chat_id",
    "error",
    "body_preview",
]


def save_action_log(
    subject: str,
    body: str,
    success: bool,
    message: str,
    channel: str = "telegram",
    telegram_attempted: bool = False,
    telegram_sent: bool = False,
    telegram_message_count: int = 0,
    telegram_messages_sent: int = 0,
    chat_id: str = "",
    error: str = "",
) -> bool:
    """
    Append an API action audit row to CSV.
    """

    try:
        path = action_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        existing_header = read_existing_header(path)
        fieldnames = existing_header or DEFAULT_LOG_FIELDS

        row = {
            "timestamp": now_ts(),
            "action_type": "crypto_ai_telegram_api_action",
            "channel": channel,
            "subject": safe_str(subject, 180),
            "success": bool(success),
            "message": safe_str(message, 500),
            "telegram_attempted": bool(telegram_attempted),
            "telegram_sent": bool(telegram_sent),
            "telegram_message_count": int(telegram_message_count or 0),
            "telegram_messages_sent": int(telegram_messages_sent or 0),
            "email_attempted": False,
            "email_sent": False,
            "chat_id": safe_str(chat_id, 80),
            "error": safe_str(error, 500),
            "body_preview": safe_str(body, 300),
        }

        file_exists = path.exists() and path.stat().st_size > 0

        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        return True

    except Exception:
        return False


# =========================================================
# Result payload
# =========================================================

def build_result(
    success: bool,
    message: str,
    channel: str = "telegram",
    telegram_attempted: bool = False,
    telegram_sent: bool = False,
    telegram_message_count: int = 0,
    telegram_messages_sent: int = 0,
    action_logged: bool = False,
    error: str = "",
    telegram_response: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Unified result returned to pipeline_runner.py and Streamlit.
    """

    return {
        "success": bool(success),
        "message": str(message or ""),
        "channel": channel,
        "telegram_attempted": bool(telegram_attempted),
        "telegram_sent": bool(telegram_sent),
        "telegram_message_count": int(telegram_message_count or 0),
        "telegram_messages_sent": int(telegram_messages_sent or 0),
        "email_attempted": False,
        "email_sent": False,
        "action_logged": bool(action_logged),
        "log_path": str(action_log_path()),
        "error": str(error or ""),
        "telegram_response": telegram_response or {},
        "timestamp": now_ts(),
    }


# =========================================================
# Telegram configuration
# =========================================================

def get_telegram_config() -> Dict[str, str]:
    return {
        "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        "api_url": os.getenv("TELEGRAM_API_URL", "https://api.telegram.org").strip().rstrip("/"),
    }


def telegram_configured() -> bool:
    cfg = get_telegram_config()
    return bool(cfg["bot_token"] and cfg["chat_id"])


# =========================================================
# Telegram message formatting
# =========================================================

def build_telegram_report(subject: str, body: str) -> str:
    """
    Builds a plain-text Telegram message.

    Plain text is used instead of HTML/Markdown to avoid parse_mode failures
    caused by long technical reports, angle brackets, markdown symbols, or
    incomplete escaped entities.
    """

    subject = safe_str(subject, 180)
    body = str(body or "").strip()

    return (
        "🛡️ CRYPTO AI Alert\n\n"
        f"Subject: {subject}\n\n"
        "Report:\n"
        f"{body}"
    )


def split_telegram_message(text: str, max_length: int = 3900) -> List[str]:
    """
    Telegram sendMessage limit is 4096 characters.
    We split long reports safely into multiple messages.
    """

    text = str(text or "")

    if len(text) <= max_length:
        return [text]

    chunks: List[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Prefer splitting at a newline to keep report readable.
        split_at = remaining.rfind("\n", 0, max_length)

        if split_at < max_length * 0.55:
            split_at = max_length

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    total = len(chunks)

    if total > 1:
        chunks = [
            f"[Part {idx}/{total}]\n\n{chunk}"
            for idx, chunk in enumerate(chunks, start=1)
        ]

    return chunks


# =========================================================
# Telegram sending
# =========================================================

def send_telegram_message(text: str) -> Dict[str, Any]:
    """
    Sends one Telegram message using Telegram Bot API.
    """

    cfg = get_telegram_config()

    endpoint = f"{cfg['api_url']}/bot{cfg['bot_token']}/sendMessage"

    payload = {
        "chat_id": cfg["chat_id"],
        "text": str(text or ""),
        "disable_web_page_preview": "true",
    }

    encoded_payload = urllib.parse.urlencode(payload).encode("utf-8")

    request = urllib.request.Request(
        endpoint,
        data=encoded_payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response_text = response.read().decode("utf-8", errors="replace")

        parsed = json.loads(response_text)

        return {
            "success": bool(parsed.get("ok")),
            "response": parsed,
            "error": "" if bool(parsed.get("ok")) else str(parsed),
        }

    except Exception as e:
        return {
            "success": False,
            "response": {},
            "error": str(e),
        }


def send_telegram_alert(subject: str, body: str) -> Dict[str, Any]:
    """
    Sends the CRYPTO AI final report through Telegram.
    Long reports are split into multiple Telegram messages.
    """

    cfg = get_telegram_config()

    if not cfg["bot_token"] or not cfg["chat_id"]:
        return {
            "success": False,
            "telegram_attempted": False,
            "telegram_sent": False,
            "telegram_message_count": 0,
            "telegram_messages_sent": 0,
            "message": "Telegram skipped because TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.",
            "error": "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.",
            "chat_id": cfg["chat_id"],
            "telegram_response": {},
        }

    full_text = build_telegram_report(subject, body)
    chunks = split_telegram_message(full_text, max_length=3900)

    responses: List[Dict[str, Any]] = []
    sent_count = 0

    for chunk in chunks:
        send_result = send_telegram_message(chunk)
        responses.append(send_result)

        if send_result.get("success"):
            sent_count += 1
        else:
            # Stop at first failed chunk so the result is honest.
            break

    all_sent = sent_count == len(chunks)

    if all_sent:
        return {
            "success": True,
            "telegram_attempted": True,
            "telegram_sent": True,
            "telegram_message_count": len(chunks),
            "telegram_messages_sent": sent_count,
            "message": f"Telegram alert sent successfully ({sent_count}/{len(chunks)} message part(s)).",
            "error": "",
            "chat_id": cfg["chat_id"],
            "telegram_response": {
                "parts": responses,
            },
        }

    last_error = ""
    if responses:
        last_error = str(responses[-1].get("error", ""))

    return {
        "success": False,
        "telegram_attempted": True,
        "telegram_sent": False,
        "telegram_message_count": len(chunks),
        "telegram_messages_sent": sent_count,
        "message": f"Telegram alert failed after sending {sent_count}/{len(chunks)} message part(s).",
        "error": last_error or "Unknown Telegram sending error.",
        "chat_id": cfg["chat_id"],
        "telegram_response": {
            "parts": responses,
        },
    }


# =========================================================
# Backward-compatible pipeline action
# =========================================================

def send_email_alert(subject: str, body: str) -> Dict[str, Any]:
    """
    Main API action called by pipeline_runner.py.

    Despite the function name, Telegram is the actual real-world API action.
    The name is preserved so no pipeline imports need to change.
    """

    subject = str(subject or "CRYPTO AI Alert").strip()
    body = str(body or "").strip()

    telegram_result = send_telegram_alert(subject, body)

    action_logged = save_action_log(
        subject=subject,
        body=body,
        success=bool(telegram_result.get("success")),
        message=telegram_result.get("message", ""),
        channel="telegram",
        telegram_attempted=bool(telegram_result.get("telegram_attempted")),
        telegram_sent=bool(telegram_result.get("telegram_sent")),
        telegram_message_count=int(telegram_result.get("telegram_message_count", 0)),
        telegram_messages_sent=int(telegram_result.get("telegram_messages_sent", 0)),
        chat_id=telegram_result.get("chat_id", ""),
        error=telegram_result.get("error", ""),
    )

    return build_result(
        success=bool(telegram_result.get("success")),
        message=telegram_result.get("message", ""),
        channel="telegram",
        telegram_attempted=bool(telegram_result.get("telegram_attempted")),
        telegram_sent=bool(telegram_result.get("telegram_sent")),
        telegram_message_count=int(telegram_result.get("telegram_message_count", 0)),
        telegram_messages_sent=int(telegram_result.get("telegram_messages_sent", 0)),
        action_logged=action_logged,
        error=telegram_result.get("error", ""),
        telegram_response=telegram_result.get("telegram_response", {}),
    )
