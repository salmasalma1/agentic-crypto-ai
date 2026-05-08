# agent/cipher_text_extractor.py

"""
Optional LLM + deterministic fallback extractor.

The LLM role is intentionally restricted:
- It may only choose the most likely encoded/cipher token from local regex candidates.
- It never classifies the cipher type.
- It never decodes the cipher.

If Azure/OpenAI credentials are missing, or if the LLM fails, regex fallback is used safely.

Supported extraction families:
- Base64
- Base32
- Morse
- Hex / Hexadecimal
- ASCII decimal codes
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


# ==============================
# Candidate Cleaning / Validation
# ==============================

def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _ascii_number_tokens(candidate: str) -> List[int]:
    """
    Extracts ASCII decimal numbers from a candidate.
    Only 1-3 digit tokens are considered.
    """

    values: List[int] = []

    for token in re.findall(r"\d{1,3}", str(candidate or "")):
        try:
            values.append(int(token))
        except Exception:
            pass

    return values


def is_likely_ascii_decimal_candidate(candidate: str) -> bool:
    """
    Detects ASCII decimal-code sequences such as:
        83 79 67 32 65 76 69 82 84

    Conditions are intentionally strict enough to avoid random years/IDs:
    - at least 4 numeric tokens
    - most tokens are valid printable ASCII range 32-126
    - the text contains a separator between numbers
    """

    raw = str(candidate or "").strip()

    if not raw:
        return False

    numbers = _ascii_number_tokens(raw)

    if len(numbers) < 4:
        return False

    valid = [num for num in numbers if 32 <= num <= 126]
    valid_ratio = len(valid) / max(len(numbers), 1)

    has_separator = bool(re.search(r"\d[\s,;:/|-]+\d", raw))

    return valid_ratio >= 0.75 and has_separator


def is_likely_hex_candidate(candidate: str) -> bool:
    """
    Detects Hex / hexadecimal candidates.

    Supports:
    - compact hex: 54484520534f43
    - byte-separated hex: 54 48 45 20 53 4f 43

    Requirements:
    - only hex chars and common byte separators
    - after removing separators, length is even
    - at least 12 hex characters
    """

    raw = str(candidate or "").strip()

    if not raw:
        return False

    if not re.fullmatch(r"[A-Fa-f0-9\s:,\-]+", raw):
        return False

    compact = re.sub(r"[\s:,\-]+", "", raw)

    if len(compact) < 12:
        return False

    if len(compact) % 2 != 0:
        return False

    if not re.fullmatch(r"[A-Fa-f0-9]+", compact):
        return False

    return True


def is_morse_candidate(candidate: str) -> bool:
    candidate = str(candidate or "").strip()
    return (
        bool(candidate)
        and all(ch in ".-/ " for ch in candidate)
        and any(ch in ".-" for ch in candidate)
        and len(candidate) >= 7
    )


def normalize_candidate(candidate: str) -> str:
    """
    Clean the selected candidate while preserving valid cipher characters.

    Important:
    - Morse uses dots, dashes, slashes, and spaces.
    - ASCII decimal codes use spaces/commas/separators between numbers.
    - Hex may use spaces or byte separators.
    - Base64/Base32 may use equals signs for padding.
    """

    candidate = str(candidate or "").strip()

    if not candidate:
        return ""

    # Preserve Morse exactly except for redundant outer whitespace.
    if is_morse_candidate(candidate):
        return _collapse_spaces(candidate)

    # Preserve ASCII decimal separators as spaces so decode_ascii can read them.
    if is_likely_ascii_decimal_candidate(candidate):
        # Normalize separators to single spaces.
        cleaned = re.sub(r"[,;:/|\-]+", " ", candidate)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    # Preserve Hex content, but normalize byte separators to spaces.
    # bytes.fromhex() accepts whitespace, and our decoder can also clean it.
    if is_likely_hex_candidate(candidate):
        cleaned = re.sub(r"[:,\-]+", " ", candidate)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    # Clean surrounding punctuation only, but preserve internal symbols and padding.
    return candidate.strip(" \n\r\t\"'`.,;:()[]{}<>")


def parse_llm_confidence(value: Any, default: float = 0.8) -> float:
    """
    Convert LLM confidence into a safe numeric value between 0 and 1.

    The LLM may return:
    - 0.9
    - "0.9"
    - "high"
    - "medium"
    - "low"

    This prevents fallback just because confidence was returned as text.
    """

    if value is None:
        return float(default)

    if isinstance(value, (int, float)):
        confidence = float(value)
        return max(0.0, min(1.0, confidence))

    text = str(value).strip().lower()

    word_map = {
        "very high": 0.95,
        "high": 0.90,
        "medium": 0.70,
        "moderate": 0.70,
        "normal": 0.60,
        "low": 0.40,
        "very low": 0.20,
    }

    if text in word_map:
        return word_map[text]

    # Handle strings like "90%" or "0.9"
    try:
        cleaned = text.replace("%", "").strip()
        confidence = float(cleaned)

        if confidence > 1:
            confidence = confidence / 100.0

        return max(0.0, min(1.0, confidence))

    except Exception:
        return float(default)


# ==============================
# Regex Candidate Extraction
# ==============================

def candidate_score(candidate: str) -> float:
    """
    Scores candidates for deterministic fallback.

    This does NOT classify for the ML pipeline.
    It only helps choose the most complete token when LLM selection is unavailable.
    """

    candidate = normalize_candidate(candidate)

    if not candidate:
        return 0.0

    score = float(len(candidate))

    if is_likely_ascii_decimal_candidate(candidate):
        score += 300.0

    if is_likely_hex_candidate(candidate):
        score += 260.0

    if is_morse_candidate(candidate):
        score += 220.0

    if "=" in candidate:
        score += 90.0

    if re.search(r"[+/]", candidate):
        score += 35.0

    # Prefer candidates that contain separators for ASCII/Morse style tokens.
    if re.search(r"\d\s+\d", candidate):
        score += 50.0

    return score


def _add_candidate(candidates: List[str], raw_match: str) -> None:
    candidate = normalize_candidate(raw_match)

    if not candidate:
        return

    # Avoid candidates that are only spaces or too weak.
    if candidate in candidates:
        return

    # For Morse, require at least one dot or dash.
    if all(ch in ".-/ " for ch in candidate) and not any(ch in ".-" for ch in candidate):
        return

    candidates.append(candidate)


def regex_candidate_extract(text: str) -> List[str]:
    """
    Extract possible encoded/cipher tokens from messy natural language text.

    Supported candidates:
    - ASCII decimal code sequences
    - Hex / hexadecimal strings
    - Morse-like strings
    - Base32-like strings
    - Base64-like strings
    """

    text = str(text or "").strip()

    if not text:
        return []

    candidates: List[str] = []

    # -----------------------------------------------------
    # 1) ASCII decimal codes
    # Example:
    # 83 79 67 32 65 76 69 82 84
    #
    # Must be separated numbers to avoid catching regular numeric IDs.
    # -----------------------------------------------------
    ascii_patterns = [
        r"(?<!\d)((?:\d{1,3}[\s,;:/|\-]+){3,}\d{1,3})(?!\d)",
    ]

    for pattern in ascii_patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                match = "".join(match)

            if is_likely_ascii_decimal_candidate(match):
                _add_candidate(candidates, match)

    # -----------------------------------------------------
    # 2) Hex / hexadecimal
    # Compact:
    # 54484520534f43204445544543544544
    #
    # Byte separated:
    # 54 48 45 20 53 4f 43
    #
    # Important: pure hex can look like Base64, so it is extracted before
    # Base64 and later validated by the pipeline guardrail.
    # -----------------------------------------------------
    hex_patterns = [
        r"(?<![A-Fa-f0-9])([A-Fa-f0-9]{12,})(?![A-Fa-f0-9])",
        r"(?<![A-Fa-f0-9])((?:[A-Fa-f0-9]{2}[\s:,\-]+){5,}[A-Fa-f0-9]{2})(?![A-Fa-f0-9])",
    ]

    for pattern in hex_patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                match = "".join(match)

            if is_likely_hex_candidate(match):
                _add_candidate(candidates, match)

    # -----------------------------------------------------
    # 3) Morse-like
    # Example:
    # .... . .-.. .-.. --- / .-- --- .-. .-.. -..
    #
    # At least 7 chars to avoid picking random punctuation.
    # -----------------------------------------------------
    morse_patterns = [
        r"(?<![A-Za-z0-9])([.\-/ ]{7,})(?![A-Za-z0-9])",
    ]

    for pattern in morse_patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                match = "".join(match)

            if is_morse_candidate(match):
                _add_candidate(candidates, match)

    # -----------------------------------------------------
    # 4) Base32-like with optional padding.
    # Example:
    # JBSWY3DPEBLW64TMMQ======
    # -----------------------------------------------------
    base32_patterns = [
        r"(?<![A-Z2-7=])([A-Z2-7]{10,}={0,8})(?![A-Z2-7=])",
    ]

    for pattern in base32_patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                match = "".join(match)

            _add_candidate(candidates, match)

    # -----------------------------------------------------
    # 5) Base64-like with optional padding.
    # Example:
    # SGVsbG8gV29ybGQ=
    # -----------------------------------------------------
    base64_patterns = [
        r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{12,}={0,4})(?![A-Za-z0-9+/=])",
    ]

    for pattern in base64_patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                match = "".join(match)

            _add_candidate(candidates, match)

    # Sort strongest / most complete first.
    candidates = sorted(candidates, key=candidate_score, reverse=True)

    return candidates


def regex_fallback_extract(text: str) -> str:
    """
    Fallback extractor if LLM is unavailable or fails.
    """

    candidates = regex_candidate_extract(text)

    if not candidates:
        return ""

    return max(candidates, key=candidate_score).strip()


# ==============================
# LLM Client
# ==============================

def _get_azure_client():
    """
    Create Azure OpenAI client if credentials exist.

    Required .env:
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_DEPLOYMENT

    Optional:
    AZURE_OPENAI_API_VERSION
    """

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip() or "gpt-4o"
    api_version = os.getenv(
        "AZURE_OPENAI_API_VERSION",
        "2024-05-01-preview"
    ).strip()

    if not endpoint or not api_key or not deployment:
        return None

    try:
        from langchain_openai import AzureChatOpenAI  # type: ignore
        from pydantic import SecretStr

        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=SecretStr(api_key),
            azure_deployment=deployment,
            api_version=api_version,
            temperature=0,
        )

    except Exception:
        return None


def _clean_json_response(content: str) -> str:
    """
    Remove accidental markdown fences from LLM JSON output.
    """

    content = str(content or "").strip()

    if "```json" in content:
        return content.split("```json", 1)[1].split("```", 1)[0].strip()

    if "```" in content:
        return content.split("```", 1)[1].split("```", 1)[0].strip()

    return content


# ==============================
# LLM Candidate Selection
# ==============================

def choose_candidate_with_llm(candidates: List[str]) -> Dict[str, Any]:
    """
    Ask the LLM to choose one token from regex-extracted candidates only.

    The LLM does not see the full suspicious text.
    The LLM does not classify.
    The LLM does not decode.
    """

    safe_candidates = [
        normalize_candidate(candidate)
        for candidate in candidates
        if normalize_candidate(candidate)
    ]

    if not safe_candidates:
        return {
            "cypher_text": "",
            "reason": "No candidate tokens were available.",
            "confidence": 0.0,
        }

    llm = _get_azure_client()

    if llm is None:
        return {
            "cypher_text": "",
            "reason": "LLM credentials are missing; regex fallback used.",
            "confidence": 0.0,
        }

    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        system_prompt = """
You are a neutral token selector inside CRYPTO AI.

Task:
Choose exactly one candidate from the provided candidate list.

Rules:
- Return exactly one candidate from the list.
- Do not transform the selected candidate.
- Preserve all characters exactly, including equals signs, dots, dashes, slashes, spaces, and numeric separators.
- Do not classify the cipher type.
- Do not decode the cipher.
- Return raw JSON only.
- confidence must be a numeric value between 0 and 1 only, for example 0.9.
- Do not return words like high, medium, or low.

JSON format:
{
  "cypher_text": "...",
  "reason": "...",
  "confidence": 0.9
}
"""

        payload = {
            "instruction": "Select the most complete encoded-looking token from this list.",
            "candidates": safe_candidates,
        }

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])

        parsed = json.loads(_clean_json_response(response.content))

        selected = normalize_candidate(parsed.get("cypher_text", ""))

        # Safety: LLM must choose from the exact candidate list only.
        if selected not in safe_candidates:
            selected = regex_fallback_extract(" ".join(safe_candidates))

        return {
            "cypher_text": selected,
            "reason": parsed.get(
                "reason",
                "LLM selected the most likely token from regex candidates."
            ),
            "confidence": parse_llm_confidence(parsed.get("confidence", 0.8)),
        }

    except Exception as e:
        return {
            "cypher_text": "",
            "reason": f"LLM selection failed: {str(e)}",
            "confidence": 0.0,
        }


# ==============================
# Main Extractor
# ==============================

def extract_cypher_text(user_text: str) -> Dict[str, Any]:
    """
    Extract only cypher_text from messy input.

    Returns:
    {
        "original_text": "...",
        "cypher_text": "...",
        "reason": "...",
        "confidence": 0.0-1.0,
        "source": "llm" | "fallback" | "none",
        "candidates": [...]
    }
    """

    original_text = str(user_text or "").strip()

    if not original_text:
        return {
            "original_text": "",
            "cypher_text": "",
            "reason": "No input text was provided.",
            "confidence": 0.0,
            "source": "none",
            "candidates": [],
        }

    candidates = regex_candidate_extract(original_text)

    if not candidates:
        return {
            "original_text": original_text,
            "cypher_text": "",
            "reason": (
                "No supported encoded token candidate was found. "
                "Supported forms include Base64, Base32, Morse, Hex, and ASCII decimal codes."
            ),
            "confidence": 0.0,
            "source": "none",
            "candidates": [],
        }

    llm_result = choose_candidate_with_llm(candidates)
    selected = normalize_candidate(llm_result.get("cypher_text", ""))

    if selected:
        return {
            "original_text": original_text,
            "cypher_text": selected,
            "reason": (
                "LLM selected cypher_text from regex candidates. "
                + str(llm_result.get("reason", ""))
            ),
            "confidence": parse_llm_confidence(
                llm_result.get("confidence", 0.8),
                default=0.8,
            ),
            "source": "llm",
            "candidates": candidates,
        }

    fallback = regex_fallback_extract(original_text)

    return {
        "original_text": original_text,
        "cypher_text": fallback,
        "reason": llm_result.get("reason", "Regex fallback used."),
        "confidence": 0.45 if fallback else 0.0,
        "source": "fallback" if fallback else "none",
        "candidates": candidates,
    }
