# agent/decoders_utils.py

"""
decoders_utils.py — Cipher decoding tools

Each function is decorated with @tool so LangChain can bind it to the LLM.
The actual decoding is performed by deterministic Python functions.

Important for the final CRYPTO AI pipeline:
- The LLM should NOT classify the cipher type.
- The LLM should NOT decode the cipher.
- The SVM predicts the cipher type.
- decode_by_type() decodes using the deterministic decoder selected by the SVM prediction.

Supported deterministic decoders:
- Base64
- Base32
- Morse
- Hex / Hexadecimal
- ASCII decimal codes
"""

from __future__ import annotations

import base64
import re
import sys
from langchain_core.tools import tool


# Safe UTF-8 stdout configuration
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# ==============================
# Base64 Decoder
# ==============================

@tool
def decode_base64(data: str) -> str:
    """
    Decodes a Base64-encoded string and returns the plain-text result.
    Use this tool whenever the cipher type is identified as Base64.
    """
    try:
        clean_data = re.sub(r"[^A-Za-z0-9+/=]", "", str(data or "").strip())

        if not clean_data:
            return "Error decoding Base64: empty input."

        # Keep valid content, normalize padding.
        clean_data = clean_data.rstrip("=")
        clean_data += "=" * (-len(clean_data) % 4)

        decoded_bytes = base64.b64decode(clean_data, validate=True)

        return decoded_bytes.decode("utf-8", errors="replace")

    except Exception as e:
        return f"Error decoding Base64: {e}"


# ==============================
# Base32 Decoder
# ==============================

@tool
def decode_base32(data: str) -> str:
    """
    Decodes a Base32-encoded string and returns the plain-text result.
    Use this tool whenever the cipher type is identified as Base32.
    """
    try:
        clean_data = re.sub(r"[^A-Z2-7=]", "", str(data or "").upper().strip())

        if not clean_data:
            return "Error decoding Base32: empty input."

        clean_data = clean_data.rstrip("=")
        clean_data += "=" * (-len(clean_data) % 8)

        decoded_bytes = base64.b32decode(clean_data, casefold=True)

        return decoded_bytes.decode("utf-8", errors="replace")

    except Exception as e:
        return f"Error decoding Base32: {e}"


# ==============================
# Morse Decoder
# ==============================

@tool
def decode_morse(morse_code: str) -> str:
    """
    Decodes a Morse code string and returns the plain-text result.
    Words should preferably be separated by ' / ' and letters by spaces.
    """
    try:
        morse_dict = {
            ".-": "A", "-...": "B", "-.-.": "C", "-..": "D",
            ".": "E", "..-.": "F", "--.": "G", "....": "H",
            "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
            "--": "M", "-.": "N", "---": "O", ".--.": "P",
            "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
            "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
            "-.--": "Y", "--..": "Z",
            "-----": "0", ".----": "1", "..---": "2",
            "...--": "3", "....-": "4", ".....": "5",
            "-....": "6", "--...": "7", "---..": "8", "----.": "9",
        }

        morse_code = str(morse_code or "").strip()

        if not morse_code:
            return "Error decoding Morse: empty input."

        # Normalize common word separators.
        morse_code = re.sub(r"\s*/\s*", " / ", morse_code)

        words = morse_code.split(" / ")
        decoded_words = []

        for word in words:
            letters = word.split()
            decoded_word = "".join(
                morse_dict.get(letter, "?")
                for letter in letters
            )
            decoded_words.append(decoded_word)

        return " ".join(decoded_words)

    except Exception as e:
        return f"Error decoding Morse: {e}"


# ==============================
# Hex Decoder
# ==============================

@tool
def decode_hex(hex_data: str) -> str:
    """
    Decodes a Hex/Hexadecimal encoded string and returns the plain-text result.
    Supports formats such as:
    - 48656c6c6f20576f726c64
    - 48 65 6c 6c 6f 20 57 6f 72 6c 64
    - 0x48 0x65 0x6c 0x6c 0x6f
    """
    try:
        raw = str(hex_data or "").strip()

        if not raw:
            return "Error decoding Hex: empty input."

        # Remove common 0x prefixes, spaces, commas, dashes, colons, and other separators.
        cleaned = re.sub(r"0x", "", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^0-9A-Fa-f]", "", cleaned)

        if not cleaned:
            return "Error decoding Hex: no hexadecimal characters found."

        if len(cleaned) % 2 != 0:
            return "Error decoding Hex: hexadecimal input must contain an even number of characters."

        decoded_bytes = bytes.fromhex(cleaned)

        return decoded_bytes.decode("utf-8", errors="replace")

    except Exception as e:
        return f"Error decoding Hex: {e}"


# ==============================
# ASCII Decimal Decoder
# ==============================

@tool
def decode_ascii(ascii_data: str) -> str:
    """
    Decodes ASCII decimal codes and returns the plain-text result.
    Supports formats such as:
    - 72 101 108 108 111
    - 72,101,108,108,111
    - ASCII: 72-101-108-108-111
    """
    try:
        raw = str(ascii_data or "").strip()

        if not raw:
            return "Error decoding ASCII: empty input."

        numbers = re.findall(r"\d+", raw)

        if not numbers:
            return "Error decoding ASCII: no decimal ASCII codes found."

        chars = []

        for num in numbers:
            value = int(num)

            # Standard ASCII is 0-127. We allow 0-255 for common byte-style datasets,
            # but reject larger values because they are probably not ASCII codes.
            if value < 0 or value > 255:
                return f"Error decoding ASCII: value out of byte range: {value}"

            chars.append(chr(value))

        return "".join(chars)

    except Exception as e:
        return f"Error decoding ASCII: {e}"


# Convenience list used by LogicEngines.py to bind all decoding tools at once.
DECODING_TOOLS = [
    decode_base64,
    decode_base32,
    decode_morse,
    decode_hex,
    decode_ascii,
]


def decode_by_type(cypher_text: str, cipher_type: str) -> dict:
    """
    Decode cypher_text using the deterministic decoder selected by the SVM prediction.

    This function is used by pipeline_runner.py after SVM predicts the cipher type.

    Important:
    - The LLM does not decode here.
    - The SVM predicts the cipher type.
    - This function only calls the matching deterministic decoder.
    """

    cypher_text = str(cypher_text or "").strip()

    cipher_type_normalized = (
        str(cipher_type or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )

    if not cypher_text:
        return {
            "success": False,
            "decoded_text": "",
            "message": "No cypher_text was provided for decoding.",
        }

    try:
        decoder_name = ""

        if "base64" in cipher_type_normalized:
            decoder_name = "Base64"
            decoded = decode_base64.invoke({"data": cypher_text})

        elif "base32" in cipher_type_normalized:
            decoder_name = "Base32"
            decoded = decode_base32.invoke({"data": cypher_text})

        elif "morse" in cipher_type_normalized:
            decoder_name = "Morse"
            decoded = decode_morse.invoke({"morse_code": cypher_text})

        elif "hexadecimal" in cipher_type_normalized or "hex" in cipher_type_normalized:
            decoder_name = "Hex"
            decoded = decode_hex.invoke({"hex_data": cypher_text})

        elif (
            "ascii" in cipher_type_normalized
            or "ascll" in cipher_type_normalized  # protects against common typo
            or "decimalascii" in cipher_type_normalized
            or "asciidecimal" in cipher_type_normalized
        ):
            decoder_name = "ASCII"
            decoded = decode_ascii.invoke({"ascii_data": cypher_text})

        else:
            return {
                "success": False,
                "decoded_text": "",
                "message": f"Unsupported cipher type for decoding: {cipher_type}",
            }

        decoded = str(decoded).strip()

        if decoded.lower().startswith("error"):
            return {
                "success": False,
                "decoded_text": decoded,
                "message": decoded,
                "decoder_used": decoder_name,
            }

        return {
            "success": True,
            "decoded_text": decoded,
            "message": f"Decoded successfully using deterministic {decoder_name} decoder.",
            "decoder_used": decoder_name,
        }

    except Exception as e:
        return {
            "success": False,
            "decoded_text": "",
            "message": f"Decoding failed: {str(e)}",
        }
