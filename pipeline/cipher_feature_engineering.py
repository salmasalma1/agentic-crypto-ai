"""Feature engineering for encoded/cipher text.

The LLM may extract only `cypher_text`. This module turns that text into
numerical features used by DBSCAN -> FCM -> SVM-RBF.

Updated cipher families:
- Base64
- Base32
- Morse
- Hex / Hexadecimal
- ASCII decimal codes
"""

import math
import re
from collections import Counter
import pandas as pd


def calculate_entropy(text: str) -> float:
    text = str(text or "")
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return round(float(entropy), 4)


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else round(float(numerator / denominator), 4)


def is_valid_text_value(value) -> bool:
    value = str(value or "").strip()
    return bool(value and value.lower() not in {"nan", "none", "null", "unknown"})


def find_existing_column(df: pd.DataFrame, candidates: list):
    normalized = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def expand_encoded_dataset(df: pd.DataFrame, max_rows_per_type: int = 700) -> pd.DataFrame:
    """Expand wide encoded columns into compact one-row-per-cipher samples.

    This implementation is deliberately compact and fast: it does not copy every
    original long encoded column into every expanded row. It keeps only the fields
    needed for the ML pipeline and dashboard.
    """
    df = df.copy()

    cipher_columns = {
        "Base64": [
            "base64_encoded", "base64", "Base64_encoded", "BASE64_encoded",
            "base64_text", "Base64", "BASE64",
        ],
        "Base32": [
            "base32_encoded", "base32", "Base32_encoded", "BASE32_encoded",
            "base32_text", "Base32", "BASE32",
        ],
        "Morse": [
            "morse_encoded", "morse", "Morse_encoded", "MORSE_encoded",
            "morse_text", "Morse", "MORSE",
        ],
        "Hex": [
            "hex_encoded", "hex", "Hex_encoded", "HEX_encoded",
            "hexadecimal", "hexadecimal_encoded", "Hex", "HEX",
        ],
        "ASCII": [
            "ascii_encoded", "ascii", "Ascii_encoded", "ASCII_encoded",
            "ascii_decimal", "ascii_codes", "decimal_ascii", "ASCII",
        ],
    }

    frames = []

    # Keep live/LLM-extracted rows.
    if "cypher_text" in df.columns:
        live_mask = df["cypher_text"].astype(str).map(is_valid_text_value)
        if live_mask.any():
            live = pd.DataFrame({
                "cypher_text": df.loc[live_mask, "cypher_text"].astype(str).str.strip(),
                "source_cipher_type": "Unknown",
                "source_cipher_column": "cypher_text",
                "source_row_id": df.index[live_mask],
                "is_llm_extracted": True,
            })
            if "plain_text" in df.columns:
                live["plain_text"] = df.loc[live_mask, "plain_text"].astype(str).values
            frames.append(live)

    for cipher_type, candidates in cipher_columns.items():
        col = find_existing_column(df, candidates)
        if col is None:
            continue
        valid_mask = df[col].astype(str).map(is_valid_text_value)
        if not valid_mask.any():
            continue
        sample = df.loc[
            valid_mask,
            [col] + (["plain_text"] if "plain_text" in df.columns else []),
        ].copy()
        sample = sample.sample(n=min(len(sample), max_rows_per_type), random_state=42)
        frame = pd.DataFrame({
            "cypher_text": sample[col].astype(str).str.strip(),
            "source_cipher_type": cipher_type,
            "source_cipher_column": col,
            "source_row_id": sample.index,
            "is_llm_extracted": False,
        })
        if "plain_text" in sample.columns:
            frame["plain_text"] = sample["plain_text"].astype(str).values
        frames.append(frame)

    if not frames:
        return pd.DataFrame({
            "cypher_text": df.get("cypher_text", pd.Series([""] * len(df))).astype(str),
            "source_cipher_type": "Unknown",
            "source_cipher_column": "",
            "source_row_id": df.index,
            "is_llm_extracted": False,
        }).reset_index(drop=True)

    expanded_df = pd.concat(frames, ignore_index=True)
    return expanded_df.reset_index(drop=True)


def dominant_encoded_source(row: pd.Series) -> str:
    candidate_columns = [
        "cypher_text", "cipher_text",
        "Base64_encoded", "Base32_encoded", "Morse_encoded", "Hex_encoded", "ASCII_encoded",
        "base64_encoded", "base32_encoded", "morse_encoded", "hex_encoded", "ascii_encoded",
        "hex", "hexadecimal", "ascii", "ascii_decimal", "ascii_codes", "decimal_ascii",
    ]

    for col in candidate_columns:
        if col in row.index and is_valid_text_value(row.get(col, "")):
            return str(row.get(col, "")).strip()
    return ""


def count_base64_chars(text: str) -> int:
    return sum(ch.isalnum() or ch in ["+", "/", "="] for ch in str(text or ""))


def count_base32_chars(text: str) -> int:
    text = str(text or "").upper()
    return sum(("A" <= ch <= "Z") or ("2" <= ch <= "7") or ch == "=" for ch in text)


def count_morse_chars(text: str) -> int:
    return sum(ch in [".", "-", "/", " "] for ch in str(text or ""))


def count_hex_chars(text: str) -> int:
    """Counts characters that are valid in hexadecimal payloads.

    Spaces, colons, and hyphens are allowed separators for human-written hex,
    but only 0-9/A-F characters are counted as hex characters.
    """
    text = str(text or "").strip()
    return sum(ch.upper() in "0123456789ABCDEF" for ch in text)


def count_hex_separator_chars(text: str) -> int:
    return sum(ch in [" ", ":", "-", "\\n", "\\t"] for ch in str(text or ""))


def extract_ascii_numbers(text: str) -> list[int]:
    """Extracts decimal ASCII tokens from a string."""
    tokens = re.findall(r"\d+", str(text or ""))
    numbers = []
    for token in tokens:
        try:
            numbers.append(int(token))
        except Exception:
            continue
    return numbers


def count_ascii_tokens(text: str) -> int:
    return len(extract_ascii_numbers(text))


def count_valid_ascii_tokens(text: str) -> int:
    numbers = extract_ascii_numbers(text)
    # Printable ASCII range is the useful range for natural-language decoded commands.
    return sum(32 <= number <= 126 for number in numbers)


def count_ascii_separator_chars(text: str) -> int:
    return sum(ch in [" ", ",", ";", "|", ":", "-", "\\n", "\\t"] for ch in str(text or ""))


def add_cypher_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cypher_text" not in df.columns:
        df["cypher_text"] = ""

    df["cypher_text"] = df.apply(
        lambda row: str(row.get("cypher_text", "")).strip()
        if is_valid_text_value(row.get("cypher_text", ""))
        else dominant_encoded_source(row),
        axis=1,
    )

    s = df["cypher_text"].astype(str)

    df["cypher_length"] = s.apply(len)
    df["cypher_entropy"] = s.apply(calculate_entropy)
    df["cypher_digit_count"] = s.apply(lambda x: sum(ch.isdigit() for ch in x))
    df["cypher_alpha_count"] = s.apply(lambda x: sum(ch.isalpha() for ch in x))
    df["cypher_upper_count"] = s.apply(lambda x: sum(ch.isupper() for ch in x))
    df["cypher_lower_count"] = s.apply(lambda x: sum(ch.islower() for ch in x))
    df["cypher_symbol_count"] = s.apply(lambda x: sum((not ch.isalnum()) and (not ch.isspace()) for ch in x))
    df["cypher_space_count"] = s.apply(lambda x: sum(ch.isspace() for ch in x))
    df["cypher_equal_count"] = s.apply(lambda x: x.count("="))
    df["cypher_plus_slash_count"] = s.apply(lambda x: sum(ch in ["+", "/"] for ch in x))
    df["cypher_dot_dash_count"] = s.apply(lambda x: sum(ch in [".", "-"] for ch in x))

    # Original families
    df["cypher_base64_char_count"] = s.apply(count_base64_chars)
    df["cypher_base32_char_count"] = s.apply(count_base32_chars)
    df["cypher_morse_char_count"] = s.apply(count_morse_chars)

    # New families: Hex + ASCII decimal codes
    df["cypher_hex_char_count"] = s.apply(count_hex_chars)
    df["cypher_hex_separator_count"] = s.apply(count_hex_separator_chars)
    df["cypher_ascii_token_count"] = s.apply(count_ascii_tokens)
    df["cypher_ascii_valid_token_count"] = s.apply(count_valid_ascii_tokens)
    df["cypher_ascii_separator_count"] = s.apply(count_ascii_separator_chars)

    for num, den, out in [
        ("cypher_digit_count", "cypher_length", "cypher_digit_ratio"),
        ("cypher_alpha_count", "cypher_length", "cypher_alpha_ratio"),
        ("cypher_upper_count", "cypher_length", "cypher_upper_ratio"),
        ("cypher_lower_count", "cypher_length", "cypher_lower_ratio"),
        ("cypher_symbol_count", "cypher_length", "cypher_symbol_ratio"),
        ("cypher_space_count", "cypher_length", "cypher_space_ratio"),
        ("cypher_base64_char_count", "cypher_length", "cypher_base64_ratio"),
        ("cypher_base32_char_count", "cypher_length", "cypher_base32_ratio"),
        ("cypher_morse_char_count", "cypher_length", "cypher_morse_ratio"),
        ("cypher_dot_dash_count", "cypher_length", "cypher_dot_dash_ratio"),
        ("cypher_hex_char_count", "cypher_length", "cypher_hex_ratio"),
        ("cypher_hex_separator_count", "cypher_length", "cypher_hex_separator_ratio"),
        ("cypher_ascii_token_count", "cypher_length", "cypher_ascii_token_ratio"),
        ("cypher_ascii_valid_token_count", "cypher_ascii_token_count", "cypher_ascii_valid_token_ratio"),
        ("cypher_ascii_separator_count", "cypher_length", "cypher_ascii_separator_ratio"),
    ]:
        df[out] = df.apply(lambda row: safe_ratio(row[num], row[den]), axis=1)

    # Strong shape indicators. These are numeric flags useful for SVM separation.
    df["cypher_is_likely_hex_shape"] = df.apply(
        lambda row: int(
            row["cypher_length"] >= 4
            and row["cypher_hex_ratio"] >= 0.85
            and row["cypher_morse_ratio"] < 0.60
            and row["cypher_ascii_token_count"] <= 1
        ),
        axis=1,
    )

    df["cypher_is_likely_ascii_shape"] = df.apply(
        lambda row: int(
            row["cypher_ascii_token_count"] >= 2
            and row["cypher_ascii_valid_token_ratio"] >= 0.80
            and row["cypher_digit_ratio"] >= 0.45
        ),
        axis=1,
    )

    return df


def add_original_column_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    plain_col = "plain_text" if "plain_text" in df.columns else None
    if plain_col:
        text = df[plain_col].astype(str)
        df["plain_text_length"] = text.apply(len)
        df["plain_text_word_count"] = text.apply(lambda x: len(x.split()))
        df["plain_text_entropy"] = text.apply(calculate_entropy)
    return df


def add_cipher_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "record_date" not in df.columns:
        df["record_date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
    df = add_cypher_text_features(df)
    df = add_original_column_features(df)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(0 if pd.isna(median) else median)
    return df
