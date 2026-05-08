"""Compatibility bridge for older CRYPTO AI code."""
from typing import Any, Dict
from agent.cipher_text_extractor import extract_cypher_text


def run_cipher_agent(user_command: str) -> Dict[str, Any]:
    result = extract_cypher_text(user_command)
    return {
        "original_command": user_command or "",
        "decoded_command": result.get("cypher_text", "") or user_command or "",
        "encryption_type": "unknown",
        "reason": result.get("reason", "Extractor bridge used."),
        "is_decoded": False,
        "agent_reply": "Extractor bridge returned cypher_text only; decoding is handled after SVM prediction.",
    }
