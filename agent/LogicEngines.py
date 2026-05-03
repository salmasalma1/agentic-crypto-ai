# agent/LogicEngines.py

"""
LogicEngines.py — CRYPTO AI Agent Logic Engines

Role inside CRYPTO AI:
1. LLM classifies the user command encoding type: base64 / base32 / morse / none.
2. Decoder tools perform the actual deterministic decoding.
3. General chat only understands normal commands; it does not decode.
4. This file is compatible with running from:
   - root app.py
   - agent/Orchestrator.py directly
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

from pydantic import SecretStr
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

try:
    from agent.state import AgentState
    from agent.decoders_utils import (
        decode_base64,
        decode_base32,
        decode_morse,
        DECODING_TOOLS
    )
except ImportError:
    from state import AgentState
    from decoders_utils import (
        decode_base64,
        decode_base32,
        decode_morse,
        DECODING_TOOLS
    )


# ==============================
# Azure / LLM Configuration
# ==============================

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o").strip()
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview").strip()


def _has_llm_credentials() -> bool:
    return bool(azure_endpoint and azure_api_key and azure_deployment)


def _build_base_llm():
    """
    Builds the base AzureChatOpenAI client.
    Returns None if credentials are missing, so the Streamlit app will not crash.
    """

    if not _has_llm_credentials():
        return None

    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=SecretStr(azure_api_key),
        azure_deployment=azure_deployment,
        api_version=azure_api_version,
        temperature=0,
    )


_base_llm = _build_base_llm()

if _base_llm is not None:
    # LLM with tools bound.
    # This instance is used only inside transformation_engine.
    InferenceUnit = _base_llm.bind_tools(
        DECODING_TOOLS,
        tool_choice="required"
    )

    # Classifier has no tools bound.
    # It only classifies type; it does not decode.
    ClassifierUnit = _base_llm
else:
    InferenceUnit = None
    ClassifierUnit = None


# ==============================
# Tool Dispatch Map
# ==============================

_TOOL_DISPATCH = {
    "decode_base64": decode_base64,
    "decode_base32": decode_base32,
    "decode_morse": decode_morse,
}

_TYPE_TO_TOOL_NAME = {
    "base64": "decode_base64",
    "base32": "decode_base32",
    "morse": "decode_morse",
}


# ==============================
# Helper Functions
# ==============================

def _clean_json_response(content: str) -> str:
    """
    Removes accidental markdown wrappers around JSON.
    """

    content = str(content).strip()

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return content


def _safe_json_loads(content: str) -> dict:
    """
    Safely parses JSON.
    """

    try:
        return json.loads(content)
    except Exception:
        return {
            "encryption_type": "none",
            "reason": "The classifier response was not valid JSON."
        }


def _invoke_tool_directly(tool_name: str, cipher_text: str):
    """
    Direct deterministic tool execution fallback.
    This guarantees decoding still works even if LLM tool-calling fails.
    """

    tool_fn = _TOOL_DISPATCH.get(tool_name)

    if tool_fn is None:
        return f"Error: Unknown tool '{tool_name}'."

    try:
        # Works with @tool-decorated LangChain tools.
        return tool_fn.invoke({"data": cipher_text})
    except Exception:
        try:
            # Fallback if your tool schema uses another param name.
            return tool_fn.invoke({"morse_code": cipher_text})
        except Exception:
            try:
                # Fallback if function is not @tool-decorated.
                return tool_fn(cipher_text)
            except Exception as e:
                return f"Error: Tool execution failed: {str(e)}"


def _extract_clean_decoded_text(raw_text: str) -> str:
    """
    Keeps the decoded command clean for CRYPTO AI.
    """

    text = str(raw_text).strip()

    prefixes = [
        "The decoded text is:",
        "Decoded text:",
        "Decoded Text:",
        "Result:",
        "The result is:",
    ]

    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    return text


# ==============================
# NODE 1 — Heuristic Analyzer
# ==============================

def heuristic_analyzer_engine(state: AgentState):
    """
    LLM role:
    Detect whether the user command is Base64, Base32, Morse, or normal text.

    Important:
    This node classifies only.
    It does not decode.
    """

    user_text = state.get("cypher_text", "")

    if not str(user_text).strip():
        return {
            "encryption_type": "none",
            "reason": "No user command was provided."
        }

    # If LLM credentials are missing, use a simple rule-based fallback.
    if ClassifierUnit is None:
        stripped = str(user_text).strip()

        if all(ch in ".-/ " for ch in stripped) and any(ch in ".-" for ch in stripped):
            return {
                "encryption_type": "morse",
                "reason": "Fallback classifier detected Morse-like symbols."
            }

        return {
            "encryption_type": "none",
            "reason": "LLM credentials missing. Fallback classifier used."
        }

    system_msg = (
        "You are the Cipher Type Classifier inside CRYPTO AI. "
        "Your role is to examine the provided user command and identify its encoding format. "
        "Respond with raw JSON only. "
        "The JSON must contain exactly two keys: "
        "'encryption_type' and 'reason'. "
        "Allowed values for 'encryption_type' are: 'base64', 'base32', 'morse', 'none'. "
        "If the command is normal cryptocurrency analysis text, return 'none'."
    )

    try:
        safe_user_msg = f"Classify this command format only: {user_text}"

        analysis_raw = ClassifierUnit.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=safe_user_msg),
        ])

        content = _clean_json_response(analysis_raw.content)
        analysis_dict = _safe_json_loads(content)

        encryption_type = analysis_dict.get("encryption_type", "none").lower()

        if encryption_type not in ["base64", "base32", "morse", "none"]:
            encryption_type = "none"

        return {
            "encryption_type": encryption_type,
            "reason": analysis_dict.get("reason", "Classified by CRYPTO AI Cipher Agent.")
        }

    except Exception as e:
        return {
            "encryption_type": "none",
            "reason": f"Classifier error: {str(e)}"
        }


# ==============================
# NODE 2 — Transformation Engine
# ==============================

def transformation_engine(state: AgentState):
    """
    Tool role:
    Performs actual decoding using deterministic decoder tools.

    LLM role here:
    It may be used to call the tool through LangChain tool-calling.
    If tool-calling fails, we execute the correct Python tool directly.

    This makes the system reliable and still agentic.
    """

    cipher_type = state.get("encryption_type", "none")
    cipher_text = state.get("cypher_text", "")

    if cipher_type not in _TYPE_TO_TOOL_NAME:
        return {
            "plain_text": None,
            "is_resolved": False
        }

    expected_tool_name = _TYPE_TO_TOOL_NAME[cipher_type]

    # ==============================
    # Path A: LLM tool-calling path
    # ==============================

    if InferenceUnit is not None:
        try:
            system_prompt = (
                "You are a CRYPTO AI data processing assistant. "
                "Use the available tool that matches the input format and return the tool result."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=cipher_text),
            ]

            ai_response = InferenceUnit.invoke(messages)

            if ai_response.tool_calls:
                tool_results = []

                for tool_call in ai_response.tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id")

                    tool_fn = _TOOL_DISPATCH.get(tool_name)

                    if tool_fn is None:
                        result = f"Error: Unknown tool '{tool_name}'."
                    else:
                        try:
                            result = tool_fn.invoke(tool_args)
                        except Exception as tool_error:
                            result = f"Error: Tool call failed: {str(tool_error)}"

                    tool_results.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id
                        )
                    )

                # Use actual tool output as the decoded text.
                decoded_text = tool_results[0].content if tool_results else ""

                # Optional: let the base LLM format it, but keep the raw tool result if formatting fails.
                try:
                    summary_prompt = (
                        "Present the tool result clearly as a decoded command only. "
                        "Do not add unrelated explanation."
                    )

                    messages_with_result = (
                        [SystemMessage(content=summary_prompt)]
                        + messages[1:]
                        + [ai_response]
                        + tool_results
                    )

                    final_response = _base_llm.invoke(messages_with_result)
                    formatted_text = final_response.content

                    if formatted_text:
                        decoded_text = formatted_text

                except Exception:
                    pass

                decoded_text = _extract_clean_decoded_text(decoded_text)
                has_error = "Error" in str(decoded_text)

                return {
                    "plain_text": decoded_text,
                    "is_resolved": not has_error
                }

        except Exception:
            # Continue to deterministic fallback below.
            pass

    # ==============================
    # Path B: Deterministic fallback
    # ==============================

    result = _invoke_tool_directly(expected_tool_name, cipher_text)
    decoded_text = _extract_clean_decoded_text(result)
    has_error = "Error" in str(decoded_text)

    return {
        "plain_text": decoded_text,
        "is_resolved": not has_error
    }


# ==============================
# NODE 3 — General Chat Engine
# ==============================

def general_chat_engine(state: AgentState):
    """
    Handles normal text commands.

    In CRYPTO AI:
    This node does not decode.
    It only interprets the user's normal crypto analysis command.
    """

    user_text = state.get("cypher_text", "")

    if not str(user_text).strip():
        return {
            "messages": [
                AIMessage(
                    content="Default command used: Analyze the cryptocurrency market and generate a risk report."
                )
            ]
        }

    if _base_llm is None:
        return {
            "messages": [
                AIMessage(
                    content=f"Normal command detected and passed to CRYPTO AI pipeline: {user_text}"
                )
            ]
        }

    try:
        system_prompt = (
            "You are CRYPTO AI's command understanding layer. "
            "Briefly interpret the user's cryptocurrency analysis command. "
            "Do not perform decoding here."
        )

        safe_user_msg = f"The user command is: {user_text}"

        response = _base_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=safe_user_msg),
        ])

        return {"messages": [response]}

    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=f"Command understanding fallback used: {str(e)}"
                )
            ]
        }