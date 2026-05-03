# agent/Orchestrator.py

"""
Orchestrator.py — CRYPTO AI Cipher Agent Graph

Role inside the integrated CRYPTO AI project:
1. Receive a normal or encrypted analyst command.
2. Use the LLM-based classifier to detect:
   base64 / base32 / morse / none.
3. If encrypted, route to deterministic decoder tools.
4. If normal text, route to the chat / command understanding node.
5. Return the decoded/final command back to CRYPTO AI app.py.

This file is compatible with:
- Running directly: python agent/Orchestrator.py
- Importing from app.py: from agent.Orchestrator import system_core
"""

import os
from typing import cast, Dict, Any

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


# ==============================
# Safe Imports
# ==============================
# These imports allow the file to work both:
# 1. From root app.py
# 2. From inside agent/ folder

try:
    from agent.state import AgentState
    from agent.LogicEngines import (
        general_chat_engine,
        heuristic_analyzer_engine,
        transformation_engine,
    )
except ImportError:
    from state import AgentState
    from LogicEngines import (
        general_chat_engine,
        heuristic_analyzer_engine,
        transformation_engine,
    )


# ==============================
# Helper Node: Decoding Failed
# ==============================

def decoding_failed_node(state: AgentState):
    """
    Reached when transformation_engine could not decode the cipher.

    It returns a user-friendly failure message without guessing the decoded text.
    """

    from langchain_core.messages import AIMessage

    return {
        "messages": [
            AIMessage(
                content=(
                    "Decoding failed. The decoding tool did not return a valid result. "
                    "Please check that the input is a well-formed Base64, Base32, "
                    "or Morse code string and try again."
                )
            )
        ],
        "plain_text": None,
        "is_resolved": False,
    }


# ==============================
# Router 1: After Classification
# ==============================

def executive_router(state: AgentState) -> str:
    """
    Routes the command after LLM classification.

    If the command is encoded:
        heuristic_analyzer → transformation_unit

    If the command is normal text:
        heuristic_analyzer → chat_node
    """

    valid_types = ["base64", "base32", "morse"]

    if state.get("encryption_type") in valid_types:
        return "transformation_unit"

    return "chat_node"


# ==============================
# Router 2: After Transformation
# ==============================

def resolution_router(state: AgentState) -> str:
    """
    Routes the result after the decoding tool runs.

    If decoding succeeded:
        transformation_unit → END

    If decoding failed:
        transformation_unit → decoding_failed
    """

    if state.get("is_resolved"):
        return "success"

    return "failed"


# ==============================
# Build LangGraph
# ==============================

builder = StateGraph(AgentState)

builder.add_node("heuristic_analyzer", heuristic_analyzer_engine)
builder.add_node("transformation_unit", transformation_engine)
builder.add_node("chat_node", general_chat_engine)
builder.add_node("decoding_failed", decoding_failed_node)

builder.add_edge(START, "heuristic_analyzer")

builder.add_conditional_edges(
    "heuristic_analyzer",
    executive_router,
    {
        "transformation_unit": "transformation_unit",
        "chat_node": "chat_node",
    },
)

builder.add_conditional_edges(
    "transformation_unit",
    resolution_router,
    {
        "success": END,
        "failed": "decoding_failed",
    },
)

builder.add_edge("chat_node", END)
builder.add_edge("decoding_failed", END)


# ==============================
# Compile Graph
# ==============================
# Important:
# We compile without MemorySaver here so app.py can call:
# system_core.invoke(initial_state)
# without needing thread_id config.

system_core = builder.compile()


# ==============================
# Helper Function for app.py / Bridge
# ==============================

def build_initial_state(user_command: str) -> AgentState:
    """
    Builds a clean AgentState for every new command.
    Used by app.py or crypto_agent_bridge.py.
    """

    return cast(
        AgentState,
        {
            "messages": [],
            "cypher_text": user_command,
            "plain_text": None,
            "encryption_type": "none",
            "encryption_metadata": {},
            "reason": "",
            "is_resolved": False,
        },
    )


def run_orchestrator_command(user_command: str) -> Dict[str, Any]:
    """
    Runs the Cipher Agent graph once and returns the final state.

    This function is useful when integrating the Cipher Agent into CRYPTO AI.
    """

    if user_command is None:
        user_command = ""

    initial_state = build_initial_state(user_command.strip())

    try:
        final_state = system_core.invoke(initial_state)

        return {
            "success": True,
            "final_state": final_state,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "final_state": initial_state,
            "error": str(e),
        }


# ==============================
# Optional Graph Visualization
# ==============================

def save_graph_visualization(output_path: str = "graph_structure.png"):
    """
    Optional helper.
    Saves the LangGraph structure as PNG if supported.
    It is NOT executed automatically during import.
    """

    try:
        image_data = system_core.get_graph().draw_mermaid_png()

        with open(output_path, "wb") as file:
            file.write(image_data)

        print(f"Graph structure saved as {output_path}")

    except Exception:
        print("Graph visualization failed. ASCII graph:")
        print(system_core.get_graph().draw_ascii())


# ==============================
# Main CLI Loop
# ==============================

if __name__ == "__main__":

    print("CRYPTO AI Cipher Agent is live. Type 'exit' to quit.")
    print("-" * 50)

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break

        if not user_input:
            continue

        inputs = build_initial_state(user_input)

        try:
            for event in system_core.stream(inputs):
                for node_name, node_output in event.items():

                    if node_name == "heuristic_analyzer":
                        enc_type = node_output.get("encryption_type", "none")
                        reason = node_output.get("reason", "")

                        if enc_type != "none":
                            print(f"Detected: {enc_type.upper()} — {reason}")
                        else:
                            print(f"Detected: NORMAL TEXT — {reason}")

                    elif node_name == "transformation_unit":
                        if node_output.get("is_resolved"):
                            print(f"Agent decoded command: {node_output.get('plain_text')}")

                    elif node_name == "decoding_failed":
                        messages = node_output.get("messages", [])

                        if messages:
                            print(f"Agent: {messages[-1].content}")
                        else:
                            print("Agent: Decoding failed.")

                    elif node_name == "chat_node":
                        messages = node_output.get("messages", [])

                        if messages:
                            print(f"Agent: {messages[-1].content}")
                        else:
                            print("Agent: Normal command received.")

        except Exception as e:
            print(f"Agent error: {str(e)}")