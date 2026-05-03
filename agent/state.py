"""
state.py  —  Shared agent state schema
No structural changes required. Provided here for completeness.
The 'messages' field (with add_messages reducer) already supports the
ToolMessage objects that the fixed transformation_engine now produces.
"""

from typing import Annotated, TypedDict, Optional, Dict, Any
from langgraph.graph.message import add_messages


class AgentState(TypedDict):

    # Full conversation history; add_messages reducer handles appending
    messages: Annotated[list, add_messages]

    # The raw encoded text submitted by the user
    cypher_text: str

    # Final decoded plain text (populated by transformation_engine)
    plain_text: Optional[str]

    # Identified encoding type: 'base64' | 'base32' | 'morse' | 'none'
    encryption_type: Optional[str]

    # Optional metadata (padding info, key length, etc.)
    encryption_metadata: Dict[str, Any]

    # Human-readable reason from the classifier
    reason: str

    # True once decoding succeeds without errors
    is_resolved: bool
