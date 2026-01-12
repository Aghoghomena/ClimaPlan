import json
from typing import Any, Dict
from typing import Optional, Dict, Any, List
from langchain.messages import ToolMessage

def _tool_content_to_dict(content: Any) -> Dict[str, Any]:
    if content is None:
        return {}
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        content = content.strip()
        if not content:
            return {}
        # Try JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Sometimes tools return plain text; wrap it
            return {"text": content}
    # Fallback
    return {"value": content}


def _get_trailing_tool_messages(messages: List[Any]) -> List[ToolMessage]:
    """ToolNode may append multiple ToolMessages; we grab the consecutive block at the end."""
    trailing = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            trailing.append(msg)
        else:
            break
    return list(reversed(trailing))


def inc_count(counts: dict[str, int], key: str) -> dict[str, int]:
    if not key: return {}
    count = counts.get(key, 0) + 1
    return {key : count }
