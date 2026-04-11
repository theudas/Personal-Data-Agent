from __future__ import annotations

import json
from typing import Any

from .errors import ValidationError


TOOL_SCHEMA: dict[str, dict[str, type]] = {
    "list_notes": {"directory": str},
    "semantic_search": {"query": str},
    "read_note": {"filename": str},
    "write_note": {"filename": str, "content": str},
    "append_note": {"filename": str, "content": str},
    "delete_note": {"filename": str},
    "extract_highlights": {"filename": str, "focus": str},
}


def parse_tool_args(raw_args: Any) -> dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Tool arguments are not valid JSON: {exc}") from exc
    raise ValidationError(f"Unsupported tool argument type: {type(raw_args).__name__}")


def validate_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name not in TOOL_SCHEMA:
        raise ValidationError(f"Unknown tool: {tool_name}")

    expected = TOOL_SCHEMA[tool_name]
    missing = [k for k in expected.keys() if k not in args]
    if missing:
        raise ValidationError(f"Missing required args for {tool_name}: {missing}")

    normalized: dict[str, Any] = {}
    for key, typ in expected.items():
        value = args[key]
        if not isinstance(value, typ):
            raise ValidationError(
                f"Arg `{key}` for `{tool_name}` should be {typ.__name__}, got {type(value).__name__}"
            )
        normalized[key] = value

    return normalized
