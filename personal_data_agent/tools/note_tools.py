from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from personal_data_agent.runtime.errors import SecurityError, ToolExecutionError
from personal_data_agent.vector_store.index import ALLOWED_SUFFIXES, FileVectorIndex


def get_openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_notes",
                "description": "列出目录下所有笔记及最后修改时间",
                "parameters": {
                    "type": "object",
                    "properties": {"directory": {"type": "string"}},
                    "required": ["directory"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": "语义检索笔记切片，返回最相关片段及文件名",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_note",
                "description": "读取完整笔记",
                "parameters": {
                    "type": "object",
                    "properties": {"filename": {"type": "string"}},
                    "required": ["filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_note",
                "description": "新建或覆盖笔记",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["filename", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "append_note",
                "description": "追加写入笔记末尾",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["filename", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_note",
                "description": "删除指定笔记文件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                    },
                    "required": ["filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_highlights",
                "description": "按 focus 提取关键部分并结构化输出",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "focus": {"type": "string"},
                    },
                    "required": ["filename", "focus"],
                },
            },
        },
    ]


@dataclass
class ToolContext:
    notes_root: Path
    vector_index: FileVectorIndex
    llm_client: OpenAI
    llm_model: str
    top_k: int = 5


class ToolRegistry:
    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "list_notes": self.list_notes,
            "semantic_search": self.semantic_search,
            "read_note": self.read_note,
            "write_note": self.write_note,
            "append_note": self.append_note,
            "delete_note": self.delete_note,
            "extract_highlights": self.extract_highlights,
        }

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self._handlers:
            raise ToolExecutionError(f"Unknown tool: {tool_name}", retriable=False)
        return self._handlers[tool_name](**args)

    def _resolve_in_root(self, rel_path: str, must_exist: bool = False) -> Path:
        root = self.ctx.notes_root.resolve()
        candidate = Path(rel_path)
        target = candidate if candidate.is_absolute() else (root / candidate)
        target = target.resolve()

        try:
            target.relative_to(root)
        except ValueError:
            raise SecurityError("Access denied: path escapes selected notes directory")

        if must_exist and not target.exists():
            raise ToolExecutionError(f"Path not found: {rel_path}", retriable=False)

        return target

    def _resolve_note_file(self, filename: str, must_exist: bool = False) -> Path:
        path = self._resolve_in_root(filename, must_exist=must_exist)
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise SecurityError("Only .txt / .md / .markdown files are supported")
        return path

    def list_notes(self, directory: str) -> dict[str, Any]:
        target_dir = self._resolve_in_root(directory, must_exist=True)
        if not target_dir.is_dir():
            raise ToolExecutionError(f"Not a directory: {directory}", retriable=False)

        items: list[dict[str, str]] = []
        for f in target_dir.rglob("*"):
            if not f.is_file():
                continue
            if ".personal_data_agent" in f.parts:
                continue
            if f.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            stat = f.stat()
            items.append(
                {
                    "filename": str(f.relative_to(self.ctx.notes_root)),
                    "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                }
            )

        items.sort(key=lambda x: x["mtime"], reverse=True)
        return {"directory": str(target_dir.relative_to(self.ctx.notes_root)), "count": len(items), "files": items}

    def semantic_search(self, query: str) -> dict[str, Any]:
        hits = self.ctx.vector_index.search(query=query, top_k=self.ctx.top_k)
        return {
            "query": query,
            "count": len(hits),
            "results": [
                {
                    "filename": h.filename,
                    "score": round(h.score, 6),
                    "text": h.chunk_text,
                }
                for h in hits
            ],
        }

    def read_note(self, filename: str) -> dict[str, Any]:
        file_path = self._resolve_note_file(filename, must_exist=True)
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        return {
            "filename": str(file_path.relative_to(self.ctx.notes_root)),
            "content": content,
            "length": len(content),
        }

    def write_note(self, filename: str, content: str) -> dict[str, Any]:
        file_path = self._resolve_note_file(filename, must_exist=False)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(file_path)

        return {
            "filename": str(file_path.relative_to(self.ctx.notes_root)),
            "written_chars": len(content),
            "mode": "overwrite",
        }

    def append_note(self, filename: str, content: str) -> dict[str, Any]:
        file_path = self._resolve_note_file(filename, must_exist=False)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("a", encoding="utf-8") as f:
            f.write(content)

        return {
            "filename": str(file_path.relative_to(self.ctx.notes_root)),
            "appended_chars": len(content),
            "mode": "append",
        }

    def delete_note(self, filename: str) -> dict[str, Any]:
        file_path = self._resolve_note_file(filename, must_exist=True)
        if not file_path.is_file():
            raise ToolExecutionError(f"Not a file: {filename}", retriable=False)

        file_path.unlink()
        return {
            "filename": str(file_path.relative_to(self.ctx.notes_root)),
            "deleted": True,
        }

    def extract_highlights(self, filename: str, focus: str) -> dict[str, Any]:
        file_path = self._resolve_note_file(filename, must_exist=True)
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

        prompt = (
            "你是一个笔记分析助手。请根据给定 focus 提取关键内容，并输出 JSON。"
            "JSON字段固定为: focus, summary, key_points(list), action_items(list), risks(list), quotes(list)。"
            "不要输出任何额外解释。"
        )

        user_input = (
            f"文件名: {file_path.name}\n"
            f"focus: {focus}\n"
            f"正文:\n{content[:24000]}"
        )

        try:
            resp = self.ctx.llm_client.chat.completions.create(
                model=self.ctx.llm_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.2,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            raise ToolExecutionError(f"extract_highlights failed: {exc}", retriable=True) from exc

        parsed = _try_parse_json(raw)
        if parsed is None:
            parsed = {
                "focus": focus,
                "summary": raw,
                "key_points": [],
                "action_items": [],
                "risks": [],
                "quotes": [],
            }

        return {
            "filename": str(file_path.relative_to(self.ctx.notes_root)),
            "highlights": parsed,
        }


def _try_parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    try:
        val = json.loads(text)
        if isinstance(val, dict):
            return val
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        snippet = m.group(0)
        try:
            val = json.loads(snippet)
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            return None

    return None
