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
                "description": (
                    "列出指定目录及其子目录下的所有可访问笔记文件，并返回文件名和最后修改时间。"
                    "适用于确认某个文件是否存在、查看某个目录里有哪些笔记。"
                    "不适用于按主题搜索内容，也不适用于读取文件正文。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "相对于笔记根目录的目录路径。根目录可传 `.`。",
                        }
                    },
                    "required": ["directory"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": (
                    "按主题、问题或关键词对全部笔记做语义检索，返回最相关的文本片段及来源文件。"
                    "适用于用户没有给出明确文件名，只描述了概念、问题或主题的场景。"
                    "如果已经知道准确文件名并需要全文，应优先使用 read_note。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要检索的主题、问题或关键词，使用自然语言描述，不要填文件路径。",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_note",
                "description": (
                    "读取单个笔记文件的完整正文。"
                    "适用于已知准确文件名，需要查看全文、总结或回答基于该文件的问题。"
                    "不适用于仅判断文件是否存在或按主题找内容；这些场景优先使用 list_notes 或 semantic_search。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "相对于笔记根目录的文件路径，例如 `DPO总结.md` 或 `rl/notes.md`。应是文件，不是目录。",
                        }
                    },
                    "required": ["filename"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_note",
                "description": (
                    "创建新笔记或用新内容完整覆盖已有笔记。"
                    "只有在用户明确要求新建、保存或覆盖文件时才能使用。"
                    "如果只是想在末尾补充内容，应优先使用 append_note。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "相对于笔记根目录的目标文件路径，必须是 `.txt`、`.md` 或 `.markdown` 文件。",
                        },
                        "content": {
                            "type": "string",
                            "description": "要写入文件的完整内容。该工具会覆盖原文件全部内容。",
                        },
                    },
                    "required": ["filename", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "append_note",
                "description": (
                    "在笔记文件末尾追加内容；如果文件不存在则新建。"
                    "只有在用户明确要求追加、补充到末尾时才能使用。"
                    "如果需要完全改写文件内容，应优先使用 write_note。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "相对于笔记根目录的目标文件路径，必须是 `.txt`、`.md` 或 `.markdown` 文件。",
                        },
                        "content": {
                            "type": "string",
                            "description": "要追加到文件末尾的文本内容。",
                        },
                    },
                    "required": ["filename", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_note",
                "description": (
                    "删除指定笔记文件。"
                    "这是破坏性操作，只有在用户明确要求删除某个文件时才能使用。"
                    "不适用于清空内容、重写内容或读取内容。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "相对于笔记根目录的目标文件路径，必须指向具体文件，不能是目录。",
                        },
                    },
                    "required": ["filename"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_highlights",
                "description": (
                    "围绕指定 focus 从单个文件中提取结构化重点，返回 summary、key_points、action_items、risks、quotes 等字段。"
                    "适用于用户要某份笔记的重点、行动项、风险或摘要，且目标文件已经确定。"
                    "不适用于跨多个文件找内容，也不适用于读取原文全文。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "相对于笔记根目录的目标文件路径，应为一个已确定的单个笔记文件。",
                        },
                        "focus": {
                            "type": "string",
                            "description": "提炼重点时关注的角度，例如“会议决策”“风险点”“后续行动”“DPO 核心结论”。",
                        },
                    },
                    "required": ["filename", "focus"],
                    "additionalProperties": False,
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
    llm_temperature: float
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
                temperature=self.ctx.llm_temperature,
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
