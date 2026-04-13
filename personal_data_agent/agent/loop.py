from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from personal_data_agent.agent.prompts import build_system_prompt
from personal_data_agent.config import (
    AgentConfig,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_MODEL_NAME,
)
from personal_data_agent.runtime.errors import AgentError, SecurityError, ToolExecutionError, ValidationError
from personal_data_agent.runtime.retry import with_retry
from personal_data_agent.runtime.schema import parse_tool_args, validate_tool_args
from personal_data_agent.tools.note_tools import ToolContext, ToolRegistry, get_openai_tools
from personal_data_agent.vector_store.index import FileVectorIndex


@dataclass
class AgentRunResult:
    answer: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)


class PersonalDataAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.config.ensure_paths()

        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        self.vector_index = FileVectorIndex(
            notes_root=self.config.notes_root,
            index_dir=self.config.index_dir,
            embedding_model_path=self.config.embedding_model_path,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.tools = get_openai_tools()
        self.registry = ToolRegistry(
            ToolContext(
                notes_root=self.config.notes_root,
                vector_index=self.vector_index,
                llm_client=self.client,
                llm_model=self.config.model_name,
                top_k=self.config.top_k,
            )
        )

        self.dialog_history: list[dict[str, Any]] = []

    def run(
        self,
        user_input: str,
        on_tool_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentRunResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(str(self.config.notes_root))}
        ]
        messages.extend(self.dialog_history)
        messages.append({"role": "user", "content": user_input})

        tool_trace: list[dict[str, Any]] = []
        total_tool_calls = 0
        failed_signatures: dict[str, int] = {}

        for _ in range(self.config.max_steps):
            response = self._chat_with_retry(messages)
            msg = response.choices[0].message

            assistant_payload: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }

            if msg.tool_calls:
                assistant_payload["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            messages.append(assistant_payload)

            if not msg.tool_calls:
                final_answer = msg.content or ""
                self._remember_dialog_turn(user_input, final_answer)
                return AgentRunResult(answer=final_answer, tool_trace=tool_trace)

            for tool_call in msg.tool_calls:
                total_tool_calls += 1
                if total_tool_calls > self.config.max_tool_calls:
                    return AgentRunResult(
                        answer="工具调用次数已达到上限，已停止。请缩小问题范围后重试。",
                        tool_trace=tool_trace,
                    )

                tool_name = tool_call.function.name
                raw_args = tool_call.function.arguments

                tool_result = self._execute_tool_call(tool_name, raw_args)
                tool_trace.append(
                    {
                        "tool": tool_name,
                        "arguments": raw_args,
                        "result": tool_result,
                    }
                )
                if on_tool_event is not None:
                    try:
                        on_tool_event(
                            {
                                "tool": tool_name,
                                "arguments": raw_args,
                                "result": tool_result,
                                "tool_call_id": tool_call.id,
                            }
                        )
                    except Exception:
                        # UI callback errors should not break agent execution.
                        pass

                signature = f"{tool_name}:{raw_args}"
                terminal_answer: str | None = None
                if tool_result.get("ok") is False:
                    failed_signatures[signature] = failed_signatures.get(signature, 0) + 1
                    terminal_answer = self._maybe_build_terminal_tool_error_answer(
                        tool_name=tool_name,
                        raw_args=raw_args,
                        tool_result=tool_result,
                        failure_count=failed_signatures[signature],
                    )
                    if failed_signatures[signature] >= 2:
                        tool_result = {
                            "ok": False,
                            "error": "Repeated tool failure. Please revise plan or use different tool.",
                        }
                        if terminal_answer is None:
                            terminal_answer = self._build_repeated_tool_failure_answer(tool_name, raw_args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
                if terminal_answer is not None:
                    self._remember_dialog_turn(user_input, terminal_answer)
                    return AgentRunResult(answer=terminal_answer, tool_trace=tool_trace)

        return AgentRunResult(
            answer="达到最大推理步数，已停止。请把任务拆小后继续。",
            tool_trace=tool_trace,
        )

    def _chat_with_retry(self, messages: list[dict[str, Any]]):
        def _request():
            try:
                return self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.2,
                )
            except Exception as exc:
                raise ToolExecutionError(f"LLM request failed: {exc}", retriable=True) from exc

        return with_retry(_request, max_retries=3)

    def _execute_tool_call(self, tool_name: str, raw_args: Any) -> dict[str, Any]:
        try:
            parsed = parse_tool_args(raw_args)
            args = validate_tool_args(tool_name, parsed)
        except ValidationError as exc:
            return {"ok": False, "error": f"Invalid tool args: {exc}"}

        def _run_tool():
            try:
                return self.registry.execute(tool_name, args)
            except SecurityError as exc:
                raise ToolExecutionError(str(exc), retriable=False) from exc
            except AgentError as exc:
                raise ToolExecutionError(str(exc), retriable=False) from exc
            except OSError as exc:
                raise ToolExecutionError(f"IO error: {exc}", retriable=True) from exc

        try:
            out = with_retry(_run_tool, max_retries=3)
            if isinstance(out, dict) and "ok" not in out:
                out["ok"] = True
            return out
        except ToolExecutionError as exc:
            return {"ok": False, "error": str(exc)}

    def _remember_dialog_turn(self, user_input: str, assistant_output: str) -> None:
        self.dialog_history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_output},
            ]
        )

    def _maybe_build_terminal_tool_error_answer(
        self,
        tool_name: str,
        raw_args: Any,
        tool_result: dict[str, Any],
        failure_count: int,
    ) -> str | None:
        error = tool_result.get("error")
        if not isinstance(error, str):
            return None

        if error.startswith("Path not found: "):
            target = self._extract_tool_target(raw_args, "filename", "directory") or error.removeprefix("Path not found: ")
            kind = "目录" if tool_name == "list_notes" else "文件"
            return (
                f"未找到你请求的{kind}：`{target}`。"
                "我现在无法继续读取它。请确认名称或路径是否正确，或先查看当前笔记目录中的可用文件后再试。"
            )

        if error.startswith("Not a file: "):
            target = self._extract_tool_target(raw_args, "filename") or error.removeprefix("Not a file: ")
            return f"`{target}` 不是可读取的文件，所以我无法按“读取文件”的方式处理它。请换一个具体文件名后重试。"

        if error.startswith("Not a directory: "):
            target = self._extract_tool_target(raw_args, "directory") or error.removeprefix("Not a directory: ")
            return f"`{target}` 不是目录，所以我无法列出其中的笔记。请确认目录路径后再试。"

        if error.startswith("Access denied: "):
            return "请求的路径超出了当前允许访问的笔记目录范围，所以我不能读取它。请提供笔记目录内的文件路径。"

        if error == "Only .txt / .md / .markdown files are supported":
            return "当前只支持 `.txt`、`.md` 和 `.markdown` 文件。我暂时不能读取这个类型的文件，请换成受支持的文本笔记。"

        if failure_count >= 2:
            return self._build_repeated_tool_failure_answer(tool_name, raw_args)

        return None

    def _build_repeated_tool_failure_answer(self, tool_name: str, raw_args: Any) -> str:
        target = self._extract_tool_target(raw_args, "filename", "directory")
        action = {
            "read_note": "读取文件",
            "delete_note": "删除文件",
            "extract_highlights": "提取文件重点",
            "list_notes": "列出目录内容",
        }.get(tool_name, "执行该操作")
        if target:
            return (
                f"我尝试对 `{target}` {action} 时连续失败，继续重复同样的工具调用意义不大。"
                "请确认路径是否正确，或换一种更明确的请求方式后再试。"
            )
        return "我连续两次执行同类工具调用都失败了。为避免无效循环，我先停止并把错误返回给你，请调整请求后再试。"

    def _extract_tool_target(self, raw_args: Any, *keys: str) -> str | None:
        parsed: dict[str, Any] | None = None
        if isinstance(raw_args, dict):
            parsed = raw_args
        elif isinstance(raw_args, str):
            try:
                loaded = json.loads(raw_args)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                parsed = loaded

        if not parsed:
            return None

        for key in keys:
            value = parsed.get(key)
            if isinstance(value, str) and value:
                return value
        return None


def build_agent(
    notes_root: str,
    embedding_model_path: str = DEFAULT_EMBEDDING_MODEL_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
) -> PersonalDataAgent:
    cfg = AgentConfig(
        notes_root=Path(notes_root),
        embedding_model_path=Path(embedding_model_path),
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )
    return PersonalDataAgent(cfg)
