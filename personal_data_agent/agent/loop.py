from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from personal_data_agent.agent.prompts import build_system_prompt
from personal_data_agent.config import AgentConfig
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
                self.dialog_history.extend(
                    [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": final_answer},
                    ]
                )
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
                if tool_result.get("ok") is False:
                    failed_signatures[signature] = failed_signatures.get(signature, 0) + 1
                    if failed_signatures[signature] >= 2:
                        tool_result = {
                            "ok": False,
                            "error": "Repeated tool failure. Please revise plan or use different tool.",
                        }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

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


def build_agent(
    notes_root: str,
    embedding_model_path: str = "./bge-base-zh-v1.5",
    model_name: str = "modelscope.cn/Qwen/Qwen3-8B-GGUF:latest",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> PersonalDataAgent:
    cfg = AgentConfig(
        notes_root=Path(notes_root),
        embedding_model_path=Path(embedding_model_path),
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )
    return PersonalDataAgent(cfg)
