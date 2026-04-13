from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.web.bootstrap import run as streamlit_bootstrap_run

from personal_data_agent.agent.loop import build_agent
from personal_data_agent.config import (
    DEFAULT_API_KEY as CONFIG_DEFAULT_API_KEY,
    DEFAULT_BASE_URL as CONFIG_DEFAULT_BASE_URL,
    DEFAULT_EMBEDDING_MODEL_PATH as CONFIG_DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_MODEL_NAME as CONFIG_DEFAULT_MODEL_NAME,
)

DEFAULT_MODEL = CONFIG_DEFAULT_MODEL_NAME
DEFAULT_BASE_URL = CONFIG_DEFAULT_BASE_URL
DEFAULT_API_KEY = CONFIG_DEFAULT_API_KEY
DEFAULT_EMBEDDING = CONFIG_DEFAULT_EMBEDDING_MODEL_PATH
DEFAULT_NOTES_DIR = "./note"


def _init_state() -> None:
    default_notes_dir = Path(DEFAULT_NOTES_DIR).expanduser().resolve()
    default_notes_dir.mkdir(parents=True, exist_ok=True)
    st.session_state.setdefault("notes_dir_input", str(default_notes_dir))
    st.session_state.setdefault("current_notes_dir", "")
    st.session_state.setdefault("notes_dir_error", "")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("agent", None)

    st.session_state.setdefault("cfg_model_name", DEFAULT_MODEL)
    st.session_state.setdefault("cfg_base_url", DEFAULT_BASE_URL)
    st.session_state.setdefault("cfg_api_key", DEFAULT_API_KEY)
    st.session_state.setdefault("cfg_embedding", DEFAULT_EMBEDDING)


def _inject_css() -> None:
    st.markdown(
        """
<style>
.main {
  background: linear-gradient(180deg, #f7f8fc 0%, #f1f4fa 100%);
}
.block-container {
  max-width: 980px;
  padding-top: 1.5rem;
}
.msg {
  border-radius: 14px;
  padding: 14px 16px;
  margin: 10px 0;
  color: #f9fbff;
  box-shadow: 0 6px 18px rgba(9, 16, 29, 0.2);
}
.msg-user {
  background: #1e2a45;
  border: 1px solid #2f3f66;
}
.msg-assistant {
  background: #202832;
  border: 1px solid #303b48;
}
.msg-role {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
  opacity: 0.9;
  margin-bottom: 6px;
}
.msg-content {
  font-size: 14px;
  line-height: 1.6;
}
.tool-title {
  color: #3d4a61;
  font-size: 13px;
  margin: 6px 0 8px 4px;
}
/* Light styling for tool traces */
div[data-testid="stExpander"] {
  background: #f8fbff;
  border: 1px solid #dbe6f3;
  border-radius: 12px;
}
div[data-testid="stExpander"] summary {
  color: #334760;
  font-weight: 600;
}
div[data-testid="stCodeBlock"] pre {
  background: #eef3f9 !important;
  color: #273445 !important;
  border-radius: 10px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_dark_message(role: str, content: str) -> None:
    role_label = "用户" if role == "user" else "助手"
    role_cls = "msg-user" if role == "user" else "msg-assistant"
    safe = html.escape(content).replace("\n", "<br>")
    st.markdown(
        f"""
<div class="msg {role_cls}">
  <div class="msg-role">{role_label}</div>
  <div class="msg-content">{safe}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _pretty_json(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _preview(text: str, max_len: int = 180) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _render_tool_trace(tool_trace: list[dict[str, Any]], prefix: str) -> None:
    if not tool_trace:
        return

    st.markdown('<div class="tool-title">工具调用轨迹</div>', unsafe_allow_html=True)
    for idx, item in enumerate(tool_trace):
        tool_name = item.get("tool", "unknown")
        args_text = _pretty_json(item.get("arguments", {}))
        result_text = _pretty_json(item.get("result", {}))
        title = f"工具 {idx + 1}: {tool_name}"
        with st.expander(title, expanded=False):
            st.caption(f"参数预览: {_preview(args_text)}")
            st.code(args_text, language="json")
            st.caption(f"返回值预览: {_preview(result_text)}")
            st.code(result_text, language="json")


def _switch_agent(notes_dir: str) -> None:
    target = Path(notes_dir).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        raise ValueError(f"目录不存在或不可用: {target}")

    st.session_state.agent = build_agent(
        notes_root=str(target),
        embedding_model_path=st.session_state.cfg_embedding,
        model_name=st.session_state.cfg_model_name,
        base_url=st.session_state.cfg_base_url,
        api_key=st.session_state.cfg_api_key,
    )
    st.session_state.current_notes_dir = str(target)
    st.session_state.notes_dir_input = str(target)
    st.session_state.notes_dir_error = ""
    st.session_state.messages = []


def _on_notes_dir_change() -> None:
    new_dir = (st.session_state.notes_dir_input or "").strip()
    if not new_dir:
        st.session_state.notes_dir_error = "目录不能为空。"
        return
    try:
        _switch_agent(new_dir)
    except Exception as exc:
        st.session_state.notes_dir_error = str(exc)


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Agent 配置")
        st.text_input("模型", key="cfg_model_name")
        st.text_input("Embedding 路径", key="cfg_embedding")
        st.text_input("Ollama Base URL", key="cfg_base_url")
        st.text_input("API Key", key="cfg_api_key", type="password")

        st.text_input(
            "笔记目录",
            key="notes_dir_input",
            on_change=_on_notes_dir_change,
        )
        if st.session_state.current_notes_dir:
            st.caption(f"当前生效目录: {st.session_state.current_notes_dir}")
        if st.session_state.notes_dir_error:
            st.error(st.session_state.notes_dir_error)


def main() -> None:
    st.set_page_config(page_title="Personal Data Assistant", page_icon="🗂️", layout="wide")
    _init_state()
    if st.session_state.agent is None:
        try:
            _switch_agent(st.session_state.notes_dir_input)
        except Exception as exc:
            st.session_state.notes_dir_error = str(exc)
    _inject_css()
    _render_sidebar()

    st.title("🤖 个人数据助手")

    if st.session_state.agent is None:
        st.info("请在左侧填写有效笔记目录，修改后会自动生效。")

    for i, msg in enumerate(st.session_state.messages):
        _render_dark_message(msg["role"], msg["content"])
        if msg["role"] == "assistant":
            _render_tool_trace(msg.get("tool_trace", []), prefix=f"历史{i}")

    user_prompt = st.chat_input("输入你的问题，例如：帮我提取今天会议纪要的待办")

    if user_prompt:
        if st.session_state.agent is None:
            st.warning("请先加载笔记目录。")
            return

        user_msg = {"role": "user", "content": user_prompt}
        st.session_state.messages.append(user_msg)
        _render_dark_message("user", user_prompt)

        live_holder = st.empty()
        live_trace: list[dict[str, Any]] = []

        def _on_tool_event(event: dict[str, Any]) -> None:
            live_trace.append(event)
            with live_holder.container():
                _render_tool_trace(live_trace, prefix="运行中")

        with st.spinner("Agent 运行中..."):
            try:
                result = st.session_state.agent.run(user_prompt, on_tool_event=_on_tool_event)
                answer = result.answer
                trace = result.tool_trace
            except Exception as exc:
                answer = f"运行失败: {exc}"
                trace = live_trace

        live_holder.empty()

        assistant_msg = {"role": "assistant", "content": answer, "tool_trace": trace}
        st.session_state.messages.append(assistant_msg)

        _render_dark_message("assistant", answer)
        _render_tool_trace(trace, prefix="本轮")


def _launch() -> None:
    if get_script_run_ctx(suppress_warning=True) is None:
        streamlit_bootstrap_run(str(Path(__file__).resolve()), False, sys.argv[1:], {})
        return
    main()


if __name__ == "__main__":
    _launch()
