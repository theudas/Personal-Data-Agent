from __future__ import annotations

import argparse
import json
from pathlib import Path

from personal_data_agent.agent.loop import build_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Personal Data Assistant Agent (Ollama + Qwen3-8B)")
    parser.add_argument("--notes-dir", required=True, help="用户笔记根目录，Agent 只能访问该目录")
    parser.add_argument("--embedding-model", default="./bge-base-zh-v1.5", help="embedding 模型路径")
    parser.add_argument(
        "--model",
        default="modelscope.cn/Qwen/Qwen3-8B-GGUF:latest",
        help="Ollama 部署的模型名",
    )
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="Ollama OpenAI 兼容接口")
    parser.add_argument("--api-key", default="ollama", help="Ollama API key (默认 ollama)")
    parser.add_argument("--query", default=None, help="单轮问题；不填则进入交互模式")
    parser.add_argument("--show-trace", action="store_true", help="输出工具调用轨迹")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    notes_dir = Path(args.notes_dir).expanduser().resolve()
    agent = build_agent(
        notes_root=str(notes_dir),
        embedding_model_path=args.embedding_model,
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    if args.query:
        result = agent.run(args.query)
        print(result.answer)
        if args.show_trace:
            print("\n--- TOOL TRACE ---")
            print(json.dumps(result.tool_trace, ensure_ascii=False, indent=2))
        return

    print("Personal Data Assistant 已启动，输入 exit 退出。")
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见。")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            print("再见。")
            break
        if not user_input:
            continue

        result = agent.run(user_input)
        print(f"\n助手: {result.answer}")

        if args.show_trace:
            print("\n--- TOOL TRACE ---")
            print(json.dumps(result.tool_trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
