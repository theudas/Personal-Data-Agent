from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    notes_root: Path
    embedding_model_path: Path = Path("./bge-base-zh-v1.5")
    model_name: str = "modelscope.cn/Qwen/Qwen3-8B-GGUF:latest"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    max_steps: int = 8
    max_tool_calls: int = 24
    top_k: int = 5
    chunk_size: int = 450
    chunk_overlap: int = 80

    @property
    def index_dir(self) -> Path:
        return self.notes_root / ".personal_data_agent" / "index"

    def ensure_paths(self) -> None:
        self.notes_root.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
