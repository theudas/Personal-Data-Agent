from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_EMBEDDING_MODEL_PATH = "./bge-base-zh-v1.5"
DEFAULT_MODEL_NAME = "modelscope.cn/dDreamer/qwen3-8b-toolcalling-1e-4-r16-500steps-gguf:latest"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"


@dataclass(frozen=True)
class AgentConfig:
    notes_root: Path
    embedding_model_path: Path = Path(DEFAULT_EMBEDDING_MODEL_PATH)
    model_name: str = DEFAULT_MODEL_NAME
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
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
