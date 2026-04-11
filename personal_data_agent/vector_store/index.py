from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import chunk_text

ALLOWED_SUFFIXES = {".txt", ".md", ".markdown"}


@dataclass
class SearchResult:
    filename: str
    chunk_text: str
    score: float


class FileVectorIndex:
    """File-based vector index (no database)."""

    def __init__(
        self,
        notes_root: Path,
        index_dir: Path,
        embedding_model_path: Path,
        chunk_size: int = 450,
        chunk_overlap: int = 80,
    ) -> None:
        self.notes_root = notes_root.resolve()
        self.index_dir = index_dir.resolve()
        self.embedding_model_path = embedding_model_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.state_path = self.index_dir / "state.json"

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._embedder: SentenceTransformer | None = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(str(self.embedding_model_path))
        return self._embedder

    def _collect_notes(self) -> list[Path]:
        files: list[Path] = []
        for path in self.notes_root.rglob("*"):
            if not path.is_file():
                continue
            if ".personal_data_agent" in path.parts:
                continue
            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            files.append(path)
        return sorted(files)

    def _fingerprint(self, files: list[Path]) -> dict[str, dict[str, Any]]:
        state: dict[str, dict[str, Any]] = {}
        for file in files:
            stat = file.stat()
            rel = str(file.relative_to(self.notes_root))
            state[rel] = {"mtime": stat.st_mtime, "size": stat.st_size}
        return state

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _need_rebuild(self, current_fp: dict[str, dict[str, Any]]) -> bool:
        saved = self._load_state().get("files", {})
        if not self.embeddings_path.exists() or not self.chunks_path.exists():
            return True
        return saved != current_fp

    def rebuild(self) -> dict[str, Any]:
        files = self._collect_notes()

        all_records: list[dict[str, Any]] = []
        all_texts: list[str] = []

        for file in files:
            rel = str(file.relative_to(self.notes_root))
            try:
                content = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file.read_text(encoding="utf-8", errors="ignore")

            chunks = chunk_text(content, self.chunk_size, self.chunk_overlap)
            for idx, ck in enumerate(chunks):
                chunk_id = f"{rel}::{idx}"
                all_records.append(
                    {
                        "chunk_id": chunk_id,
                        "filename": rel,
                        "chunk_index": idx,
                        "text": ck,
                    }
                )
                all_texts.append(ck)

        if all_texts:
            vectors = self.embedder.encode(
                all_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
        else:
            vectors = np.zeros((0, 768), dtype="float32")

        with self.chunks_path.open("w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        np.save(self.embeddings_path, vectors)

        self._save_state(
            {
                "files": self._fingerprint(files),
                "num_files": len(files),
                "num_chunks": len(all_records),
            }
        )

        return {"num_files": len(files), "num_chunks": len(all_records)}

    def ensure_fresh(self) -> None:
        files = self._collect_notes()
        if self._need_rebuild(self._fingerprint(files)):
            self.rebuild()

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.chunks_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.ensure_fresh()

        records = self._load_records()
        if not records:
            return []

        vectors = np.load(self.embeddings_path)
        if vectors.shape[0] == 0:
            return []

        q = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")[0]

        scores = vectors @ q
        top_k = min(top_k, len(records))
        idxs = np.argpartition(-scores, top_k - 1)[:top_k]
        idxs = idxs[np.argsort(-scores[idxs])]

        out: list[SearchResult] = []
        for i in idxs:
            rec = records[int(i)]
            out.append(
                SearchResult(
                    filename=rec["filename"],
                    chunk_text=rec["text"],
                    score=float(scores[int(i)]),
                )
            )
        return out
