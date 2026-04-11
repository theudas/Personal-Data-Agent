from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 450, chunk_overlap: int = 80) -> list[str]:
    """Simple character-level chunking for Chinese/English mixed notes."""
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks
