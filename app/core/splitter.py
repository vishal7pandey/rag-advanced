from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class SplitConfig:
    chunk_size: int = 800
    overlap: int = 120


def split_text(text: str, cfg: SplitConfig) -> List[str]:
    chunks: List[str] = []
    cs, ov = cfg.chunk_size, cfg.overlap
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + cs)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(end - ov, start + 1)
    return chunks
