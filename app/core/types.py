from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict


@dataclass
class Chunk:
    id: str
    doc_id: str
    ord: int
    text: str
    meta: Dict[str, Any]


@dataclass
class RetrievedDoc:
    chunk: Chunk
    score: float


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class AnswerBundle:
    answer_md: str
    citations: List[Dict[str, Any]]
    usage: Dict[str, Any]
    timings: Dict[str, float]
    metrics: Dict[str, float]
    retrieved: List[RetrievedDoc]
    extras: Dict[str, Any] = field(default_factory=dict)


# --- Typed structures for citation dicts used across the app ---
class CitationUIMap(TypedDict):
    marker: int
    title: str
    doc_short_id: str
    ord: int | None


class CitationEnvMap(TypedDict):
    marker: int
    doc_id: str
    chunk_id: str
    ord: int | None
    path: str | None


# --- Streaming context passed from orchestrator to UI ---
# Keys are optional depending on the flow/fallback path.
class StreamCtx(TypedDict, total=False):
    retrieved: List[RetrievedDoc]
    prompt: str
    t_retrieve: float | None
    t_generate: float | None
    hops: object
