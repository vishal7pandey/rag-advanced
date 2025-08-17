from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import asdict, is_dataclass

from app.core.types import RetrievedDoc


def _dc_to_dict(obj: Any) -> Any:
    # Only convert dataclass instances (not dataclass classes)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_dc_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dc_to_dict(v) for k, v in obj.items()}
    return obj


def build_envelope(
    *,
    session_id: str,
    run_id: Optional[str],
    question: str,
    answer_text: str,
    plan: Any,
    retrieved: List[RetrievedDoc],
    citations_env: List[Dict[str, Any]],
    timings: Dict[str, Any],
    usage: Dict[str, Any],
    metrics: Dict[str, Any],
    persona_hint: Optional[str],
    format_hint: Optional[str],
    exit_state: str = "ok",
    safety: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a machine-readable envelope capturing provenance, metrics, and governance data.
    Persona is included only here (not in Markdown UI).
    """
    plan_obj = _dc_to_dict(plan)
    # Serialize retrieved minimal context for provenance
    ctx_docs: List[Dict[str, Any]] = []
    for rd in retrieved:
        try:
            ctx_docs.append(
                {
                    "doc_id": rd.chunk.doc_id,
                    "chunk_id": rd.chunk.id,
                    "ord": rd.chunk.ord,
                    "score": float(rd.score),
                    "path": rd.chunk.meta.get("path", ""),
                    "meta": {k: v for k, v in rd.chunk.meta.items() if k != "text"},
                }
            )
        except Exception:
            pass

    env: Dict[str, Any] = {
        "session_id": session_id,
        "run_id": run_id,
        "question": question,
        "answer": {
            "text": answer_text,
            # Footnote markers are determined by citations_env order (1..N)
            "citations": citations_env,
        },
        "plan": plan_obj,
        "retrieval": {
            "docs": ctx_docs,
        },
        "timings": dict(timings or {}),
        "usage": dict(usage or {}),
        "metrics": dict(metrics or {}),
        "safety": dict(safety or {}),
        "hints": {
            "persona": persona_hint,
            "format": format_hint,
        },
        "exit": {"state": exit_state},
        "version": 1,
    }
    return env
