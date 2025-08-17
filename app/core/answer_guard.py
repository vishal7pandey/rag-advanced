from __future__ import annotations
from typing import Dict, Optional

NO_ANSWER_MD = """\
I don’t know from the provided sources.

**How to proceed**
- Upload or select more relevant documents
- Relax recency/filters
- Try a broader or alternative phrasing
"""


def should_no_answer(
    retrieved_count: int,
    context_precision: Optional[float],
    min_sources: int = 1,
    min_precision: float = 0.50,
) -> bool:
    """
    Decide if we should output a conservative "no answer" based on
    retrieval coverage and a lightweight precision proxy.
    """
    try:
        if retrieved_count <= 0:
            return True
        if retrieved_count <= min_sources and (
            context_precision is not None and float(context_precision) < float(min_precision)
        ):
            return True
        return False
    except Exception:
        return False


def caveat_text(metrics: Dict) -> Optional[str]:
    """
    Produce a soft caveat string when lightweight metrics signal low groundedness
    or precision. Returns None when metrics look fine.
    """
    try:
        cp = metrics.get("context_precision_lite")
        grd = metrics.get("groundedness_lite")
        if (grd is not None and float(grd) < 0.70) or (cp is not None and float(cp) < 0.80):
            return "Some statements may be weakly grounded in the retrieved sources; consider broadening the search."
        return None
    except Exception:
        return None
