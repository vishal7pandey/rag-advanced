from __future__ import annotations
from typing import List, Dict
import os


def eval_ragas(
    question: str, answer: str, contexts: List[str], model: str | None = None
) -> Dict[str, float]:
    """
    Evaluate RAGAS metrics if ragas is installed and OPENAI_API_KEY is set.
    Returns a dict with keys prefixed by 'ragas_' or empty dict if unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}
    try:
        # Lazy imports; ragas may not be installed
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            faithfulness,
            context_precision,
        )
    except Exception:
        return {}

    # Build dataset and attempt evaluation; gracefully handle runtime errors.
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    ds = Dataset.from_dict(data)
    try:
        # Note: ragas uses OpenAI via OPENAI_API_KEY when available.
        result = evaluate(ds, metrics=[answer_relevancy, faithfulness, context_precision])
        # result.scores is a dict-like; to be safe, convert to pandas and get first row
        pdf = result.to_pandas()
    except Exception:
        # If evaluation fails for any reason (e.g., metric requires ground_truth),
        # return empty metrics rather than raising so callers/UIT can continue.
        return {}
    row = pdf.iloc[0].to_dict() if not pdf.empty else {}
    out: Dict[str, float] = {}
    # Normalize keys with 'ragas_' prefix
    for k in ["answer_relevancy", "faithfulness", "context_precision"]:
        if k in row and isinstance(row[k], (int, float)):
            out[f"ragas_{k}"] = float(row[k])
    return out
