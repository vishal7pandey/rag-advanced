from __future__ import annotations

import numpy as np

from app.core.metrics import raglite as rl
from app.core.types import Chunk, RetrievedDoc


def _doc(doc_id: str, ord_: int = 0) -> RetrievedDoc:
    return RetrievedDoc(
        chunk=Chunk(id=doc_id, doc_id=f"d-{doc_id}", ord=ord_, text=f"text-{doc_id}", meta={}),
        score=1.0,
    )


def test_cosine_zero_vectors():
    a = np.array([])
    b = np.array([1.0, 2.0])
    assert rl.cosine(a, b) == 0.0
    assert rl.cosine(b, np.array([])) == 0.0


def test_cosine_identical_vectors():
    v = np.array([1.0, 0.0, 1.0])
    c = rl.cosine(v, v)
    assert 0.9999 <= c <= 1.0


def test_lite_metrics_empty_inputs():
    m1 = rl.lite_metrics("", [])
    m2 = rl.lite_metrics("answer", [])
    m3 = rl.lite_metrics("", [_doc("a")])
    for m in (m1, m2, m3):
        assert m["answer_relevancy_lite"] == 0.0
        assert m["context_precision_lite"] == 0.0
        assert m["groundedness_lite"] == 0.0


def test_lite_metrics_nonempty():
    docs = [_doc("a"), _doc("b"), _doc("c")]
    m = rl.lite_metrics("some reasonably long answer", docs)
    # values should be in [0,1] and positive
    for k in ("answer_relevancy_lite", "context_precision_lite", "groundedness_lite"):
        assert 0.0 <= m[k] <= 1.0
    assert m["groundedness_lite"] > 0.0


def test_delta_precision_lite_empty_or_no_shared():
    docs_a = [_doc("a"), _doc("b"), _doc("c")]
    docs_b = [_doc("x"), _doc("y"), _doc("z")]
    assert rl.delta_precision_lite([], docs_a) == 0.0
    assert rl.delta_precision_lite(docs_a, []) == 0.0
    assert rl.delta_precision_lite(docs_a, docs_b) == 0.0


def test_delta_precision_lite_positive_improvement_with_topn():
    # Construct lists where an item improves into the top-N but the demoted one
    # falls outside the compared top-N, yielding a positive average improvement.
    before = [_doc("C", 0), _doc("A", 1), _doc("B", 2), _doc("D", 3)]
    after = [_doc("A", 0), _doc("B", 1), _doc("C", 2), _doc("D", 3)]
    # Compare only top-2 positions; shared ids there are just {A}
    delta = rl.delta_precision_lite(before, after, n=2)
    assert delta > 0.0
