from unittest.mock import patch

from app.core.citations import _split_claims, per_claim_citations
from app.core.types import Chunk, RetrievedDoc


def make_rd(text: str, path: str, ord_: int, score: float) -> RetrievedDoc:
    ch = Chunk(id=f"c-{ord_}", doc_id="d1", ord=ord_, text=text, meta={"path": path})
    return RetrievedDoc(chunk=ch, score=score)


def test_split_claims_basic():
    assert _split_claims("") == []
    assert _split_claims("Single sentence.") == ["Single sentence."]
    claims = _split_claims("First. Second? Third!")
    assert claims == ["First.", "Second?", "Third!"]


def test_split_claims_whitespace():
    claims = _split_claims("  A.   B?   C!  ")
    assert claims == ["A.", "B?", "C!"]


@patch("app.core.citations.rerank_bge_topn")
def test_per_claim_with_reranker(mock_rerank):
    docs = [
        make_rd("alpha", "a.txt", 0, 0.9),
        make_rd("beta", "b.txt", 1, 0.8),
    ]
    # Return same order; top_n=1 should pick first
    mock_rerank.return_value = docs
    ces = per_claim_citations("A. B.", docs, top_n=1)
    assert len(ces) == 2
    assert ces[0].supports[0][1] is docs[0]


@patch("app.core.citations.rerank_bge_topn", side_effect=Exception("no model"))
def test_per_claim_fallback_no_reranker(_mock_rerank):
    docs = [
        make_rd("alpha", "a.txt", 0, 0.9),
        make_rd("beta", "b.txt", 1, 0.8),
    ]
    ces = per_claim_citations("A. B.", docs, top_n=1)
    assert len(ces) == 2
    # Fallback should use the first retrieved
    assert ces[0].supports[0][1] is docs[0]
