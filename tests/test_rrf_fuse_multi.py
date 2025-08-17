from app.core.retrievers.hybrid import rrf_fuse_multi
from app.core.types import Chunk, RetrievedDoc


def _mk(id_: str) -> RetrievedDoc:
    return RetrievedDoc(
        chunk=Chunk(
            id=id_, doc_id="d", ord=int(id_[-1]) if id_.isdigit() else 0, text=f"t{id_}", meta={}
        ),
        score=1.0,
    )


def test_rrf_fuse_multi_unions_dedup_and_respects_weights():
    # Two "queries" worth of results for bm25 and dense
    bm25_lists = [[_mk("a"), _mk("b")], [_mk("b"), _mk("c")]]
    dense_lists = [[_mk("c"), _mk("b")], [_mk("a"), _mk("c")]]

    fused = rrf_fuse_multi(bm25_lists, dense_lists, k=0, w_bm25=2.0, w_dense=0.5)
    ids = [r.chunk.id for r in fused]

    # All unique ids appear once after dedup
    assert ids == list(dict.fromkeys(ids))  # no duplicates
    assert set(ids) == {"a", "b", "c"}

    # With higher bm25 weight and multiple bm25 occurrences for 'b', it should lead
    assert ids[0] == "b"
