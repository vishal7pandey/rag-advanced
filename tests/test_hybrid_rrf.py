from app.core.retrievers.hybrid import rrf_fuse
from app.core.types import Chunk, RetrievedDoc


def _mk(doc_id):
    return RetrievedDoc(
        chunk=Chunk(id=str(doc_id), doc_id="d", ord=int(doc_id), text=f"t{doc_id}", meta={}),
        score=1.0,
    )


def test_rrf_fuse_unions_and_orders_by_rrf():
    # bm25 ranks: 1,2,3 ; dense ranks: 3,2,1
    bm25 = [_mk("1"), _mk("2"), _mk("3")]
    dense = [_mk("3"), _mk("2"), _mk("1")]
    fused = rrf_fuse(bm25, dense, k=10)
    ids = [r.chunk.id for r in fused]
    # With standard RRF (1/(k+rank)), edges can slightly outrank the middle
    assert ids[0] in {"1", "3"}
    assert set(ids) == {"1", "2", "3"}
