from unittest.mock import patch

from app.core.flows.hybrid import HybridFlow, HybridParams
from app.core.types import Chunk, RetrievedDoc


def _mk(id_: str, ord_: int = 0) -> RetrievedDoc:
    ch = Chunk(id=id_, doc_id="d", ord=ord_, text=f"t-{id_}", meta={"path": f"{id_}.txt"})
    return RetrievedDoc(chunk=ch, score=1.0)


@patch("app.core.flows.hybrid.stream_answer")
@patch("app.core.flows.hybrid.build_answer_prompt")
@patch("app.core.flows.hybrid.rrf_fuse")
@patch("app.core.flows.hybrid.rrf_fuse_multi")
@patch("app.core.flows.hybrid.retrieve_dense")
@patch("app.core.flows.hybrid.retrieve_bm25")
def test_hybrid_flow_run_stream_multi_query(
    mock_bm25, mock_dense, mock_rrf_multi, mock_rrf, mock_build, mock_stream
):
    # Arrange multi-query path
    mock_bm25.return_value = [_mk("a")]
    mock_dense.return_value = ([_mk("b")], None)
    fused_docs = [_mk("a"), _mk("b")]
    mock_rrf_multi.return_value = list(fused_docs)
    mock_build.return_value = "prompt"
    mock_stream.return_value = iter(["tok1", "tok2"])  # simple iterator

    params = HybridParams(
        bm25_k=2,
        dense_k=2,
        rrf_k=5,
        rerank=False,
        multi_query_n=3,  # triggers multi-query path
        weight_bm25=0.7,
        weight_dense=0.3,
    )
    flow = HybridFlow(False, "gpt", None, None, params)

    with patch.object(HybridFlow, "_generate_rewrites", return_value=["q1", "q2"]):
        it, ctx = flow.run_stream("base", {"memory_text": "mem"})

    # Validate fusion path
    assert mock_rrf_multi.called
    assert not mock_rrf.called

    # Validate build prompt was called with fused docs
    assert mock_build.called
    _, kwargs = mock_build.call_args
    assert kwargs.get("docs") is not None

    # Context checks
    assert isinstance(it, type(iter([])))  # iterator
    assert isinstance(ctx, dict)
    assert ctx.get("prompt") == "prompt"
    # retrieved should reflect fused (after dedupe and cap); here length 2
    assert [d.chunk.id for d in ctx.get("retrieved", [])] == [d.chunk.id for d in fused_docs]
