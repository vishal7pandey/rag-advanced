from unittest.mock import patch

from app.core.flows.hybrid import HybridFlow, HybridParams
from app.core.types import Chunk, RetrievedDoc


def _mk(id_: str, ord_: int = 0) -> RetrievedDoc:
    ch = Chunk(id=id_, doc_id="d", ord=ord_, text=f"t-{id_}", meta={"path": f"{id_}.txt"})
    return RetrievedDoc(chunk=ch, score=1.0)


@patch("app.core.flows.hybrid.build_answer_prompt")
@patch("app.core.flows.hybrid.generate_answer")
@patch("app.core.flows.hybrid.rrf_fuse_multi")
@patch("app.core.flows.hybrid.retrieve_dense")
@patch("app.core.flows.hybrid.retrieve_bm25")
def test_hybrid_flow_multi_query_expansion_calls(
    mock_bm25, mock_dense, mock_rrf_multi, mock_gen, mock_build
):
    # Arrange
    mock_bm25.return_value = [_mk("a")]
    mock_dense.return_value = ([_mk("b")], None)
    mock_rrf_multi.return_value = [_mk("a")]
    mock_build.return_value = "prompt"
    mock_gen.return_value = {
        "answer_md": "ok",
        "citations": [{"source": "a.txt", "page": 1}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    params = HybridParams(
        bm25_k=2,
        dense_k=2,
        rrf_k=5,
        rerank=False,
        multi_query_n=3,
        weight_bm25=0.7,
        weight_dense=0.3,
    )
    flow = HybridFlow(False, "gpt", None, None, params)

    # Patch method to return 2 rewrites so total queries == 3
    with patch.object(HybridFlow, "_generate_rewrites", return_value=["q1", "q2"]):
        out = flow.run("base", {})

    # Assert
    assert out.answer_md == "ok"
    assert mock_bm25.call_count == 3
    assert mock_dense.call_count == 3
    assert mock_rrf_multi.called

    args, kwargs = mock_rrf_multi.call_args
    bm_lists, de_lists = args[0], args[1]
    assert len(bm_lists) == 3
    assert len(de_lists) == 3
    # Check weights plumbed
    assert kwargs.get("w_bm25") == 0.7
    assert kwargs.get("w_dense") == 0.3
