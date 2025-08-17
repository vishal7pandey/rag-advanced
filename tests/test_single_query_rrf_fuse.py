from unittest.mock import patch

from app.core.flows.hybrid import HybridFlow, HybridParams
from app.core.types import Chunk, RetrievedDoc


def _mk(id_: str, ord_: int = 0) -> RetrievedDoc:
    ch = Chunk(id=id_, doc_id="d", ord=ord_, text=f"t-{id_}", meta={"path": f"{id_}.txt"})
    return RetrievedDoc(chunk=ch, score=1.0)


@patch("app.core.flows.hybrid.build_answer_prompt")
@patch("app.core.flows.hybrid.generate_answer")
@patch("app.core.flows.hybrid.rrf_fuse")
@patch("app.core.flows.hybrid.rrf_fuse_multi")
@patch("app.core.flows.hybrid.retrieve_dense")
@patch("app.core.flows.hybrid.retrieve_bm25")
def test_hybrid_flow_single_query_uses_rrf_fuse(
    mock_bm25, mock_dense, mock_rrf_multi, mock_rrf, mock_gen, mock_build
):
    # Arrange single-query path (multi_query_n = 1)
    mock_bm25.return_value = [_mk("a")]
    mock_dense.return_value = ([_mk("b")], None)
    mock_rrf.return_value = [_mk("a")]
    mock_build.return_value = "prompt"
    mock_gen.return_value = {
        "answer_md": "ok",
        "citations": [{"source": "a.txt", "page": 1}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }

    params = HybridParams(
        bm25_k=2,
        dense_k=2,
        rrf_k=7,
        rerank=False,
        multi_query_n=1,  # ensures single-query path
        weight_bm25=0.8,
        weight_dense=0.2,
    )
    flow = HybridFlow(False, "gpt", None, None, params)

    out = flow.run("only-one", {})

    # Assert output
    assert out.answer_md == "ok"

    # Assert rrf_fuse called, rrf_fuse_multi not called
    assert mock_rrf.called
    assert not mock_rrf_multi.called

    # Validate args to rrf_fuse include weights and k
    args, kwargs = mock_rrf.call_args
    bm_list, de_list = args[0], args[1]
    assert isinstance(bm_list, list) and isinstance(de_list, list)
    assert kwargs.get("w_bm25") == 0.8
    assert kwargs.get("w_dense") == 0.2
    assert kwargs.get("k") == 7
