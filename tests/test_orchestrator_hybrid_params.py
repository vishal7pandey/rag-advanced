from unittest.mock import patch, MagicMock

from app.core.orchestrator import Orchestrator
from app.core.types import AnswerBundle, Chunk, RetrievedDoc


def make_bundle() -> AnswerBundle:
    ch = Chunk(id="c1", doc_id="d1", ord=0, text="t", meta={"path": "a.txt"})
    rd = RetrievedDoc(chunk=ch, score=1.0)
    return AnswerBundle(
        answer_md="ok",
        citations=[],
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        timings={"t_retrieve": 0.01, "t_generate": 0.02},
        metrics={},
        retrieved=[rd],
    )


def fake_engine():
    eng = MagicMock()
    eng.run.return_value = make_bundle()
    return eng


@patch("app.core.orchestrator.make_flow")
def test_orchestrator_passes_hybrid_weights_and_multi_query(mock_make_flow):
    mock_make_flow.return_value = fake_engine()
    cfg = {
        "retrieval": {
            "top_k": 6,
            "rrf_weight_bm25": 0.6,
            "rrf_weight_dense": 0.4,
            "multi_query_n": 3,
        },
        "models": {"generation": "gpt", "offline": "true"},
    }
    o = Orchestrator(cfg=cfg)
    # Long query to avoid HyDE selection
    o.plan_and_run("Explain hybrid retrieval weighted fusion across multiple queries in detail", {})

    # Validate arguments passed to make_flow
    assert mock_make_flow.called
    args, kwargs = mock_make_flow.call_args
    # args: (flow_name, offline, gen_model, emb_st, emb_oa, params)
    flow_name, _offline, _gen, _st, _oa, params = args
    assert flow_name == "hybrid"
    assert params["weight_bm25"] == 0.6
    assert params["weight_dense"] == 0.4
    assert params["multi_query_n"] == 3
