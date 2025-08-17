from unittest.mock import patch, MagicMock

from app.core.orchestrator import Orchestrator, RetrievalPlan
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
def test_plan_and_run_hybrid_default(mock_make_flow):
    mock_make_flow.return_value = fake_engine()
    o = Orchestrator(
        cfg={"retrieval": {"top_k": 6}, "models": {"generation": "gpt", "offline": "false"}}
    )
    # Use a longer query to avoid 'short or vague' classification that selects HyDE
    out, plan = o.plan_and_run(
        "Explain retrieval augmented generation: architecture, components, benefits", {}
    )
    assert isinstance(out, AnswerBundle)
    assert plan.flow_name == "hybrid"
    assert plan.retrieval.mode == "hybrid"
    assert plan.retrieval.rrf_k >= 10
    # extras plan attached
    assert out.extras.get("flow") == plan.flow_name


@patch("app.core.orchestrator.make_flow")
def test_plan_and_run_hyde_for_short_query(mock_make_flow):
    mock_make_flow.return_value = fake_engine()
    o = Orchestrator(cfg={"retrieval": {"top_k": 5}, "models": {"generation": "gpt"}})
    out, plan = o.plan_and_run("Hi?", {})
    assert plan.flow_name == "hyde"
    assert plan.retrieval.mode == "hyde"
    assert out.answer_md == "ok"


@patch("app.core.orchestrator.make_flow")
def test_plan_and_run_multi_hop_for_connector(mock_make_flow):
    mock_make_flow.return_value = fake_engine()
    o = Orchestrator(cfg={"retrieval": {"top_k": 7}, "models": {"generation": "gpt"}})
    out, plan = o.plan_and_run("Find A then B", {})
    assert plan.flow_name == "multi_hop"
    assert plan.retrieval.subq_max == 3
    assert out.retrieved and out.answer_md


def test_build_plan_shapes():
    o = Orchestrator(
        cfg={
            "retrieval": {"top_k": 6, "rerank": True, "rerank_top_n": 4},
            "models": {"generation": "gpt"},
        }
    )
    rp = o.build_plan("A then B").retrieval
    assert isinstance(rp, RetrievalPlan)
    assert rp.rerank is True
    rp2 = o.build_plan("Short?").retrieval
    assert rp2.mode in {"hyde", "hybrid", "multi_hop"}
