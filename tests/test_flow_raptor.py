from unittest.mock import patch, MagicMock

from app.core.flows.registry import make_flow
from app.core.flows.raptor import RaptorFlow, RaptorParams
from app.core.types import AnswerBundle


def test_make_flow_creates_raptor_flow():
    cfg = {"levels": 2, "fanout": 4, "top_k_final": 7, "rerank": False}
    flow = make_flow("raptor", False, "gpt--mini", None, None, cfg)
    assert isinstance(flow, RaptorFlow)
    assert flow.params.levels == 2
    assert flow.params.top_k_final == 7


@patch("app.core.flows.raptor.retrieve_dense")
@patch("app.core.flows.raptor.generate_answer")
@patch("app.core.flows.raptor.build_answer_prompt")
def test_raptor_flow_run(mock_build_prompt, mock_generate, mock_retrieve):
    mock_docs = [MagicMock(), MagicMock()]
    mock_retrieve.return_value = (mock_docs, None)
    mock_build_prompt.return_value = "prompt"
    mock_generate.return_value = {
        "answer_md": "A",
        "citations": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    flow = RaptorFlow(False, "gpt--mini", None, None, RaptorParams())
    out = flow.run("Q?", {})
    assert isinstance(out, AnswerBundle)
    assert out.answer_md == "A"
    assert out.retrieved == mock_docs
    assert out.timings["t_retrieve"] >= 0
    assert out.timings["t_generate"] >= 0


@patch("app.core.flows.raptor.rerank_bge_topn")
@patch("app.core.flows.raptor.retrieve_dense")
@patch("app.core.flows.raptor.generate_answer")
@patch("app.core.flows.raptor.build_answer_prompt")
def test_raptor_flow_run_with_rerank(mock_build_prompt, mock_generate, mock_retrieve, mock_rerank):
    # Setup
    mock_docs = [MagicMock(), MagicMock()]
    mock_retrieve.return_value = (mock_docs, None)
    mock_rerank.return_value = list(reversed(mock_docs))
    mock_build_prompt.return_value = "prompt"
    mock_generate.return_value = {
        "answer_md": "A.",
        "citations": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    # Enable rerank
    params = RaptorParams(rerank=True, rerank_top_n=2)
    flow = RaptorFlow(False, "gpt--mini", None, None, params)
    out = flow.run("Q?", {})
    assert isinstance(out, AnswerBundle)
    # Metrics should include delta precision when rerank is on
    assert "delta_precision_lite" in out.metrics
