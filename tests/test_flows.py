"""Tests for flow registry and standard flow implementation."""

from unittest.mock import MagicMock, patch

from app.core.flows.registry import make_flow, StandardFlow, HybridFlow
from app.core.flows.standard import StandardParams
from app.core.flows.hybrid import HybridParams
from app.core.types import AnswerBundle


def test_make_flow_returns_standard_flow_by_default():
    """Test that make_flow returns a StandardFlow by default."""
    flow = make_flow("unknown_flow", False, "gpt-3.5-turbo", None, None, {})
    assert isinstance(flow, StandardFlow)
    assert flow.gen_model == "gpt-3.5-turbo"
    assert flow.params.top_k == 6  # Default value


def test_make_flow_creates_standard_flow():
    """Test creating a StandardFlow with custom parameters."""
    cfg = {"top_k": 10, "rerank": True, "rerank_top_n": 5}
    flow = make_flow("standard", False, "gpt-4", None, None, cfg)

    assert isinstance(flow, StandardFlow)
    assert flow.gen_model == "gpt-4"
    assert flow.params.top_k == 10
    assert flow.params.rerank is True
    assert flow.params.rerank_top_n == 5


def test_make_flow_creates_hybrid_flow():
    """Test creating a HybridFlow with custom parameters."""
    cfg = {"bm25_k": 25, "dense_k": 30, "rrf_k": 15, "rerank": True, "rerank_top_n": 8}
    flow = make_flow(
        "hybrid", True, "gpt-3.5-turbo", "all-MiniLM-L6-v2", "text-embedding-ada-002", cfg
    )

    assert isinstance(flow, HybridFlow)
    assert flow.offline is True
    assert flow.emb_st == "all-MiniLM-L6-v2"
    assert flow.params.bm25_k == 25
    assert flow.params.dense_k == 30
    assert flow.params.rrf_k == 15


@patch("app.core.flows.standard.retrieve_dense")
@patch("app.core.flows.standard.generate_answer")
@patch("app.core.flows.standard.build_answer_prompt")
def test_standard_flow_run(mock_build_prompt, mock_generate, mock_retrieve):
    """Test StandardFlow's run method with mocked dependencies."""
    # Setup test data
    mock_docs = [
        MagicMock(page_content="Doc 1", metadata={"source": "test.pdf", "page": 1}),
        MagicMock(page_content="Doc 2", metadata={"source": "test.pdf", "page": 2}),
    ]
    mock_retrieve.return_value = (mock_docs, None)
    mock_build_prompt.return_value = "Formatted prompt with context"
    mock_generate.return_value = {
        "answer_md": "Test answer",
        "sources": ["test.pdf"],
        "citations": [{"source": "test.pdf", "page": 1}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }

    # Initialize flow with test parameters
    params = StandardParams(top_k=5, rerank=False)
    flow = StandardFlow(False, "gpt-3.5-turbo", None, None, params)

    # Execute the flow
    result = flow.run("Test question", {})

    # Verify the result
    assert isinstance(result, AnswerBundle)
    assert result.answer_md == "Test answer"
    assert len(result.citations) == 1  # Check citations instead of sources
    assert result.citations[0]["source"] == "test.pdf"  # Verify citation content
    assert len(result.retrieved) == 2  # We provided 2 mock docs
    assert result.timings["t_retrieve"] > 0
    assert result.timings["t_generate"] > 0

    # Verify mocks were called correctly
    mock_retrieve.assert_called_once()
    mock_build_prompt.assert_called_once()
    mock_generate.assert_called_once_with(
        "Formatted prompt with context", "gpt-3.5-turbo", False, mock_docs
    )


def test_standard_flow_run_with_rerank():
    """Test StandardFlow's run method with reranking enabled."""
    # This would be similar to test_standard_flow_run but with rerank=True
    # and would verify the reranking logic
    pass  # Implementation would go here


def test_standard_flow_run_with_memory():
    """Test StandardFlow's run method with memory context."""
    # This would test that memory is correctly incorporated into the prompt
    pass  # Implementation would go here


@patch("app.core.flows.hybrid.retrieve_bm25")
@patch("app.core.flows.hybrid.retrieve_dense")
@patch("app.core.flows.hybrid.rrf_fuse")
@patch("app.core.flows.hybrid.generate_answer")
@patch("app.core.flows.hybrid.build_answer_prompt")
def test_hybrid_flow_run(
    mock_build_prompt, mock_generate, mock_rrf_fuse, mock_retrieve_dense, mock_retrieve_bm25
):
    """Test HybridFlow's run method with mocked dependencies."""
    # Setup test data
    mock_docs = [
        MagicMock(page_content="Doc 1", metadata={"source": "test.pdf", "page": 1}, score=0.9),
        MagicMock(page_content="Doc 2", metadata={"source": "test.pdf", "page": 2}, score=0.8),
    ]

    # Mock return values
    mock_retrieve_bm25.return_value = mock_docs
    mock_retrieve_dense.return_value = (mock_docs, None)
    mock_rrf_fuse.return_value = mock_docs  # Return same docs for simplicity
    mock_build_prompt.return_value = "Formatted prompt with context"
    mock_generate.return_value = {
        "answer_md": "Test hybrid answer",
        "sources": ["test.pdf"],
        "citations": [{"source": "test.pdf", "page": 1}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 60},
    }

    # Initialize flow with test parameters
    params = HybridParams(bm25_k=20, dense_k=20, rrf_k=10, rerank=False)
    flow = HybridFlow(False, "gpt-3.5-turbo", "all-MiniLM-L6-v2", "text-embedding-ada-002", params)

    # Execute the flow
    result = flow.run("Test hybrid question", {})

    # Verify the result
    assert isinstance(result, AnswerBundle)
    assert result.answer_md == "Test hybrid answer"
    assert len(result.citations) == 1
    assert result.citations[0]["source"] == "test.pdf"
    assert len(result.retrieved) == 2
    assert result.timings["t_retrieve"] > 0
    assert result.timings["t_generate"] > 0

    # Verify mocks were called correctly
    mock_retrieve_bm25.assert_called_once()
    mock_retrieve_dense.assert_called_once()
    mock_rrf_fuse.assert_called_once()
    mock_build_prompt.assert_called_once()
    mock_generate.assert_called_once_with(
        "Formatted prompt with context", "gpt-3.5-turbo", False, mock_docs
    )


@patch("app.core.flows.hybrid.retrieve_bm25")
@patch("app.core.flows.hybrid.retrieve_dense")
@patch("app.core.flows.hybrid.rrf_fuse")
@patch("app.core.flows.hybrid.rerank_bge_topn")
@patch("app.core.flows.hybrid.generate_answer")
@patch("app.core.flows.hybrid.build_answer_prompt")
def test_hybrid_flow_with_rerank(
    mock_build_prompt,
    mock_generate,
    mock_rerank,
    mock_rrf_fuse,
    mock_retrieve_dense,
    mock_retrieve_bm25,
):
    """Test HybridFlow with reranking enabled."""
    # Setup test data
    mock_docs = [
        MagicMock(page_content="Doc 1", metadata={"source": "test.pdf", "page": 1}, score=0.9),
        MagicMock(page_content="Doc 2", metadata={"source": "test.pdf", "page": 2}, score=0.8),
    ]

    # Mock return values
    mock_retrieve_bm25.return_value = mock_docs
    mock_retrieve_dense.return_value = (mock_docs, None)
    mock_rrf_fuse.return_value = mock_docs
    mock_rerank.return_value = mock_docs  # For simplicity, return same order
    mock_build_prompt.return_value = "Formatted prompt with context"
    mock_generate.return_value = {
        "answer_md": "Test hybrid answer with rerank",
        "sources": ["test.pdf"],
        "citations": [{"source": "test.pdf", "page": 1}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 60},
    }

    # Initialize flow with rerank enabled
    params = HybridParams(
        bm25_k=20,
        dense_k=20,
        rrf_k=10,
        rerank=True,
        rerank_top_n=5,
        rerank_strategy="cross_encoder",
    )
    flow = HybridFlow(False, "gpt-3.5-turbo", "all-MiniLM-L6-v2", "text-embedding-ada-002", params)

    # Execute the flow
    result = flow.run("Test hybrid question with rerank", {})

    # Verify the result
    assert isinstance(result, AnswerBundle)
    assert result.answer_md == "Test hybrid answer with rerank"

    # Verify reranking was called
    mock_rerank.assert_called_once()

    # Verify metrics include delta precision when reranking is enabled
    assert "delta_precision_lite" in result.metrics


@patch("app.core.flows.hybrid.retrieve_bm25")
@patch("app.core.flows.hybrid.retrieve_dense")
@patch("app.core.flows.hybrid.rrf_fuse")
@patch("app.core.flows.hybrid.generate_answer")
@patch("app.core.flows.hybrid.build_answer_prompt")
def test_hybrid_flow_offline_mode(
    mock_build_prompt, mock_generate, mock_rrf_fuse, mock_retrieve_dense, mock_retrieve_bm25
):
    """Test HybridFlow in offline mode with local models."""
    # Setup test data
    mock_docs = [
        MagicMock(page_content="Doc 1", metadata={"source": "test.pdf", "page": 1}, score=0.9),
        MagicMock(page_content="Doc 2", metadata={"source": "test.pdf", "page": 2}, score=0.8),
    ]

    # Mock return values
    mock_retrieve_bm25.return_value = mock_docs
    mock_retrieve_dense.return_value = (mock_docs, None)
    mock_rrf_fuse.return_value = mock_docs
    mock_build_prompt.return_value = "Formatted prompt with context"
    mock_generate.return_value = {
        "answer_md": "Test hybrid answer offline",
        "sources": ["test.pdf"],
        "citations": [{"source": "test.pdf", "page": 1}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }

    # Initialize flow in offline mode with local models
    params = HybridParams(bm25_k=20, dense_k=20, rrf_k=10, rerank=False)
    flow = HybridFlow(True, "local-model", "all-MiniLM-L6-v2", None, params)

    # Execute the flow
    result = flow.run("Test hybrid question offline", {})

    # Verify the result
    assert isinstance(result, AnswerBundle)
    assert result.answer_md == "Test hybrid answer offline"

    # Verify local model was used (offline=True in generate_answer call)
    mock_generate.assert_called_once()
    # Check positional and keyword arguments
    args, kwargs = mock_generate.call_args
    assert len(args) >= 4  # prompt, model_name, offline, docs
    assert args[0] == "Formatted prompt with context"  # prompt
    assert args[1] == "local-model"  # model_name
    assert args[2] is True  # offline
    assert args[3] == mock_docs  # docs
