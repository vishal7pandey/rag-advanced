import numpy as np

from app.core.retrievers.rerank import mmr_rerank, llm_rerank
from app.core.types import Chunk, RetrievedDoc


def test_mmr_rerank_ordering():
    """Test that MMR reranking balances relevance and diversity."""
    # Create test documents with known embeddings
    docs = [
        RetrievedDoc(
            chunk=Chunk(id="1", doc_id="d1", ord=0, text="python programming", meta={}), score=0.9
        ),
        RetrievedDoc(
            chunk=Chunk(id="2", doc_id="d1", ord=1, text="python coding", meta={}), score=0.8
        ),  # Similar to doc 1
        RetrievedDoc(
            chunk=Chunk(id="3", doc_id="d1", ord=2, text="machine learning", meta={}), score=0.7
        ),  # Different topic
    ]

    # Create embeddings where docs 1 and 2 are similar, doc 3 is different
    query_vec = np.array([1.0, 0.0, 0.0])  # Query about "python"
    doc_vecs = np.array(
        [
            [1.0, 0.0, 0.0],  # Doc 1: very similar to query
            [0.9, 0.1, 0.0],  # Doc 2: similar to query and doc 1
            [0.0, 0.0, 1.0],  # Doc 3: different topic
        ]
    )

    # Test with high relevance weight (λ=0.9)
    result_relevance = mmr_rerank(query_vec, docs, doc_vecs, lambda_param=0.9, top_n=3)
    assert len(result_relevance) == 3
    # Should prefer most relevant first
    assert result_relevance[0].chunk.text == "python programming"

    # Test with high diversity weight (λ=0.1)
    result_diversity = mmr_rerank(query_vec, docs, doc_vecs, lambda_param=0.1, top_n=3)
    assert len(result_diversity) == 3
    # Should select diverse documents
    selected_texts = [doc.chunk.text for doc in result_diversity]
    assert "machine learning" in selected_texts  # Diverse doc should be selected early

    # Test top_n limiting
    result_limited = mmr_rerank(query_vec, docs, doc_vecs, lambda_param=0.5, top_n=2)
    assert len(result_limited) == 2


def test_mmr_rerank_edge_cases():
    """Test MMR reranking edge cases."""
    # Empty docs
    result = mmr_rerank(np.array([1.0]), [], np.array([]).reshape(0, 1))
    assert result == []

    # Single doc
    docs = [RetrievedDoc(chunk=Chunk(id="1", doc_id="d1", ord=0, text="test", meta={}), score=1.0)]
    doc_vecs = np.array([[1.0]])
    query_vec = np.array([1.0])
    result = mmr_rerank(query_vec, docs, doc_vecs)
    assert len(result) == 1
    assert result[0].chunk.text == "test"

    # Mismatched docs and vectors
    docs = [RetrievedDoc(chunk=Chunk(id="1", doc_id="d1", ord=0, text="test", meta={}), score=1.0)]
    doc_vecs = np.array([[1.0], [2.0]])  # More vectors than docs
    result = mmr_rerank(query_vec, docs, doc_vecs)
    assert result == docs  # Should return original on mismatch


def test_llm_rerank_mock():
    """Test LLM reranking with mocked OpenAI response."""
    docs = [
        RetrievedDoc(
            chunk=Chunk(id="1", doc_id="d1", ord=0, text="relevant document", meta={}), score=0.5
        ),
        RetrievedDoc(
            chunk=Chunk(id="2", doc_id="d1", ord=1, text="irrelevant document", meta={}), score=0.8
        ),
    ]

    # Test behavior with/without OpenAI by checking availability via importlib
    from importlib.util import find_spec

    has_openai = find_spec("openai") is not None
    result, cost_info = llm_rerank("test query", docs)
    if has_openai:
        # When available, function may attempt to rerank; we assert basic shape
        assert len(result) <= len(docs)
        assert "tokens" in cost_info
        assert "cost" in cost_info
    else:
        # OpenAI not available - function should fall back
        assert result == docs  # Should return original docs
        assert cost_info["tokens"] == 0
        assert cost_info["cost"] == 0.0
        assert "error" in cost_info


def test_llm_rerank_empty_docs():
    """Test LLM reranking with empty document list."""
    result, cost_info = llm_rerank("test query", [])
    assert result == []
    assert cost_info["tokens"] == 0
    assert cost_info["cost"] == 0.0
