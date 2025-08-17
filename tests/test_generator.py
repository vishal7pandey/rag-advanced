"""Tests for the generator module."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock
import pytest

from app.core.generator import generate_answer, _mk_citations, OpenAI
from app.core.types import RetrievedDoc, Chunk


def create_test_doc(text: str, path: str = "test.txt", ord: int = 0) -> RetrievedDoc:
    """Helper to create test RetrievedDoc."""
    chunk = Chunk(
        id=f"doc-{ord}",
        doc_id="doc",
        ord=ord,
        text=text,
        meta={"path": path},
    )
    return RetrievedDoc(chunk=chunk, score=0.8)


def test_mk_citations():
    """Test citation generation."""
    docs = [
        create_test_doc("First doc", "doc1.txt", 0),
        create_test_doc("Second doc", "doc2.txt", 1),
        create_test_doc("Third doc", "doc3.txt", 2),
    ]

    citations = _mk_citations(docs)

    assert len(citations) == 3
    assert citations[0] == {"marker": 1, "path": "doc1.txt", "ord": 0}
    assert citations[1] == {"marker": 2, "path": "doc2.txt", "ord": 1}
    assert citations[2] == {"marker": 3, "path": "doc3.txt", "ord": 2}


def test_mk_citations_empty():
    """Test citation generation with empty docs."""
    citations = _mk_citations([])
    assert citations == []


def test_mk_citations_missing_path():
    """Test citation generation with missing path in metadata."""
    chunk = Chunk(id="x", doc_id="doc", ord=0, text="test", meta={})
    doc = RetrievedDoc(chunk=chunk, score=0.8)

    citations = _mk_citations([doc])
    assert citations == [{"marker": 1, "path": "", "ord": 0}]


def test_generate_answer_offline():
    """Test generate_answer in offline mode."""
    docs = [
        create_test_doc("This is the first document with important information.", "doc1.txt", 0),
        create_test_doc("This is the second document with more details.", "doc2.txt", 1),
    ]

    result = generate_answer(
        prompt="What is important?", model="gpt-3.5-turbo", offline=True, docs=docs
    )

    assert "answer_md" in result
    assert "citations" in result
    assert "usage" in result

    # Check offline stub format
    assert result["answer_md"].startswith("(offline stub) Based on context:")
    assert "first document" in result["answer_md"]

    # Check citations
    assert len(result["citations"]) == 2
    assert result["citations"][0]["path"] == "doc1.txt"

    # Check usage
    assert result["usage"]["prompt_tokens"] == 0
    assert result["usage"]["completion_tokens"] == 0


def test_generate_answer_offline_no_docs():
    """Test generate_answer in offline mode with no docs."""
    result = generate_answer(
        prompt="What is important?", model="gpt-3.5-turbo", offline=True, docs=[]
    )

    assert "No context available" in result["answer_md"]
    assert result["citations"] == []


def test_generate_answer_offline_long_docs():
    """Test generate_answer in offline mode with long documents."""
    long_text = "A" * 300  # Longer than 200 char limit
    docs = [create_test_doc(long_text, "long.txt", 0)]

    result = generate_answer(
        prompt="What is this about?", model="gpt-3.5-turbo", offline=True, docs=docs
    )

    # Should truncate to 200 chars per doc
    assert len([d for d in result["answer_md"] if d == "A"]) <= 200


@pytest.mark.skipif(OpenAI is None, reason="OpenAI package not available")
def test_generate_answer_no_api_key():
    """Test generate_answer with no API key (falls back to offline)."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
        docs = [create_test_doc("Test document", "test.txt", 0)]

        result = generate_answer(
            prompt="What is this?", model="gpt-3.5-turbo", offline=False, docs=docs
        )

        # Should fall back to offline mode
        assert result["answer_md"].startswith("(offline stub)")
        assert result["usage"]["prompt_tokens"] == 0


@pytest.mark.skipif(OpenAI is None, reason="OpenAI package not available")
@patch("app.core.generator.OpenAI")
def test_generate_answer_online_success(mock_openai):
    """Test generate_answer in online mode with successful API call."""
    # Setup mock
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = "This is the AI generated answer with citations [^1]."
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 25
    mock_client.chat.completions.create.return_value = mock_response

    docs = [create_test_doc("Important information here", "source.txt", 0)]

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        result = generate_answer(
            prompt="What is important?", model="gpt-4", offline=False, docs=docs
        )

    # Verify API call
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-4"
    assert call_args["temperature"] == 0.2
    assert len(call_args["messages"]) == 2
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][1]["role"] == "user"
    assert call_args["messages"][1]["content"] == "What is important?"

    # Verify result
    assert result["answer_md"] == "This is the AI generated answer with citations [^1]."
    assert result["usage"]["prompt_tokens"] == 50
    assert result["usage"]["completion_tokens"] == 25
    assert len(result["citations"]) == 1


@pytest.mark.skipif(OpenAI is None, reason="OpenAI package not available")
@patch("app.core.generator.OpenAI")
def test_generate_answer_online_no_usage(mock_openai):
    """Test generate_answer in online mode with no usage info."""
    # Setup mock
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock response with no usage
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Answer without usage info."
    mock_response.usage = None
    mock_client.chat.completions.create.return_value = mock_response

    docs = [create_test_doc("Test doc", "test.txt", 0)]

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        result = generate_answer(
            prompt="Test prompt", model="gpt-3.5-turbo", offline=False, docs=docs
        )

    # Should handle missing usage gracefully
    assert result["usage"]["prompt_tokens"] == 0
    assert result["usage"]["completion_tokens"] == 0


@pytest.mark.skipif(OpenAI is None, reason="OpenAI package not available")
@patch("app.core.generator.OpenAI")
def test_generate_answer_online_empty_response(mock_openai):
    """Test generate_answer in online mode with empty response."""
    # Setup mock
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock response with empty content
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 0
    mock_client.chat.completions.create.return_value = mock_response

    docs = [create_test_doc("Test doc", "test.txt", 0)]

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        result = generate_answer(
            prompt="Test prompt", model="gpt-3.5-turbo", offline=False, docs=docs
        )

    # Should handle None content gracefully
    assert result["answer_md"] == ""
    assert result["usage"]["prompt_tokens"] == 10
