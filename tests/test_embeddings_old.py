"""Tests for the embeddings module."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from app.core.embeddings import (
    _sha,
    _cache_path,
    embed_openai,
    embed_st,
    get_default_embedder,
    OpenAI,
    SentenceTransformer,
)


def test_sha():
    """Test SHA hash generation."""
    assert _sha("test") == "a94a8fe5ccb19ba61c4c0873d391e987982fbbd3"
    assert _sha("") == "da39a3ee5e6b4b0d3255bfef95601890afd80709"
    assert _sha("hello world") == "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"


def test_cache_path():
    """Test cache path generation."""
    path = _cache_path("test-model", "test text")
    assert "test-model_" in str(path)
    assert path.suffix == ".npy"
    assert "emb_cache" in str(path)


@pytest.mark.skipif(OpenAI is None, reason="OpenAI package not available")
@patch("app.core.embeddings.OpenAI")
def test_embed_openai(mock_openai, tmp_path):
    """Test OpenAI embedding with mock."""
    # Setup mock client and response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock response data
    mock_embedding = [0.1, 0.2, 0.3]
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = mock_embedding
    mock_client.embeddings.create.return_value = mock_response

    # Test with mock cache directory
    with patch("app.core.embeddings.CACHE_DIR", tmp_path):
        # First call - should call OpenAI API
        result = embed_openai(["test text"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)  # 1 text, 3-dim vector
        # Cache file should exist now
        from app.core.embeddings import _cache_path

        cp = _cache_path("text-embedding-3-small", "test text")
        assert cp.exists()

        # Verify API was called once
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="test text"
        )

        # Reset mocks for second test
        mock_client.embeddings.create.reset_mock()

        # Second call - should use cache
        result2 = embed_openai(["test text"])
        assert np.allclose(result, result2)

        # Verify API was not called again (used cache)
        mock_client.embeddings.create.assert_not_called()


@pytest.mark.skipif(SentenceTransformer is None, reason="sentence-transformers not available")
@patch("app.core.embeddings.SentenceTransformer")
def test_embed_st(mock_st, tmp_path):
    """Test sentence-transformers embedding with mock."""
    # Setup mocks
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_st.return_value = mock_model

    # Test with mock cache directory and reset in-memory model cache
    with (
        patch("app.core.embeddings.CACHE_DIR", tmp_path),
        patch("app.core.embeddings._ST_MODELS", {}),
    ):
        # Test 1: Single text with cache miss
        # Ensure no pre-existing cache file
        from app.core.embeddings import _cache_path

        cp = _cache_path("test-model", "test text")
        if cp.exists():
            cp.unlink()
        result1 = embed_st(["test text"], model_name="test-model")

        # Verify results and calls
        assert isinstance(result1, np.ndarray)
        assert result1.shape == (1, 3)
        mock_model.encode.assert_called_once_with("test text", normalize_embeddings=True)
        # Cache file should exist
        from app.core.embeddings import _cache_path

        assert _cache_path("test-model", "test text").exists()

        # Reset mocks for next test
        mock_model.encode.reset_mock()

        # Test 2: Multiple texts with first in cache, second new
        result2 = embed_st(["test text", "new text"], model_name="test-model")

        # Verify results and calls
        assert result2.shape == (2, 3)
        # Should only encode the new text, not the cached one
        mock_model.encode.assert_called_once_with("new text", normalize_embeddings=True)
        # New text now cached as well
        assert _cache_path("test-model", "new text").exists()


def test_get_default_embedder_offline():
    """Test get_default_embedder in offline mode."""
    embedder = get_default_embedder(offline=True, st_model="test-model")
    assert callable(embedder)

    # Should use sentence-transformers in offline mode
    with patch("app.core.embeddings.embed_st") as mock_embed_st:
        mock_embed_st.return_value = np.array([[1, 2, 3]])
        result = embedder(["test"])
        mock_embed_st.assert_called_once_with(["test"], "test-model")
        assert np.array_equal(result, np.array([[1, 2, 3]]))


@patch("app.core.embeddings.SentenceTransformer")
def test_get_default_embedder_offline_cache(mock_st, tmp_path):
    """Test get_default_embedder in offline mode with cache."""
    # Setup mock model
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([1, 2, 3])
    mock_st.return_value = mock_model

    # Test with mock cache directory
    with (
        patch("app.core.embeddings.CACHE_DIR", tmp_path),
        patch("app.core.embeddings._ST_MODELS", {}),
    ):
        # Test 1: First call - cache miss
        # Ensure no pre-existing cache file
        from app.core.embeddings import _cache_path

        cp = _cache_path("test-model", "test cache")
        if cp.exists():
            cp.unlink()
        result1 = embed_st(["test cache"], "test-model")

        # Verify encode was called with correct args
        mock_model.encode.assert_called_once_with("test cache", normalize_embeddings=True)
        assert np.array_equal(result1, np.array([[1, 2, 3]]))
        # File should exist now
        from app.core.embeddings import _cache_path

        assert _cache_path("test-model", "test cache").exists()

        # Reset mocks for next test
        mock_model.encode.reset_mock()

        # Test 2: Second call with same input - should use cache
        result2 = embed_st(["test cache"], "test-model")

        # Verify encode was not called (used cache)
        mock_model.encode.assert_not_called()
        assert np.array_equal(result1, result2)


@patch("app.core.embeddings.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_get_default_embedder_online_with_key(mock_openai, tmp_path):
    """Test get_default_embedder in online mode with API key."""
    # Setup mock client and response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock response data
    mock_embedding = [0.1, 0.2, 0.3]
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = mock_embedding
    mock_client.embeddings.create.return_value = mock_response

    # Use isolated cache dir to avoid accidental cache hits
    with patch("app.core.embeddings.CACHE_DIR", tmp_path):
        # Ensure no pre-existing cache file
        from app.core.embeddings import _cache_path

        cp = _cache_path("test-model", "test")
        if cp.exists():
            cp.unlink()
        # Get the embedder and test it
        embedder = get_default_embedder(offline=False, oa_model="test-model")
        result = embedder(["test"])

    # Verify results
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)

    # Verify API was called with correct arguments
    mock_client.embeddings.create.assert_called_once()
    call_args = mock_client.embeddings.create.call_args[1]
    assert call_args["model"] == "test-model"
    assert call_args["input"] == "test"


def test_cache_behavior(tmp_path):
    """Test that caching works correctly."""
    # Create a mock for SentenceTransformer and use real filesystem cache
    with (
        patch("app.core.embeddings.SentenceTransformer") as mock_st,
        patch("app.core.embeddings.CACHE_DIR", tmp_path),
        patch("app.core.embeddings._ST_MODELS", {}),
    ):
        # Setup mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([1, 2, 3])
        mock_st.return_value = mock_model

        # Test 1: First call - cache miss
        result1 = embed_st(["test cache"], "test-model")

        # Verify encode was called with correct args
        mock_model.encode.assert_called_once_with("test cache", normalize_embeddings=True)
        assert np.array_equal(result1, np.array([[1, 2, 3]]))
        # File should exist now
        from app.core.embeddings import _cache_path

        assert _cache_path("test-model", "test cache").exists()

        # Reset mocks for next test
        mock_model.encode.reset_mock()

        # Test 2: Second call with same input - should use cache
        result2 = embed_st(["test cache"], "test-model")

        # Verify encode was not called (used cache)
        mock_model.encode.assert_not_called()
        assert np.array_equal(result1, result2)

        # Reset mocks for next test
        mock_model.encode.reset_mock()

        # Test 3: Different text should trigger new embedding
        mock_model.encode.return_value = np.array([4, 5, 6])  # New embedding
        result3 = embed_st(["different text"], "test-model")

        # Verify encode was called with new text
        mock_model.encode.assert_called_once_with("different text", normalize_embeddings=True)
        assert not np.array_equal(result2, result3)
