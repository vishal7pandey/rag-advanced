"""Simple, focused tests for the embeddings module."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from app.core.embeddings import (
    _sha,
    _cache_path,
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


def test_get_default_embedder_offline():
    """Test get_default_embedder in offline mode."""
    embedder = get_default_embedder(offline=True, st_model="test-model")
    assert callable(embedder)


def test_get_default_embedder_online_no_key():
    """Test get_default_embedder in online mode but no API key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
        embedder = get_default_embedder(offline=False)
        assert callable(embedder)


@pytest.mark.skipif(OpenAI is None, reason="OpenAI package not available")
def test_get_default_embedder_online_with_key():
    """Test get_default_embedder in online mode with API key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        embedder = get_default_embedder(offline=False, oa_model="test-model")
        assert callable(embedder)


@pytest.mark.skipif(SentenceTransformer is None, reason="sentence-transformers not available")
def test_get_default_embedder_st():
    """Test get_default_embedder returns ST function."""
    embedder = get_default_embedder(offline=True)
    assert callable(embedder)


# Integration test with minimal mocking
@pytest.mark.skipif(SentenceTransformer is None, reason="sentence-transformers not available")
def test_embed_st_basic(tmp_path):
    """Test basic ST embedding functionality."""
    with (
        patch("app.core.embeddings.SentenceTransformer") as mock_st,
        patch("app.core.embeddings.CACHE_DIR", tmp_path),
    ):
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_st.return_value = mock_model

        # Import after patching
        from app.core.embeddings import embed_st

        # Test
        result = embed_st(["test"], "test-model")

        # Verify
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        mock_model.encode.assert_called_once_with("test", normalize_embeddings=True)


@pytest.mark.skipif(OpenAI is None, reason="OpenAI package not available")
def test_embed_openai_basic(tmp_path):
    """Test basic OpenAI embedding functionality."""
    with (
        patch("app.core.embeddings.OpenAI") as mock_openai,
        patch("app.core.embeddings.CACHE_DIR", tmp_path),
    ):
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_response

        # Import after patching
        from app.core.embeddings import embed_openai

        # Test
        result = embed_openai(["test"])

        # Verify
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        mock_client.embeddings.create.assert_called_once()
