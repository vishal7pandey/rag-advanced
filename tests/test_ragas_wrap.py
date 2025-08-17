from __future__ import annotations

import os
import sys
import types
from unittest.mock import patch, MagicMock

import pytest

# Mock the ragas imports at the module level
with patch.dict(
    "sys.modules",
    {
        "datasets": MagicMock(),
        "ragas": MagicMock(),
        "ragas.metrics": MagicMock(),
        "ragas.metrics.answer_relevancy": MagicMock(),
        "ragas.metrics.faithfulness": MagicMock(),
        "ragas.metrics.context_precision": MagicMock(),
    },
):
    from app.core.metrics.ragas_wrap import eval_ragas


def test_eval_ragas_no_api_key():
    """Test that eval_ragas returns empty dict when OPENAI_API_KEY is not set."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
        result = eval_ragas("question", "answer", ["context"])
        assert result == {}


def test_eval_ragas_import_error():
    """Test that eval_ragas returns empty dict when ragas import fails."""
    with patch.dict("sys.modules", {"datasets": None, "ragas": None, "ragas.metrics": None}):
        # Reload the module to trigger the import error
        if "app.core.metrics.ragas_wrap" in sys.modules:
            del sys.modules["app.core.metrics.ragas_wrap"]
        from app.core.metrics import ragas_wrap

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
            result = ragas_wrap.eval_ragas("question", "answer", ["context"])
            assert result == {}


def test_eval_ragas_success():
    """Test successful evaluation with ragas metrics."""
    # Setup mocks
    mock_dataset = MagicMock()
    mock_dataset_class = MagicMock()
    mock_dataset_class.from_dict.return_value = mock_dataset

    # Create a mock DataFrame-like object that supports
    #   - pdf.empty -> False
    #   - pdf.iloc[0].to_dict() -> desired row
    mock_metrics = MagicMock()
    row_obj = MagicMock()
    row_obj.to_dict.return_value = {
        "answer_relevancy": 0.9,
        "faithfulness": 0.85,
        "context_precision": 0.8,
    }
    iloc_mock = MagicMock()
    iloc_mock.__getitem__.return_value = row_obj
    mock_df = MagicMock()
    mock_df.empty = False
    mock_df.iloc = iloc_mock

    mock_result = MagicMock()
    mock_result.to_pandas.return_value = mock_df
    mock_evaluate = MagicMock(return_value=mock_result)

    # Build fake modules matching import paths used in ragas_wrap
    ds_module = types.ModuleType("datasets")
    ds_module.Dataset = mock_dataset_class
    ragas_module = types.ModuleType("ragas")
    ragas_module.evaluate = mock_evaluate
    ragas_metrics_module = types.ModuleType("ragas.metrics")
    ragas_metrics_module.answer_relevancy = mock_metrics
    ragas_metrics_module.faithfulness = mock_metrics
    ragas_metrics_module.context_precision = mock_metrics

    # Patch sys.modules with these fake modules
    with patch.dict(
        "sys.modules",
        {
            "datasets": ds_module,
            "ragas": ragas_module,
            "ragas.metrics": ragas_metrics_module,
        },
    ):
        # Reload the module to use the mocked imports
        if "app.core.metrics.ragas_wrap" in sys.modules:
            del sys.modules["app.core.metrics.ragas_wrap"]
        from app.core.metrics import ragas_wrap

        # Test with API key set
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
            result = ragas_wrap.eval_ragas(
                "test question", "test answer", ["context 1", "context 2"]
            )

            # Verify the result
            assert result == {
                "ragas_answer_relevancy": 0.9,
                "ragas_faithfulness": 0.85,
                "ragas_context_precision": 0.8,
            }

            # Verify Dataset was created with correct data
            mock_dataset_class.from_dict.assert_called_once_with(
                {
                    "question": ["test question"],
                    "answer": ["test answer"],
                    "contexts": [["context 1", "context 2"]],
                }
            )

            # Verify evaluate was called with correct arguments
            mock_evaluate.assert_called_once()
            args, kwargs = mock_evaluate.call_args
            assert args[0] == mock_dataset


def test_eval_ragas_evaluation_error():
    """Test that eval_ragas handles evaluation errors gracefully."""
    # Setup mocks
    mock_dataset_class = MagicMock()
    mock_dataset = MagicMock()
    mock_dataset_class.from_dict.return_value = mock_dataset

    # Patch the imports with a failing evaluate function
    with patch.dict(
        "sys.modules",
        {
            "datasets.Dataset": mock_dataset_class,
            "ragas.evaluate": MagicMock(side_effect=Exception("Test error")),
            "ragas.metrics.answer_relevancy": MagicMock(),
            "ragas.metrics.faithfulness": MagicMock(),
            "ragas.metrics.context_precision": MagicMock(),
        },
    ):
        # Reload the module to use the mocked imports
        if "app.core.metrics.ragas_wrap" in sys.modules:
            del sys.modules["app.core.metrics.ragas_wrap"]
        from app.core.metrics import ragas_wrap

        # Test with API key set
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
            result = ragas_wrap.eval_ragas("question", "answer", ["context"])
            assert result == {}


def test_eval_ragas_empty_results():
    """Test that eval_ragas handles empty results from ragas."""
    # Setup mocks
    mock_dataset_class = MagicMock()
    mock_dataset = MagicMock()
    mock_dataset_class.from_dict.return_value = mock_dataset

    # Mock empty result
    mock_result = MagicMock()
    mock_result.to_pandas.return_value = MagicMock(empty=True)

    # Patch the imports
    with patch.dict(
        "sys.modules",
        {
            "datasets.Dataset": mock_dataset_class,
            "ragas.evaluate": MagicMock(return_value=mock_result),
            "ragas.metrics.answer_relevancy": MagicMock(),
            "ragas.metrics.faithfulness": MagicMock(),
            "ragas.metrics.context_precision": MagicMock(),
        },
    ):
        # Reload the module to use the mocked imports
        if "app.core.metrics.ragas_wrap" in sys.modules:
            del sys.modules["app.core.metrics.ragas_wrap"]
        from app.core.metrics import ragas_wrap

        # Test with API key set
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
            result = ragas_wrap.eval_ragas("question", "answer", ["context"])
            assert result == {}


if __name__ == "__main__":
    pytest.main([__file__])
