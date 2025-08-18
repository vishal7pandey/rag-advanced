from unittest.mock import patch, MagicMock

from app.core.flows.guardrails import (
    should_skip_advanced_flow,
    _is_keyword_query,
    get_low_cost_mode_config,
)
from app.core.flows.hyde import HyDEFlow, HyDEParams
from app.core.flows.multi_hop import MultiHopFlow, MultiHopParams


def test_should_skip_advanced_flow():
    """Test guardrails for skipping advanced flows."""

    # Short query should be skipped
    should_skip, reason = should_skip_advanced_flow("AI", "hyde")
    assert should_skip
    assert "query_too_short" in reason

    # Keyword query should be skipped
    should_skip, reason = should_skip_advanced_flow("machine learning python", "hyde")
    assert should_skip
    assert "keyword_query_detected" in reason

    # Natural language question should pass
    should_skip, reason = should_skip_advanced_flow(
        "What are the benefits of machine learning?", "hyde"
    )
    assert not should_skip
    assert reason == "passed_guardrails"

    # Custom config with higher cost limit
    config = {"max_cost_estimate": 0.20}
    should_skip, reason = should_skip_advanced_flow(
        "What is artificial intelligence?", "hyde", config
    )
    assert not should_skip


def test_is_keyword_query():
    """Test keyword query detection."""

    # Keyword queries
    assert _is_keyword_query("python machine learning")
    assert _is_keyword_query("AI neural networks")
    assert _is_keyword_query("database optimization")

    # Natural language queries
    assert not _is_keyword_query("What is machine learning?")
    assert not _is_keyword_query("How do I optimize my database?")
    assert not _is_keyword_query("Can you explain neural networks to me?")
    assert not _is_keyword_query("I need help with Python programming.")


def test_get_low_cost_mode_config():
    """Test low-cost mode configuration."""
    config = get_low_cost_mode_config()

    assert config["max_cost_estimate"] == 0.05  # Lower cost limit
    assert config["min_query_tokens"] == 4  # Higher token requirement
    assert config["disable_hyde"] is True
    assert config["disable_multi_hop"] is True
    assert config["disable_rerank"] is True


@patch("app.core.flows.standard.StandardFlow")
def test_hyde_auto_skip(mock_standard_flow):
    """Test HyDE flow auto-skip functionality."""
    # Mock the standard flow fallback
    mock_fallback = MagicMock()
    mock_bundle = MagicMock()
    mock_bundle.extras = {}
    mock_fallback.run.return_value = mock_bundle
    mock_standard_flow.return_value = mock_fallback

    # Create HyDE flow with auto mode enabled
    params = HyDEParams(auto_mode=True)
    flow = HyDEFlow(offline=True, gen_model="gpt--mini", emb_st=None, emb_oa=None, params=params)

    # Test with short query that should be skipped
    result = flow.run("AI", {})

    # Should have called standard flow fallback
    mock_standard_flow.assert_called_once()
    mock_fallback.run.assert_called_once_with("AI", {})

    # Should have added skip reason to extras
    assert "hyde_skipped" in result.extras


@patch("app.core.flows.standard.StandardFlow")
def test_multi_hop_auto_skip(mock_standard_flow):
    """Test Multi-hop flow auto-skip functionality."""
    # Mock the standard flow fallback
    mock_fallback = MagicMock()
    mock_bundle = MagicMock()
    mock_bundle.extras = {}
    mock_fallback.run.return_value = mock_bundle
    mock_standard_flow.return_value = mock_fallback

    # Create Multi-hop flow with auto mode enabled
    params = MultiHopParams(auto_mode=True)
    flow = MultiHopFlow(
        offline=True, gen_model="gpt--mini", emb_st=None, emb_oa=None, params=params
    )

    # Test with keyword query that should be skipped
    result = flow.run("python ML", {})

    # Should have called standard flow fallback
    mock_standard_flow.assert_called_once()
    mock_fallback.run.assert_called_once_with("python ML", {})

    # Should have added skip reason to extras
    assert "multi_hop_skipped" in result.extras


@patch("app.core.flows.standard.StandardFlow")
def test_hyde_auto_mode_disabled(mock_standard_flow):
    """Test HyDE flow with auto mode disabled."""
    # Create HyDE flow with auto mode disabled
    params = HyDEParams(auto_mode=False)
    flow = HyDEFlow(offline=True, gen_model="gpt--mini", emb_st=None, emb_oa=None, params=params)

    # Mock the HyDE seed generation to avoid actual OpenAI calls
    with patch("app.core.flows.hyde._generate_hyde_seed") as mock_seed:
        mock_seed.return_value = "test seed"

        # Mock retrieve_dense to avoid database calls
        with patch("app.core.retrievers.dense.retrieve_dense") as mock_retrieve:
            mock_retrieve.return_value = ([], [])

            # Mock generate_answer to avoid OpenAI calls
            with patch("app.core.generator.generate_answer") as mock_generate:
                mock_generate.return_value = {
                    "answer_md": "test answer",
                    "citations": [],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                }

                # Test with short query - should NOT be skipped when auto_mode=False
                flow.run("AI", {})

                # Should NOT have called standard flow fallback
                mock_standard_flow.assert_not_called()

                # Should have proceeded with HyDE flow
                mock_seed.assert_called_once()


def test_cost_estimation():
    """Test cost estimation for different flow types and query lengths."""
    from app.core.flows.guardrails import _estimate_flow_cost

    # Short query
    cost_short = _estimate_flow_cost("AI", "hyde")

    # Long query
    cost_long = _estimate_flow_cost(
        "What are the detailed benefits and drawbacks of using machine learning in healthcare applications?",
        "hyde",
    )

    # Long query should cost more
    assert cost_long > cost_short

    # Multi-hop should be more expensive than HyDE
    cost_multi_hop = _estimate_flow_cost("What is AI?", "multi_hop")
    cost_hyde = _estimate_flow_cost("What is AI?", "hyde")
    assert cost_multi_hop > cost_hyde
