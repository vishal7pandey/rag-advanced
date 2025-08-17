from __future__ import annotations
import re
from typing import Dict, Any


def should_skip_advanced_flow(
    query: str, flow_type: str, config: Dict[str, Any] | None = None
) -> tuple[bool, str]:
    """Determine if advanced flows (HyDE, Multi-hop) should be skipped for cost/efficiency.

    Args:
        query: User query
        flow_type: "hyde" or "multi_hop"
        config: Optional config with guardrail settings

    Returns:
        Tuple of (should_skip, reason)
    """
    if not config:
        config = {}

    # Default guardrail settings
    min_tokens = config.get("min_query_tokens", 3)
    max_cost_estimate = config.get("max_cost_estimate", 0.10)  # $0.10
    keyword_threshold = config.get("keyword_threshold", 0.8)

    # Tokenize query (simple whitespace split)
    tokens = query.strip().split()
    ql = query.strip().lower()
    # Natural-language indicators
    has_punct = bool(re.search(r"[?.!]", ql))
    qw = ["what", "how", "why", "when", "where", "who", "which", "can", "should", "would", "could"]
    has_qw = any(re.search(rf"\b{w}\b", ql) for w in qw)

    # Skip if too short, unless it looks like a natural-language question (punctuation or question words)
    if len(tokens) < min_tokens and not (has_punct or has_qw):
        return True, f"query_too_short ({len(tokens)} < {min_tokens} tokens)"

    # Skip if looks like simple keyword search
    if _is_keyword_query(query, keyword_threshold):
        return True, "keyword_query_detected"

    # Rough cost estimation based on query complexity
    estimated_cost = _estimate_flow_cost(query, flow_type)
    if estimated_cost > max_cost_estimate:
        return True, f"cost_too_high (${estimated_cost:.3f} > ${max_cost_estimate:.3f})"

    return False, "passed_guardrails"


def _is_keyword_query(query: str, threshold: float = 0.8) -> bool:
    """Detect if query looks like simple keyword search vs natural language question."""
    query = query.strip().lower()

    # Early exits: presence of sentence punctuation or clear question words => natural language
    if re.search(r"[?.!]", query):
        return False

    qw = ["what", "how", "why", "when", "where", "who", "which", "can", "should", "would", "could"]
    if any(re.search(rf"\b{w}\b", query) for w in qw):
        return False

    # Count indicators of natural language vs keywords
    nl_indicators = 0
    total_checks = 0

    # Check for question words
    question_words = [
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "can",
        "should",
        "would",
        "could",
    ]
    total_checks += 1
    if any(word in query for word in question_words):
        nl_indicators += 1

    # Check for sentence structure (verbs, articles, etc.)
    structure_words = [
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
    ]
    total_checks += 1
    if any(word in query for word in structure_words):
        nl_indicators += 1

    # Check for punctuation (questions, complete sentences)
    total_checks += 1
    if re.search(r"[.?!]", query):
        nl_indicators += 1

    # Check length (longer queries more likely to be natural language)
    total_checks += 1
    if len(query.split()) > 5:
        nl_indicators += 1

    # If most indicators suggest natural language, it's not a keyword query
    nl_ratio = nl_indicators / total_checks if total_checks > 0 else 0
    return nl_ratio < threshold


def _estimate_flow_cost(query: str, flow_type: str) -> float:
    """Rough cost estimation for advanced flows."""
    # Base costs (very rough estimates)
    base_costs = {
        "hyde": 0.002,  # HyDE seed generation
        "multi_hop": 0.005,  # Sub-query generation + multiple retrievals
    }

    base_cost = base_costs.get(flow_type, 0.001)

    # Scale by query complexity (longer queries = more expensive)
    query_length_factor = min(len(query.split()) / 10.0, 2.0)  # Cap at 2x

    return base_cost * query_length_factor


def get_low_cost_mode_config() -> Dict[str, Any]:
    """Return configuration for low-cost mode."""
    return {
        "min_query_tokens": 4,
        "max_cost_estimate": 0.05,  # $0.05 limit
        "keyword_threshold": 0.6,  # More aggressive keyword detection
        "disable_rerank": True,
        "disable_hyde": True,
        "disable_multi_hop": True,
        # Caps for economical behavior when features are enabled by config/tests
        "rerank_top_n_cap": 6,
        "hyde_seed_max_tokens": 80,
        "multi_hop_max_subqs": 2,
    }
