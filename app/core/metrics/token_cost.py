from __future__ import annotations


def estimate_tokens_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Minimal placeholder, prices depend on model; set tiny default
    rate_in = 0.00015 / 1000.0
    rate_out = 0.0006 / 1000.0
    return prompt_tokens * rate_in + completion_tokens * rate_out
