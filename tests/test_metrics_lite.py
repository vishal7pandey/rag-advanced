from app.core.metrics.raglite import lite_metrics
from app.core.metrics.token_cost import estimate_tokens_cost
from app.core.types import Chunk, RetrievedDoc


def test_lite_metrics_bounds():
    docs = [RetrievedDoc(chunk=Chunk(id="c1", doc_id="d", ord=0, text="Alpha", meta={}), score=1.0)]
    m = lite_metrics("Answer text", docs)
    for v in m.values():
        assert 0.0 <= float(v) <= 1.0


def test_token_cost_positive():
    c = estimate_tokens_cost("gpt--mini", prompt_tokens=1000, completion_tokens=500)
    assert c > 0.0
