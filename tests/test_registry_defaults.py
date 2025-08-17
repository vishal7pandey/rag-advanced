from app.core.flows.registry import make_flow, StandardFlow


def test_registry_default_unknown_flow_returns_standard():
    f = make_flow("not_a_flow", False, "gpt", None, None, {})
    assert isinstance(f, StandardFlow)
    # defaults present
    assert f.params.top_k == 6
    assert f.params.rerank is False


def test_registry_standard_defaults():
    f = make_flow("standard", False, "gpt", None, None, {})
    assert isinstance(f, StandardFlow)
    assert f.params.top_k == 6
