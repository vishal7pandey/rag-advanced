from app.core.metrics.ragas_wrap import eval_ragas


def test_ragas_skips_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = eval_ragas("q", "a", ["ctx1"])  # should gracefully skip
    assert out == {}


def test_ragas_skips_when_package_missing(monkeypatch):
    # Simulate OPENAI key present but ragas import fails
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Force ImportError for ragas and datasets by patching __import__
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("ragas") or name == "datasets":
            raise ImportError("simulated missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = eval_ragas("q", "a", ["ctx1"])  # should return empty dict
    assert out == {}
