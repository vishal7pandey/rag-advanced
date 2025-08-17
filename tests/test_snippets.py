from app.core.snippets import make_snippet


def test_snippet_highlights_terms():
    s = make_snippet("Deep learning uses neural networks.", ["learning"])
    assert "**learning**" in s or "**Learning**" in s


def test_snippet_truncates_with_ellipsis():
    long = "x" * 300
    s = make_snippet(long, [], 120)
    assert s.endswith("…")
