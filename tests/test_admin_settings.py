from app.core.storage import set_settings, get_settings, connect


def test_admin_settings_persist_roundtrip(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    # Force connect() to use temp DB
    monkeypatch.setattr("app.core.storage.DB_FILE", db)
    # Ensure connection and schema
    conn = connect(db)
    conn.close()

    values = {
        "models.generation": "gpt-4o-mini",
        "models.embedding": "text-embedding-3-small",
        "models.offline": True,
        "retrieval.top_k": 7,
        "retrieval.rerank": True,
        "retrieval.rerank_top_n": 8,
        "memory.window_size": 5,
        "memory.summarize": True,
        "flow.default": "hybrid",
        "metrics.ragas_enabled": False,
        "metrics.ragas_model": "gpt-4o-mini",
    }
    set_settings(values, db_path=db)
    out = get_settings(db_path=db)
    # All keys round-trip with correct values
    for k, v in values.items():
        assert out.get(k) == v
