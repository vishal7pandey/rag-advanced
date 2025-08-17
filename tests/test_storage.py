from __future__ import annotations

import contextlib
import sqlite3
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from app.core.storage import (
    connect,
    init_db,
    get_recent_runs,
    add_message,
    get_recent_messages,
    set_settings,
    get_settings,
    add_run,
)


@contextlib.contextmanager
def temp_db() -> Generator[Path, None, None]:
    """Context manager that creates a temporary database file for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        db_path = temp_dir / "test.db"
        yield db_path
    finally:
        # Clean up the temporary directory and its contents
        for file in temp_dir.glob("*"):
            file.unlink()
        temp_dir.rmdir()


def test_connect_creates_schema():
    """Test that connect() creates the database schema if it doesn't exist."""
    with temp_db() as db_path:
        conn = connect(db_path)
        try:
            # Verify tables were created
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('documents', 'chunks', 'indices', 'messages', 'runs', 'memory', 'settings')"
            )
            tables = {row[0] for row in cursor.fetchall()}
            assert tables == {
                "documents",
                "chunks",
                "indices",
                "messages",
                "runs",
                "memory",
                "settings",
            }
        finally:
            conn.close()


def test_init_db_creates_schema():
    """Test that init_db() creates the database schema."""
    with temp_db() as db_path:
        # First, initialize the database
        init_db(db_path)

        # Verify tables were created by trying to insert data
        conn = sqlite3.connect(str(db_path))  # Use str() for Windows compatibility
        try:
            cursor = conn.cursor()
            # Test documents table
            cursor.execute(
                "INSERT INTO documents (id, path, sha) VALUES (?, ?, ?)",
                ("test1", "/test/path", "sha1"),
            )
            # Test chunks table
            cursor.execute(
                "INSERT INTO chunks (id, doc_id, ord, text) VALUES (?, ?, ?, ?)",
                ("chunk1", "test1", 0, "test text"),
            )
            conn.commit()

            # Verify data was inserted
            cursor.execute("SELECT COUNT(*) FROM documents")
            assert cursor.fetchone()[0] == 1

            # Test that we can query the chunks table
            cursor.execute("SELECT text FROM chunks WHERE doc_id = ?", ("test1",))
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == "test text"
        finally:
            conn.close()


def test_message_persistence():
    """Test that messages can be added and retrieved."""
    with temp_db() as db_path:
        init_db(db_path)

        # Add test messages
        session_id = "test-session-123"
        test_messages = [
            ("user", "Hello, world!"),
            ("assistant", "Hi there! How can I help?"),
            ("user", "What's the weather like?"),
        ]

        for role, content in test_messages:
            add_message(session_id, role, content, db_path)

        # Retrieve messages
        messages = get_recent_messages(session_id, limit=10, db_path=db_path)

        # Verify messages were stored and retrieved correctly
        assert len(messages) == 3
        for (role, content), (expected_role, expected_content) in zip(messages, test_messages):
            assert role == expected_role
            assert content == expected_content


def test_settings_persistence():
    """Test that settings can be stored and retrieved."""
    with temp_db() as db_path:
        init_db(db_path)

        # Test data
        test_settings = {
            "theme": "dark",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "nested": {"key": "value"},
        }

        # Save settings
        set_settings(test_settings, db_path)

        # Retrieve settings
        loaded_settings = get_settings(db_path)

        # Verify settings were saved and loaded correctly
        assert loaded_settings["theme"] == "dark"
        assert loaded_settings["model"] == "gpt-4"
        assert loaded_settings["temperature"] == 0.7
        assert loaded_settings["max_tokens"] == 1000
        assert loaded_settings["nested"] == {"key": "value"}

        # Test updating settings
        set_settings({"theme": "light", "new_setting": True}, db_path)
        updated_settings = get_settings(db_path)
        assert updated_settings["theme"] == "light"  # Updated
        assert updated_settings["new_setting"] is True  # Added
        assert updated_settings["model"] == "gpt-4"  # Unchanged


def test_run_history():
    """Test that run history can be recorded and retrieved."""
    with temp_db() as db_path:
        init_db(db_path)

        # Add test runs
        test_runs = [
            {
                "session_id": "session-1",
                "flow": "standard",
                "question": "What is AI?",
                "answer": "AI stands for Artificial Intelligence.",
                "citations": [{"text": "AI is intelligence...", "source": "source1"}],
                "timings": {"retrieval": 0.5, "generation": 1.2},
                "metrics": {"relevance": 0.9, "fluency": 0.95},
                "cost": 0.0025,
            },
            {
                "session_id": "session-2",
                "flow": "hyde",
                "question": "Explain machine learning",
                "answer": "Machine learning is a subset of AI...",
                "citations": [{"text": "ML is a method...", "source": "source2"}],
                "timings": {"retrieval": 0.7, "generation": 1.5},
                "metrics": {"relevance": 0.85, "fluency": 0.92},
                "cost": 0.0030,
            },
        ]

        for run in test_runs:
            add_run(
                session_id=run["session_id"],
                flow=run["flow"],
                question=run["question"],
                answer=run["answer"],
                citations=run["citations"],
                timings=run["timings"],
                metrics=run["metrics"],
                cost=run["cost"],
                db_path=db_path,
            )

        # Retrieve recent runs
        runs = get_recent_runs(limit=5, db_path=db_path)

        # Verify runs were stored and retrieved correctly
        assert len(runs) == 2

        # Check runs are in the correct order (oldest first)
        assert runs[0]["session_id"] == test_runs[0]["session_id"]
        assert runs[0]["flow"] == test_runs[0]["flow"]
        assert runs[0]["question"] == test_runs[0]["question"]
        assert runs[0]["answer"] == test_runs[0]["answer"]
        assert runs[0]["citations"] == test_runs[0]["citations"]
        assert runs[0]["timings"] == test_runs[0]["timings"]
        assert runs[0]["metrics"] == test_runs[0]["metrics"]
        assert runs[0]["cost"] == test_runs[0]["cost"]
        assert "created_at" in runs[0]

        # Check that runs are ordered by created_at DESC (newest first)
        assert runs[0]["created_at"] <= runs[1]["created_at"]


def test_message_limit():
    """Test that get_recent_messages respects the limit parameter."""
    with temp_db() as db_path:
        init_db(db_path)

        # Add test messages
        session_id = "test-limit"
        for i in range(10):
            add_message(session_id, "user", f"Message {i}", db_path)

        # Test different limits
        for limit in [1, 3, 5, 10]:
            messages = get_recent_messages(session_id, limit=limit, db_path=db_path)
            assert len(messages) == min(limit, 10)

            # Verify order is oldest to newest (as per implementation)
            for i, (_, content) in enumerate(messages):
                assert content == f"Message {i}"


if __name__ == "__main__":
    pytest.main([__file__])
