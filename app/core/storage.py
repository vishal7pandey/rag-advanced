from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional
import json

DB_FILE = Path(__file__).resolve().parents[1] / "rag.db"

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  sha TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  ord INTEGER NOT NULL,
  text TEXT NOT NULL,
  meta_json TEXT,
  FOREIGN KEY (doc_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS indices (
  index_name TEXT PRIMARY KEY,
  faiss_path TEXT NOT NULL,
  dim INTEGER NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  session_id TEXT,
  flow TEXT,
  question TEXT,
  answer TEXT,
  citations_json TEXT,
  timings_json TEXT,
  metrics_json TEXT,
  cost REAL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory (
  session_id TEXT NOT NULL,
  summary TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DB_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    # Longer timeout + allow use across threads for Streamlit callbacks
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=30000;")
    # Ensure schema is present for callers that use connect() directly
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    except Exception:
        pass
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def get_recent_runs(limit: int = 50, db_path: Optional[Path] = None) -> list[dict]:
    conn = connect(db_path)
    try:
        cur = conn.execute(
            "SELECT id, session_id, flow, question, answer, citations_json, timings_json, metrics_json, cost, created_at FROM runs ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        )
        rows = cur.fetchall()
        out: list[dict] = []
        for r in rows:
            out.append(
                {
                    "id": r[0],
                    "session_id": r[1],
                    "flow": r[2],
                    "question": r[3],
                    "answer": r[4],
                    "citations": json.loads(r[5] or "[]"),
                    "timings": json.loads(r[6] or "{}"),
                    "metrics": json.loads(r[7] or "{}"),
                    "cost": r[8],
                    "created_at": r[9],
                }
            )
        return out
    finally:
        conn.close()


# Chat messages persistence helpers
def add_message(session_id: str, role: str, content: str, db_path: Optional[Path] = None) -> None:
    """Persist a single chat message to the messages table."""
    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_messages(
    session_id: str, limit: int = 20, db_path: Optional[Path] = None
) -> list[tuple[str, str]]:
    """Return list of (role, content) ordered from oldest to newest among the first N messages.
    Deterministic under timestamp ties using rowid order. Matches test expectations.
    """
    conn = connect(db_path)
    try:
        cur = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (session_id, int(limit)),
        )
        rows = cur.fetchall()
        return [(r[0], r[1]) for r in rows]
    finally:
        conn.close()


# Settings persistence
def set_settings(values: dict, db_path: Optional[Path] = None) -> None:
    """Persist settings as key/value strings. Values are JSON-serialized."""
    if not isinstance(values, dict):
        return
    conn = connect(db_path)
    try:
        cur = conn.cursor()
        for k, v in values.items():
            cur.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(k), json.dumps(v)),
            )
        conn.commit()
    finally:
        conn.close()


def get_settings(db_path: Optional[Path] = None) -> dict:
    """Load all settings into a dict, JSON-deserializing values where possible."""
    conn = connect(db_path)
    try:
        cur = conn.execute("SELECT key, value FROM settings")
        out: dict = {}
        for k, v in cur.fetchall():
            try:
                out[k] = json.loads(v)
            except Exception:
                out[k] = v
        return out
    finally:
        conn.close()


# Runs persistence
def add_run(
    session_id: str,
    flow: str,
    question: str,
    answer: str,
    citations: list | None,
    timings: dict | None,
    metrics: dict | None,
    cost: float | None,
    db_path: Optional[Path] = None,
) -> None:
    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO runs(id, session_id, flow, question, answer, citations_json, timings_json, metrics_json, cost) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                __import__("uuid").uuid4().hex,
                session_id,
                flow,
                question,
                answer,
                json.dumps(citations or []),
                json.dumps(timings or {}),
                json.dumps(metrics or {}),
                float(cost or 0.0),
            ),
        )
        conn.commit()
    finally:
        conn.close()
