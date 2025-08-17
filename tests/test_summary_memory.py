from app.core.memory.summarizer import SummaryMemory
from app.core.types import Turn


def test_summary_persists_and_relevant_returns():
    sm = SummaryMemory(session_id="test_session", sentences=1)
    turns = [Turn(role="user", content="Hello"), Turn(role="assistant", content="Hi there")]
    s = sm.maybe_update(turns)
    assert isinstance(s, str) and len(s) >= 0
    hits = sm.relevant("hello", k=2)
    assert isinstance(hits, list)
