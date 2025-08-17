from __future__ import annotations
from typing import List
from app.core.types import Turn
from app.core.storage import connect

# TextRank via sumy
from sumy.parsers.plaintext import PlaintextParser  # type: ignore
from sumy.nlp.tokenizers import Tokenizer  # type: ignore
from sumy.summarizers.text_rank import TextRankSummarizer  # type: ignore


class SummaryMemory:
    def __init__(self, session_id: str, sentences: int = 5):
        self.session_id = session_id
        self.sentences = sentences

    def maybe_update(self, turns: List[Turn]) -> str:
        """Create/update a summary for current session and return latest summary text."""
        full_text = "\n".join([f"{t.role}: {t.content}" for t in turns])
        if not full_text.strip():
            return ""
        # Try TextRank first; gracefully fall back if tokenizers are unavailable
        try:
            parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            sents = summarizer(parser.document, self.sentences)
            summary = " ".join(str(s) for s in sents)
        except Exception:
            # Fallback: naive sentence splitting, then take first N
            # Split on common sentence boundaries and newlines
            raw = full_text.replace("\r", "\n")
            # Prefer periods as boundaries; then newlines
            parts = []
            for block in raw.split("\n"):
                parts.extend([p.strip() for p in block.split(". ") if p.strip()])
            if not parts:
                parts = [full_text.strip()]
            summary = ". ".join(parts[: max(1, self.sentences)])
        conn = connect()
        conn.execute(
            "INSERT INTO memory(session_id, summary) VALUES (?, ?)", (self.session_id, summary)
        )
        conn.commit()
        conn.close()
        return summary

    def relevant(self, query: str, k: int = 3) -> List[str]:
        # Simple latest-k summaries; can be replaced by FAISS over summaries later
        conn = connect()
        cur = conn.execute(
            "SELECT summary FROM memory WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (self.session_id, k),
        )
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]
