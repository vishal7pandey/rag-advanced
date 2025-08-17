from __future__ import annotations
import re
from typing import Iterable, List, Dict, Any
import os


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


escape_re = re.escape


def _highlight_terms(text: str, terms: List[str]) -> str:
    if not terms:
        return text
    try:
        pattern = re.compile("|".join([escape_re(t) for t in terms if t]), re.IGNORECASE)
        return pattern.sub(lambda m: f"**{m.group(0)}**", text)
    except Exception:
        return text


def make_snippet(raw: str, terms: List[str], max_len: int = 120) -> str:
    t = _collapse_ws(raw)
    if len(t) <= max_len:
        return _highlight_terms(t, terms)
    # try to center around the first term hit; else lead
    for term in terms:
        if not term:
            continue
        m = re.search(re.escape(term), t, re.IGNORECASE)
        if m:
            start = max(m.start() - max_len // 3, 0)
            end = min(start + max_len, len(t))
            frag = t[start:end]
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(t) else ""
            return _highlight_terms(prefix + frag + suffix, terms)
    # no term found; lead fragment
    return (t[:max_len] + "…") if len(t) > max_len else t


def build_snippet_rows(retrieved: Iterable[object], query_text: str) -> List[Dict[str, Any]]:
    # naive termization; you can swap in a smarter keyword extractor later
    terms = [w for w in re.split(r"[^\w]+", query_text or "") if len(w) > 2][:6]
    rows: List[Dict[str, Any]] = []
    for r in retrieved or []:
        try:
            meta = getattr(getattr(r, "chunk", object()), "meta", {}) or {}
            # Prefer human title/name; else basename of path; else doc_id
            title = meta.get("title") or meta.get("name")
            if not title:
                p = meta.get("path")
                if p:
                    title = os.path.basename(str(p))
            if not title:
                title = getattr(getattr(r, "chunk", object()), "doc_id", "")
            rows.append(
                {
                    "title": title,
                    "doc_id_short": str(getattr(getattr(r, "chunk", object()), "doc_id", ""))[:8],
                    "chunk_id": getattr(getattr(r, "chunk", object()), "id", None)
                    or getattr(getattr(r, "chunk", object()), "ord", ""),
                    "snippet": make_snippet(
                        getattr(getattr(r, "chunk", object()), "text", ""), terms, 120
                    ),
                }
            )
        except Exception:
            continue
    return rows
