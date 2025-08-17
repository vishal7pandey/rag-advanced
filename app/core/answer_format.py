from __future__ import annotations
from typing import List, Dict, Any, Optional

# Lightweight Markdown helpers for the human-facing message


def _fmt_source_line(c: Dict[str, Any]) -> str:
    """
    UI-safe source line formatter.
    Expects keys: marker, title, doc_short_id, chunk_ord (optional)
    """
    m = c.get("marker")
    title = str(c.get("title") or "Source")
    sid = str(c.get("doc_short_id") or "")
    ord_val = c.get("ord") or c.get("chunk_ord")
    ord_str = f" (chunk {ord_val})" if ord_val is not None else ""
    if sid:
        return f"[^{m}] {title} · {sid}{ord_str}"
    return f"[^{m}] {title}{ord_str}"


def render_sources(citations_ui: List[Dict[str, Any]]) -> str:
    if not citations_ui:
        return ""
    lines = ["### Sources"]
    for c in citations_ui:
        try:
            lines.append(_fmt_source_line(c))
        except Exception:
            pass
    return "\n".join(lines)


def render_status_footer(
    model: str, flow: str, rerank: bool, elapsed_s: float, online: bool, cost: Optional[float]
) -> str:
    online_badge = "🟢 Online" if online else "🔒 Offline"
    c = f" • cost ${float(cost):.4f}" if cost is not None else ""
    return f"{online_badge} • {model} • flow: {flow} (rerank={'on' if rerank else 'off'}) • {elapsed_s:.1f}s{c}"


def render_markdown(
    answer_text: str,
    citations_ui: List[Dict[str, Any]],
    status_footer: Optional[str] = None,
    bullets: Optional[List[str]] = None,
    caveat: Optional[str] = None,
) -> str:
    """
    Assemble the final human-facing Markdown card.

    - answer_text: brief/direct answer text
    - bullets: optional key points to list under the answer
    - citations_ui: deterministic, redacted sources (for a Sources section)
    - caveat: optional soft warning string
    - status_footer: single-line status (model/flow/time/cost)
    """
    parts: List[str] = []
    if answer_text:
        parts.append(answer_text.strip())
    if caveat:
        parts.append(f"_{caveat}_")
    if bullets:
        blines = ["**Key points**"] + [f"- {b}" for b in bullets if b]
        parts.append("\n".join(blines))
    src = render_sources(citations_ui)
    if src:
        parts.append(src)
    if status_footer:
        parts.append(f"\n---\n{status_footer}")
    return "\n\n".join([p for p in parts if p])
