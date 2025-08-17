from __future__ import annotations
from typing import List, Sequence
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

from app.core.types import RetrievedDoc

TEMPLATES = Path(__file__).resolve().parent / "templates"
_env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape())
# Allow templates to use Python's enumerate directly (used in answer.j2)
_env.globals["enumerate"] = enumerate


def build_answer_prompt(
    question: str,
    memory: str,
    docs: List[RetrievedDoc],
    format_hint: str | None = None,
    persona_hint: str | None = None,
) -> str:
    """Render the main answer prompt.

    Args:
        question: User question
        memory: Prepared memory text (can be empty)
        docs: Retrieved documents (used as context)
        format_hint: Optional formatting instructions to guide the LLM output
        persona_hint: Optional persona/tone instructions to guide the LLM voice
    """
    tmpl = _env.get_template("answer.j2")
    return tmpl.render(
        question=question,
        memory=memory,
        chunks=[d.chunk for d in docs],
        format_hint=format_hint,
        persona_hint=persona_hint,
    )


def build_memory_text(
    memory_window: Sequence[str] | None,
    memory_summary: str | None,
    memory_hits: Sequence[str] | None,
) -> str:
    parts: List[str] = []
    if memory_window:
        parts.append("Memory (recent):\n" + "\n".join(memory_window))
    if memory_summary:
        parts.append("Memory (summary):\n" + memory_summary)
    if memory_hits:
        parts.append("Memory (relevant):\n" + "\n".join(memory_hits))
    return "\n\n".join(parts)
