from app.core.prompting.builder import build_answer_prompt
from app.core.types import Chunk, RetrievedDoc


def test_prompt_includes_memory_and_context_and_citations():
    mem = "Memory (summary): prior context"
    docs = [RetrievedDoc(chunk=Chunk(id="c1", doc_id="d", ord=0, text="Alpha", meta={}), score=1.0)]
    prompt = build_answer_prompt("What?", mem, docs)
    assert "Memory" in prompt
    assert "Context" in prompt
    assert "Alpha" in prompt
    # Prompt should instruct inline citation pattern even if no dynamic marker appears yet
    assert "[^i]" in prompt
