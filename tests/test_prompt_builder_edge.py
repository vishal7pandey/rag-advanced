from app.core.prompting.builder import build_answer_prompt, build_memory_text
from app.core.types import Chunk, RetrievedDoc


def make_rd(text: str, ord_: int) -> RetrievedDoc:
    ch = Chunk(id=f"c{ord_}", doc_id="d1", ord=ord_, text=text, meta={"path": f"p{ord_}.txt"})
    return RetrievedDoc(chunk=ch, score=1.0)


def test_build_memory_text_edges():
    # All None -> empty
    assert build_memory_text(None, None, None) == ""
    # Some present
    out = build_memory_text(["a", "b"], "sum", ["c"])
    assert "Memory (recent):" in out and "Memory (summary):" in out and "Memory (relevant):" in out


def test_build_answer_prompt_enumerate_and_docs():
    # Ensure template renders with enumerate available and docs list
    docs = [make_rd("txt1", 0), make_rd("txt2", 1)]
    prompt = build_answer_prompt("Q?", memory="M", docs=docs)
    assert "Q?" in prompt
    assert "txt1" in prompt and "txt2" in prompt
    assert "M" in prompt
