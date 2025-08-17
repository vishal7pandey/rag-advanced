from __future__ import annotations
import os
from typing import Dict, List, Iterator

from dotenv import load_dotenv
from app.core.utils import load_config

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    requests = None  # type: ignore

from app.core.types import RetrievedDoc


def generate_answer(prompt: str, model: str, offline: bool, docs: List[RetrievedDoc]) -> Dict:
    """Generate answer using configured provider.
    - provider: openai|ollama|stub (from config)
    - model: overrides config if provided
    Fallbacks to stub if provider unavailable.
    """
    load_dotenv()
    cfg = load_config()
    gcfg = cfg.get("generator", {})
    provider = str(gcfg.get("provider", "openai")).lower()
    timeout_s = float(gcfg.get("timeout_s", 30))
    model_name = model or str(gcfg.get("model", "gpt-4o-mini"))

    # Provider: stub or offline enforced
    if offline or provider == "stub":
        return _stub_answer(docs)

    if provider == "ollama":
        return _ollama_answer(
            prompt, docs, model=str(gcfg.get("ollama_model", model_name)), timeout_s=timeout_s
        )

    # Default: OpenAI
    return _openai_answer(prompt, docs, model=model_name, timeout_s=timeout_s)


def stream_answer(
    prompt: str, model: str, offline: bool, docs: List[RetrievedDoc]
) -> Iterator[str]:
    """Yield answer text incrementally based on configured provider.
    This is a non-breaking addition; existing `generate_answer` remains the default path.
    """
    load_dotenv()
    cfg = load_config()
    gcfg = cfg.get("generator", {})
    provider = str(gcfg.get("provider", "openai")).lower()
    timeout_s = float(gcfg.get("timeout_s", 30))
    model_name = model or str(gcfg.get("model", "gpt-4o-mini"))

    # Provider: stub or offline enforced
    if offline or provider == "stub":
        yield from _stub_stream(docs)
        return

    if provider == "ollama":
        ollama_model = str(gcfg.get("ollama_model", model_name))
        yield from _ollama_stream(prompt, model=ollama_model, timeout_s=timeout_s)
        return

    # Default: OpenAI
    yield from _openai_stream(prompt, model=model_name, timeout_s=timeout_s)


def _stub_answer(docs: List[RetrievedDoc]) -> Dict:
    text = "\n\n".join([d.chunk.text[:200] for d in docs])
    answer = "(offline stub) Based on context: " + (text[:500] or "No context available.")
    return {
        "answer_md": answer,
        "citations": _mk_citations(docs),
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


def _stub_stream(docs: List[RetrievedDoc]) -> Iterator[str]:
    text = "\n\n".join([d.chunk.text[:200] for d in docs])
    answer = "(offline stub) Based on context: " + (text[:500] or "No context available.")
    # simple word-wise stream
    for tok in answer.split():
        yield tok + " "


def _openai_answer(prompt: str, docs: List[RetrievedDoc], model: str, timeout_s: float) -> Dict:
    if (OpenAI is None) or not os.getenv("OPENAI_API_KEY"):
        return _stub_answer(docs)
    try:
        client = OpenAI()
        sys = "You are a grounded assistant. Cite sources as [^i] matching the provided context order."
        content = prompt
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": content}],
            temperature=0.2,
            timeout=timeout_s,
        )
        ans = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }
        return {"answer_md": ans, "citations": _mk_citations(docs), "usage": usage}
    except Exception:
        return _stub_answer(docs)


def _ollama_stream(prompt: str, model: str, timeout_s: float) -> Iterator[str]:
    if requests is None:
        yield ""
        return
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    url = base.rstrip("/") + "/api/generate"
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.2},
        }
        with requests.post(url, json=payload, timeout=timeout_s, stream=True) as r:  # type: ignore[arg-type]
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    import json

                    obj = json.loads(line)
                    piece = obj.get("response", "")
                    if piece:
                        yield str(piece)
                except Exception:
                    # ignore non-JSON lines
                    pass
    except Exception:
        return


def _openai_stream(prompt: str, model: str, timeout_s: float) -> Iterator[str]:
    if (OpenAI is None) or not os.getenv("OPENAI_API_KEY"):
        # Fallback: yield nothing to let caller decide, or a minimal message
        yield ""
        return
    try:
        client = OpenAI()
        sys = "You are a grounded assistant. Cite sources as [^i] matching the provided context order."
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=timeout_s,
            stream=True,
        )
        for event in stream:
            try:
                delta = event.choices[0].delta  # type: ignore[attr-defined]
                if delta and getattr(delta, "content", None):
                    yield str(delta.content)
            except Exception:
                # Be forgiving of event shapes
                pass
    except Exception:
        # On any error, just end the stream
        return


def _ollama_answer(prompt: str, docs: List[RetrievedDoc], model: str, timeout_s: float) -> Dict:
    # Use Ollama's /api/generate if available; otherwise stub
    if requests is None:
        return _stub_answer(docs)
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    url = base.rstrip("/") + "/api/generate"
    try:
        # Simple prompt; could switch to /api/chat for better formatting later
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        r = requests.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        ans = data.get("response", "")
        return {
            "answer_md": ans,
            "citations": _mk_citations(docs),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
    except Exception:
        return _stub_answer(docs)


def _mk_citations(docs: List[RetrievedDoc]):
    cites = []
    for i, d in enumerate(docs, start=1):
        cites.append({"marker": i, "path": d.chunk.meta.get("path", ""), "ord": d.chunk.ord})
    return cites
