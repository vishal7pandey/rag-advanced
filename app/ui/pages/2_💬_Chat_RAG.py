from __future__ import annotations
import streamlit as st
import os as _os

from app.core.utils import load_config, get_logger
from typing import Any, List, Optional, cast
from app.core.types import StreamCtx, CitationUIMap, CitationEnvMap, Turn
from app.core.orchestrator import Orchestrator
from app.core.memory.window import WindowMemory
from app.core.memory.summarizer import SummaryMemory
from app.core.prompting.builder import build_memory_text
from app.core.storage import add_message, get_recent_messages, add_run, get_recent_runs
from app.core.metrics.ragas_wrap import eval_ragas
from app.core.citations import build_citation_map
from app.core.metrics.raglite import lite_metrics
from app.core.metrics.token_estimate import estimate_tokens
from app.core.metrics.token_cost import estimate_tokens_cost
from app.core.answer_format import render_status_footer, render_markdown
from app.core.envelope import build_envelope
from app.core.answer_guard import should_no_answer, caveat_text, NO_ANSWER_MD
from app.core.snippets import build_snippet_rows
import uuid

st.title("💬 Chat RAG")
# Evidence-first UX toggle (session)
if "evidence_first" not in st.session_state:
    st.session_state.evidence_first = False
with st.sidebar:
    st.checkbox("Evidence-first (show sources before answer)", key="evidence_first")
cfg = load_config()
# Header badges: Online/Offline, Provider, Model
_has_key = bool((_os.getenv("OPENAI_API_KEY") or "").strip())
_provider = "OpenAI" if _has_key else "Offline"
_model = cfg.get("models", {}).get("generation", "gpt-4o-mini")
st.caption(f"{'🟢 Online' if _has_key else '🔒 Offline'} • Provider: {_provider} • Model: {_model}")
# Auto mode: Orchestrator decides retrieval/generation; memory from config
enable_memory = True
try:
    window = int(cfg["memory"]["window_size"])
except Exception:
    window = 6
try:
    summarize = bool(cfg["memory"]["summarize"])
except Exception:
    summarize = True
sum_every = int(cfg.get("memory", {}).get("summary_every_n", 6))


def _stream_text(text: str):
    for tok in text.split():
        yield tok + " "


def _microbatch(it, flush_ms: int = 40, max_chars: int = 400, on_tick=None):
    """Aggregate tiny chunks to reduce flicker and optionally report elapsed time.
    on_tick receives elapsed_seconds periodically (~200ms).
    """
    import time as __t

    start = __t.time()
    last_flush = start
    last_tick = start
    buf: list[str] = []
    size = 0
    for chunk in it:
        s = str(chunk or "")
        buf.append(s)
        size += len(s)
        now = __t.time()
        if on_tick and (now - last_tick) >= 0.2:
            try:
                on_tick(now - start)
            except Exception:
                pass
            last_tick = now
        if (now - last_flush) * 1000.0 >= float(flush_ms) or size >= max_chars:
            out = "".join(buf)
            buf.clear()
            size = 0
            last_flush = now
            if out:
                yield out
    if buf:
        yield "".join(buf)
    if on_tick:
        try:
            on_tick(__t.time() - start)
        except Exception:
            pass


def _ensure_text(x: Any) -> str:
    """Coerce a possibly list-like streamed output into a plain string.
    Streamlit's write_stream may return a string or list of chunks.
    """
    try:
        if isinstance(x, list):
            return "".join(str(t or "") for t in x)
        return str(x or "")
    except Exception:
        return str(x)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "wmemory" not in st.session_state:
    st.session_state.wmemory = WindowMemory()
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator()
if "hydrated" not in st.session_state:
    # Hydrate messages and window memory from DB (last N*2 to cover user+assistant pairs)
    try:
        recent = get_recent_messages(st.session_state.session_id, limit=window * 2)
        if recent:
            st.session_state.messages = [
                {"role": role, "content": content} for (role, content) in recent
            ]
            for role, content in recent:
                st.session_state.wmemory.add(role, content)
            st.session_state.turn_count = len(recent)
    except Exception:
        pass
    st.session_state.hydrated = True
    # Initialize cost accumulator from DB for this session
    try:
        runs: list[dict[str, object]] = get_recent_runs(limit=1000)
        session_id = st.session_state.session_id
        total_cost = 0.0
        for row in runs:
            try:
                if row.get("session_id") == session_id:
                    val_o = row.get("cost")
                    val = cast(float | int | str | None, val_o)
                    total_cost += float(val if val is not None else 0.0)
            except Exception:
                pass
        st.session_state.total_cost = total_cost
    except Exception:
        st.session_state.total_cost = 0.0

for m in st.session_state.messages:
    _avatar = "👤" if m["role"] == "user" else "🤖"
    with st.chat_message(m["role"], avatar=_avatar):
        st.markdown(m["content"])

# Desktop UX hint near input
st.caption("Tip: Shift+Enter for newline, Enter to send")

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.wmemory.add("user", prompt)
    st.session_state.turn_count += 1
    # persist user message
    try:
        add_message(st.session_state.session_id, "user", prompt)
    except Exception:
        pass
    with st.spinner("Thinking…"):
        # Build memory text
        memory_text = ""
        mem_window_texts: list[str] = []
        mem_summary_text = ""
        mem_hits: list[str] = []  # match build_memory_text expected type
        if enable_memory:
            recent_turns: List[Turn] = st.session_state.wmemory.get(window)
            mem_window_texts = [f"{t.role}: {t.content}" for t in recent_turns]
            if summarize and st.session_state.turn_count % int(sum_every) == 0:
                sm = SummaryMemory(session_id=st.session_state.session_id)
                mem_summary_text = sm.maybe_update(st.session_state.wmemory.turns)
            else:
                sm = SummaryMemory(session_id=st.session_state.session_id)
                # fetch latest summary if exists
                hits = sm.relevant("", k=1)
                if hits:
                    mem_summary_text = str(hits[0])
            memory_text = build_memory_text(
                mem_window_texts, mem_summary_text, mem_hits if mem_hits else None
            )
        # Orchestrator-driven streaming run
        orchestrator = st.session_state.orchestrator
        try:
            it, plan, ctx = orchestrator.plan_and_stream(
                prompt,
                session_state={
                    "window": window,
                    "summarize": summarize,
                    "memory_text": memory_text,
                },
            )
            ctx = cast(StreamCtx, ctx)
        except Exception as e:
            msg = str(e)
            if "No FAISS index" in msg:
                st.error("No FAISS index found. Please build the index in the Upload & Index page.")
            else:
                st.error(f"Error while answering: {msg}")
            st.stop()
    # Deterministic citations from merged context
    try:
        citations_ui: list[CitationUIMap]
        citations_env: list[CitationEnvMap]
        _cidx: dict[str, int]
        citations_ui, citations_env, _cidx = build_citation_map(ctx.get("retrieved", []))
    except Exception:
        citations_ui, citations_env, _cidx = [], [], {}

    # Evidence-first option (show before streaming answer)
    if st.session_state.evidence_first:
        if citations_ui:
            st.markdown("### Sources")
            for cit in citations_ui:
                try:
                    m = cit.get("marker")
                    title = str(cit.get("title") or "Source")
                    sid = str(cit.get("doc_short_id") or "")
                    ordv = cit.get("ord")
                    ord_str = f" (chunk {ordv})" if ordv is not None else ""
                    st.markdown(f"[^{m}] {title} · {sid}{ord_str}")
                except Exception as e:
                    st.markdown(f"- [Error displaying citation: {str(e)}]")
    # Stream assistant tokens
    with st.chat_message("assistant", avatar="🤖"):
        import time as _t

        _t0 = _t.time()
        timer_ph = st.empty()
        stats_ph = st.empty()
        # Mutable holder for cumulative streamed text
        _cum = {"text": ""}
        # Resolve model and prompt text for live token/cost estimates
        try:
            model_name = getattr(getattr(plan, "generation", object()), "model", _model)
        except Exception:
            model_name = _model
        _prompt_for_tokens = str(ctx.get("prompt") or "")

        def _update_stats(elapsed_s: float):
            try:
                pt, ct = estimate_tokens(model_name, _prompt_for_tokens, _cum["text"])
                tps = (ct / max(1e-3, elapsed_s)) if ct else 0.0
                cost_now = float(estimate_tokens_cost(str(model_name), pt, ct))
                # Typing indicator + live stats
                timer_ph.caption(f"Streaming… {elapsed_s:.1f}s")
                stats_ph.caption(f"~{ct} tokens • {tps:.1f} tok/s • est cost ${cost_now:.4f}")
            except Exception:
                # Best-effort; ignore estimation errors
                try:
                    timer_ph.caption(f"Streaming… {elapsed_s:.1f}s")
                except Exception:
                    pass

        def _on_tick(elapsed_s: float):
            _update_stats(elapsed_s)

        def _with_stats(gen):
            for chunk in gen:
                s = str(chunk or "")
                _cum["text"] += s
                # Update stats on each emitted chunk
                _update_stats(_t.time() - _t0)
                yield s

        streamed_text = st.write_stream(
            _with_stats(_microbatch(it, flush_ms=40, max_chars=400, on_tick=_on_tick))
        )
        t_generate = _t.time() - _t0
        streamed_text_s = _ensure_text(streamed_text)
        # Clear typing indicators once streaming completes
        try:
            timer_ph.empty()
            stats_ph.empty()
        except Exception:
            pass
        try:
            model_name = getattr(getattr(plan, "generation", object()), "model", _model)
            footer = render_status_footer(
                model=model_name,
                flow=plan.flow_name,
                rerank=bool(getattr(plan.retrieval, "rerank", False)),
                elapsed_s=float(t_generate),
                online=_has_key,
                cost=None,
            )
            st.caption(footer)
        except Exception:
            pass
        # store assistant turn in memory
        st.session_state.messages.append({"role": "assistant", "content": streamed_text_s})
        st.session_state.wmemory.add("assistant", streamed_text_s)
        st.session_state.turn_count += 1
        # persist assistant message
        try:
            add_message(st.session_state.session_id, "assistant", streamed_text_s)
        except Exception:
            pass
        # Compute metrics (lite) from streamed text and ctx
        mets = {}
        try:
            mets = lite_metrics(streamed_text_s, ctx.get("retrieved", []))
        except Exception:
            mets = {}
        # include plan (without persona), memory, prompt snapshots
        try:
            try:
                # Build a persona-free plan dict
                pr = getattr(plan, "retrieval", object())
                pg = getattr(plan, "generation", object())
                plan_clean = {
                    "flow_name": getattr(plan, "flow_name", "auto"),
                    "retrieval": getattr(pr, "__dict__", {}),
                    "generation": {
                        k: v for k, v in getattr(pg, "__dict__", {}).items() if k != "persona_hint"
                    },
                    "notes": list(getattr(plan, "notes", []) or []),
                }
            except Exception:
                plan_clean = {}
            mets["plan"] = plan_clean
            mem_obj: dict[str, object] = {}
            if mem_window_texts:
                mem_obj["window"] = list(mem_window_texts)
            if mem_summary_text:
                mem_obj["summary"] = mem_summary_text
            if mem_obj:
                mets["memory"] = mem_obj
            if ctx.get("prompt"):
                mets["prompt"] = {"text": ctx.get("prompt")}
        except Exception:
            pass
        ragas_enabled = bool(cfg.get("metrics", {}).get("ragas_enabled", False))
        if ragas_enabled:
            contexts = [rd.chunk.text for rd in ctx.get("retrieved", [])]
            ragas_model = cfg.get("metrics", {}).get("ragas_model", None)
            try:
                # Check prerequisites explicitly to avoid misleading 'skipped' toasts
                import os
                from importlib.util import find_spec

                has_key = bool(os.getenv("OPENAI_API_KEY"))
                has_pkgs = bool(find_spec("ragas")) and bool(find_spec("datasets"))
                if not has_key or not has_pkgs:
                    st.toast(
                        "RAGAS skipped (missing OPENAI_API_KEY or ragas/datasets not installed)",
                        icon="⚠️",
                    )
                else:
                    with st.spinner("Running RAGAS evaluation..."):
                        rmetrics = eval_ragas(prompt, streamed_text_s, contexts, model=ragas_model)
                        mets = {**mets, **rmetrics}
                    if not rmetrics:
                        st.toast("RAGAS returned no scores for this query.", icon="ℹ️")
            except Exception as e:
                st.toast(f"RAGAS error: {e}", icon="❌")
        # persist run row
        # Compute and persist cost, update session total
        # Token usage and cost estimate (post-stream)
        pt: int | None = None
        ct: int | None = None
        run_cost: float = 0.0
        try:
            model_for_cost = getattr(plan.generation, "model", None) or cfg.get("models", {}).get(
                "generation", "gpt-4o-mini"
            )
            prompt_text_for_tokens = str(ctx.get("prompt") or "")
            pt, ct = estimate_tokens(model_for_cost, prompt_text_for_tokens, streamed_text_s)
            run_cost = float(estimate_tokens_cost(str(model_for_cost), pt, ct))
            # Update session running total
            _prev_total_o = st.session_state.get("total_cost", 0.0)
            _prev_total = float(cast(float | int | str, _prev_total_o))
            st.session_state["total_cost"] = _prev_total + run_cost
        except Exception:
            pt, ct, run_cost = None, None, 0.0
        # Consolidate final human-facing card (no-answer gating + caveat)
        exit_state = "ok"
        retrieved_list = ctx.get("retrieved", []) or []
        try:
            cp = mets.get("context_precision_lite") if isinstance(mets, dict) else None
        except Exception:
            cp = None
        footer_final: Optional[str]
        try:
            model_name = getattr(getattr(plan, "generation", object()), "model", _model)
            footer_final = render_status_footer(
                model=model_name,
                flow=plan.flow_name,
                rerank=bool(getattr(plan.retrieval, "rerank", False)),
                elapsed_s=float(t_generate),
                online=_has_key,
                cost=run_cost,
            )
        except Exception:
            footer_final = None
        citations_env_out: list[CitationEnvMap] = citations_env
        try:
            if should_no_answer(len(retrieved_list), cp):
                final_md = NO_ANSWER_MD
                exit_state = "no_answer"
                citations_env_out = []
            else:
                cav = caveat_text(mets if isinstance(mets, dict) else {})
                final_md = render_markdown(
                    answer_text=streamed_text_s.strip(),
                    citations_ui=cast(list[dict[str, Any]], citations_ui),
                    status_footer=footer_final,
                    bullets=None,
                    caveat=cav,
                )
            # Replace the stored assistant message with consolidated markdown for future re-renders
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                st.session_state.messages[-1]["content"] = final_md
        except Exception:
            final_md = streamed_text_s
        # Build machine envelope and log + persist inside metrics
        try:
            usage_obj = {}
            if pt is not None and ct is not None:
                usage_obj = {
                    "prompt_tokens": int(pt),
                    "completion_tokens": int(ct),
                    "estimated_cost": float(run_cost),
                }
            env = build_envelope(
                session_id=st.session_state.session_id,
                run_id=None,
                question=prompt,
                answer_text=final_md or streamed_text_s,
                plan=plan,
                retrieved=retrieved_list,
                citations_env=cast(list[dict[str, Any]], citations_env_out),
                timings={
                    "t_retrieve": float(ctx.get("t_retrieve") or 0.0),
                    "t_generate": float(t_generate),
                },
                usage=usage_obj,
                metrics=dict(mets or {}),
                persona_hint=getattr(getattr(plan, "generation", object()), "persona_hint", None),
                format_hint=getattr(getattr(plan, "generation", object()), "format_hint", None),
                exit_state=exit_state,
                safety={},
            )
            # Structured log
            try:
                get_logger().info("rag.run_envelope", envelope=env)
            except Exception:
                pass
            # Persist envelope JSON under metrics
            try:
                if isinstance(mets, dict):
                    mets = dict(mets)
                    mets["envelope"] = env
            except Exception:
                pass
        except Exception:
            pass
        try:
            add_run(
                session_id=st.session_state.session_id,
                flow=plan.flow_name if "plan" in locals() else "auto",
                question=prompt,
                answer=final_md or streamed_text_s,
                citations=cast(list[dict[str, Any]], citations_env_out),
                timings={
                    "t_retrieve": float(ctx.get("t_retrieve") or 0.0),
                    "t_generate": float(t_generate),
                },
                metrics=mets,
                cost=run_cost,
            )
        except Exception:
            pass
        # Now that persistence is done, rerun once to swap the streamed placeholder with the consolidated card
        try:
            st.rerun()
        except Exception:
            pass
        try:
            if "total_cost" not in st.session_state:
                st.session_state.total_cost = 0.0
            st.session_state.total_cost = float(st.session_state.total_cost) + run_cost
        except Exception:
            pass
        # Evidence snippets (sanitized) + optional memory
        with st.expander("Show context", expanded=False):
            try:
                rows = build_snippet_rows(retrieved_list, prompt)
                for row in rows:
                    label = f"{row['title']} [doc:{row['doc_id_short']}#{row['chunk_id']}]"
                    st.markdown(f"- **{label}** — {row['snippet']}")
            except Exception:
                pass
            if enable_memory and (mem_window_texts or mem_summary_text):
                st.markdown("#### Memory")
                if mem_window_texts:
                    st.markdown("- Recent:")
                    for line in mem_window_texts:
                        st.write(line)
                if mem_summary_text:
                    st.markdown("- Summary:")
                    st.write(mem_summary_text)
        st.markdown("---")
        if "delta_precision_lite" in mets:
            st.caption(
                f"Metrics: answer_relevancy_lite={mets.get('answer_relevancy_lite', 0):.3f}, context_precision_lite={mets.get('context_precision_lite', 0):.3f}, "
                f"groundedness_lite={mets.get('groundedness_lite', 0):.3f}, Δprecision={mets.get('delta_precision_lite', 0):.3f}"
            )
        else:
            st.caption(
                f"Metrics: answer_relevancy_lite={mets.get('answer_relevancy_lite', 0):.3f}, context_precision_lite={mets.get('context_precision_lite', 0):.3f}, "
                f"groundedness_lite={mets.get('groundedness_lite', 0):.3f}"
            )
        tr_o = ctx.get("t_retrieve")
        tr = float(cast(float | int | str, tr_o)) if tr_o is not None else 0.0
        timings_obj = {
            "t_retrieve": tr,
            "t_generate": float(t_generate),
        }
        ths = ctx.get("t_hyde_seed")
        t_seed = float(cast(float | int | str, ths)) if ths is not None else None
        t_subq = None
        if t_seed is not None:
            st.caption(
                f"Timings: hyde_seed={t_seed:.3f}s, retrieve={timings_obj.get('t_retrieve', 0):.3f}s, generate={timings_obj.get('t_generate', 0):.3f}s"
            )
        elif t_subq is not None:
            st.caption(
                f"Timings: subq={t_subq:.3f}s, retrieve={timings_obj.get('t_retrieve', 0):.3f}s, generate={timings_obj.get('t_generate', 0):.3f}s"
            )
        else:
            st.caption(
                f"Timings: retrieve={timings_obj.get('t_retrieve', 0):.3f}s, generate={timings_obj.get('t_generate', 0):.3f}s"
            )
        # Cost footer
        try:
            if pt is not None and ct is not None:
                _tc_o = st.session_state.get("total_cost", 0.0)
                _tc = float(cast(float | int | str, _tc_o))
                st.caption(
                    f"Cost: this run ≈ ${run_cost:.4f} (tokens: prompt={int(pt)}, completion={int(ct)}), "
                    f"session total ≈ ${_tc:.4f}"
                )
            else:
                _tc_o2 = st.session_state.get("total_cost", 0.0)
                _tc2 = float(cast(float | int | str, _tc_o2))
                st.caption(f"Cost: this run ≈ ${run_cost:.4f}, session total ≈ ${_tc2:.4f}")
        except Exception:
            pass
