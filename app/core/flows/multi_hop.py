from __future__ import annotations
import os
import time
import re
from dataclasses import dataclass
from typing import Dict, List, Iterator, Tuple, Any

from dotenv import load_dotenv

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

from app.core.flows.base import Flow
from app.core.flows.guardrails import should_skip_advanced_flow
from app.core.generator import generate_answer
from app.core.prompting.builder import build_answer_prompt
from app.core.types import AnswerBundle, RetrievedDoc
from app.core.retrievers.dense import DenseConfig, retrieve_dense
from app.core.retrievers.hybrid import rrf_fuse
from app.core.flows import standard as standard_flow
from app.core.metrics.raglite import lite_metrics, delta_precision_lite
from app.core.metrics.token_cost import estimate_tokens_cost
from app.core.retrievers.rerank import rerank_bge_topn, mmr_rerank, llm_rerank
from app.core.embeddings import get_default_embedder
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "prompting" / "templates"
_env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape())


def _dedupe_by_chunk_id(docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
    seen = set()
    out: List[RetrievedDoc] = []
    for d in docs:
        cid = getattr(d.chunk, "id", None) or f"{d.chunk.meta.get('path', '')}::{d.chunk.ord}"
        if cid in seen:
            continue
        seen.add(cid)
        out.append(d)
    return out


def _gen_subquestions(question: str, subq_max: int, model: str, offline: bool) -> List[str]:
    """Generate up to subq_max sub-questions using condense_query.j2 template.
    Falls back to a trivial single-subq in offline mode.
    """
    load_dotenv()
    tmpl = _env.get_template("condense_query.j2")
    prompt = tmpl.render(question=question, max_n=subq_max)
    if offline or (OpenAI is None) or not os.getenv("OPENAI_API_KEY"):
        return [question]
    client = OpenAI()
    sys = "You decompose complex questions into 2-3 concise subquestions."
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    txt = resp.choices[0].message.content or ""
    # Parse lines; accept bullets or numbered
    lines = [line.strip(" -\t") for line in txt.splitlines() if line.strip()]
    # Keep up to subq_max
    out: list[str] = []
    for line in lines:
        if line and len(out) < subq_max:
            # drop any numbering like "1. ..."
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            out.append(line.strip())
    if not out:
        out = [question]
    return out


@dataclass
class MultiHopParams:
    subq_max: int = 3
    k_each: int = 5
    top_k_final: int = 8
    rrf_k: int = 60
    rerank: bool = False
    rerank_top_n: int = 6
    auto_mode: bool = False  # Enable auto-skip only when explicitly requested
    guardrails_config: Dict | None = None
    rerank_strategy: str = "mmr"  # none|mmr|cross_encoder|llm_judge
    mmr_lambda: float = 0.5
    llm_judge_model: str = "gpt-4o-mini"


class MultiHopFlow(Flow):
    def __init__(
        self,
        offline: bool,
        gen_model: str,
        emb_st: str | None,
        emb_oa: str | None,
        params: MultiHopParams,
    ):
        self.offline = offline
        self.gen_model = gen_model
        self.params = params
        self.emb_st = emb_st
        self.emb_oa = emb_oa

    def _apply_rerank(self, question: str, docs: List[RetrievedDoc]):
        info: Dict = {"strategy": (self.params.rerank_strategy or "").lower()}
        strat = (self.params.rerank_strategy or "").lower()
        if strat == "none":
            return docs, info
        if strat == "mmr":
            embedder = get_default_embedder(
                offline=self.offline, st_model=self.emb_st, oa_model=self.emb_oa
            )
            qv = embedder([question])[0]
            dv = embedder([d.chunk.text for d in docs])
            head_n = max(0, min(self.params.rerank_top_n, len(docs)))
            selected = mmr_rerank(
                qv, docs, dv, lambda_param=float(self.params.mmr_lambda), top_n=head_n
            )

            def _key(d: RetrievedDoc):
                return getattr(d.chunk, "id", None) or (d.chunk.meta.get("path", ""), d.chunk.ord)

            sel_keys = {_key(d) for d in selected}
            tail = [d for d in docs if _key(d) not in sel_keys]
            return selected + tail, info
        if strat == "llm_judge":
            rr, cost = llm_rerank(
                question, docs, model=self.params.llm_judge_model, top_n=self.params.rerank_top_n
            )
            info.update(
                {"llm": self.params.llm_judge_model, **({} if not isinstance(cost, dict) else cost)}
            )
            if cost.get("error") or self.offline:
                head = rerank_bge_topn(question, docs, top_n=self.params.rerank_top_n)
                return head, {"strategy": "cross_encoder", "fallback_from": "llm_judge"}

            def _key(d: RetrievedDoc):
                return getattr(d.chunk, "id", None) or (d.chunk.meta.get("path", ""), d.chunk.ord)

            sel_keys = {_key(d) for d in rr}
            tail = [d for d in docs if _key(d) not in sel_keys]
            return rr + tail, info
        head = rerank_bge_topn(question, docs, top_n=self.params.rerank_top_n)
        return head, info

    def run(self, question: str, session_state: Dict) -> AnswerBundle:
        # Auto-skip guardrails
        if self.params.auto_mode:
            should_skip, reason = should_skip_advanced_flow(
                question, "multi_hop", self.params.guardrails_config
            )
            if should_skip:
                # Fall back to standard flow
                fallback = standard_flow.StandardFlow(
                    self.offline,
                    self.gen_model,
                    self.emb_st,
                    self.emb_oa,
                    standard_flow.StandardParams(
                        top_k=self.params.top_k_final,
                        rerank=self.params.rerank,
                        rerank_top_n=self.params.rerank_top_n,
                        rerank_strategy=self.params.rerank_strategy,
                        mmr_lambda=self.params.mmr_lambda,
                        llm_judge_model=self.params.llm_judge_model,
                        guardrails_config=self.params.guardrails_config,
                    ),
                )
                bundle = fallback.run(question, session_state)
                # Add skip reason to extras
                bundle.extras["multi_hop_skipped"] = reason
                return bundle

        # A) subquestion generation (guardrails-aware cap)
        t_subq0 = time.time()
        gr = self.params.guardrails_config or {}
        try:
            cap_subqs = int(gr.get("multi_hop_max_subqs", self.params.subq_max))
        except Exception:
            cap_subqs = self.params.subq_max
        eff_subq_max = max(1, min(self.params.subq_max, cap_subqs))
        subqs = _gen_subquestions(question, eff_subq_max, self.gen_model, self.offline)
        t_subq = time.time() - t_subq0

        # B) per-hop retrieval
        hop_docs: List[List[RetrievedDoc]] = []
        t0 = time.time()
        for sq in subqs:
            docs, _ = retrieve_dense(
                sq,
                DenseConfig(
                    top_k=self.params.k_each,
                    offline=self.offline,
                    st_model=self.emb_st,
                    oa_model=self.emb_oa,
                ),
            )
            hop_docs.append(docs)
        # C) merge via iterative RRF
        if not hop_docs:
            fused: List[RetrievedDoc] = []
        else:
            fused = hop_docs[0]
            for i in range(1, len(hop_docs)):
                fused = rrf_fuse(fused, hop_docs[i], k=self.params.rrf_k)
        fused = _dedupe_by_chunk_id(fused)[: self.params.top_k_final]
        pre_docs = list(fused)
        # Optional rerank (guardrails-aware)
        gr = self.params.guardrails_config or {}
        do_rerank = bool(self.params.rerank) and not bool(gr.get("disable_rerank", False))
        rr_info: Dict = {}
        if do_rerank:
            cap = int(gr.get("rerank_top_n_cap", self.params.rerank_top_n))
            old_rtn = self.params.rerank_top_n
            try:
                self.params.rerank_top_n = max(1, min(old_rtn, cap))
                fused, rr_info = self._apply_rerank(question, fused)
            finally:
                self.params.rerank_top_n = old_rtn
        t_retrieve = time.time() - t0

        # D) build prompt (with memory)
        mem_text = ""
        if isinstance(session_state, dict):
            mem_text = session_state.get("memory_text", "")
        fmt_hint = None
        if isinstance(session_state, dict):
            val = session_state.get("format_hint")
            fmt_hint = str(val) if isinstance(val, str) and val.strip() else None
        per_hint = None
        if isinstance(session_state, dict):
            valp = session_state.get("persona_hint")
            per_hint = str(valp) if isinstance(valp, str) and valp.strip() else None
        # Build merge prompt specialized for multi-hop
        try:
            merge_tmpl = _env.get_template("multi_hop_merge.j2")
            hop_ctx = [
                {
                    "subq": subqs[i],
                    "contexts": [d.chunk.text for d in hop_docs[i]],
                }
                for i in range(len(hop_docs))
            ]
            merged_contexts = [d.chunk.text for d in fused]
            prompt = merge_tmpl.render(
                question=question,
                memory=mem_text,
                hops=hop_ctx,
                merged_contexts=merged_contexts,
                format_hint=fmt_hint,
                persona_hint=per_hint,
            )
        except Exception:
            # Fallback to generic builder
            prompt = build_answer_prompt(
                question, memory=mem_text, docs=fused, format_hint=fmt_hint, persona_hint=per_hint
            )

        # E) generate
        t1 = time.time()
        out = generate_answer(prompt, self.gen_model, self.offline, fused)
        t_generate = time.time() - t1

        timings = {"t_subq": t_subq, "t_retrieve": t_retrieve, "t_generate": t_generate}
        # Metrics and debug enrichment
        mets = lite_metrics(out["answer_md"], fused)

        def _mk_list(items):
            arr = []
            for rd in items:
                try:
                    arr.append(
                        {
                            "path": rd.chunk.meta.get("path", ""),
                            "ord": rd.chunk.ord,
                            "score": float(rd.score),
                        }
                    )
                except Exception:
                    pass
            return arr

        mets["retrieved"] = _mk_list(pre_docs)
        mets["reranked"] = _mk_list(fused)
        if do_rerank:
            mets["delta_precision_lite"] = delta_precision_lite(pre_docs, fused)
            old_map = {
                (d.chunk.meta.get("path", ""), d.chunk.ord): float(d.score) for d in pre_docs
            }
            new_map = {(d.chunk.meta.get("path", ""), d.chunk.ord): float(d.score) for d in fused}
            deltas = []
            for k, new_s in new_map.items():
                old_s = old_map.get(k)
                if old_s is not None:
                    deltas.append({"path": k[0], "ord": k[1], "delta": new_s - old_s})
            mets["rerank_deltas"] = deltas
        usage = out["usage"]
        usage["cost_estimate"] = estimate_tokens_cost(
            self.gen_model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        )
        extras = {
            "hops": [{"subq": subqs[i], "retrieved": hop_docs[i]} for i in range(len(hop_docs))],
            "prompt": {"text": prompt},
            "memory": {"text": mem_text},
            "format_hint": fmt_hint,
            "persona_hint": per_hint,
        }
        if rr_info:
            extras["rerank_info"] = rr_info
        return AnswerBundle(
            answer_md=out["answer_md"],
            citations=out["citations"],
            usage=usage,
            timings=timings,
            metrics=mets,
            retrieved=fused,
            extras=extras,
        )

    def run_stream(
        self, question: str, session_state: Dict
    ) -> Tuple[Iterator[str], Dict[str, Any]]:
        """Fallback streaming: reuse run() and stream the final text token-by-token.
        Provides a ctx dict compatible with orchestrator/UI expectations.
        """
        bundle = self.run(question, session_state)

        def _iter() -> Iterator[str]:
            for tok in (bundle.answer_md or "").split():
                yield tok + " "

        ctx: Dict[str, Any] = {
            "retrieved": bundle.retrieved,
            "prompt": (bundle.extras or {}).get("prompt", {}).get("text", ""),
            "t_retrieve": (bundle.timings or {}).get("t_retrieve"),
            "t_generate": (bundle.timings or {}).get("t_generate"),
        }
        try:
            ts = (bundle.timings or {}).get("t_subq")
            if ts is not None:
                ctx["t_subq"] = ts
        except Exception:
            pass
        try:
            hops = (bundle.extras or {}).get("hops")
            if hops:
                ctx["hops"] = hops
        except Exception:
            pass
        return _iter(), ctx
