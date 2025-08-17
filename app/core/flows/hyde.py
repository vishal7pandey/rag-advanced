from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

from app.core.flows.base import Flow
from app.core.flows.guardrails import should_skip_advanced_flow
import app.core.generator as generator
from app.core.prompting.builder import build_answer_prompt
from app.core.types import AnswerBundle, RetrievedDoc
import app.core.retrievers.dense as dense_retriever
from app.core.retrievers.hybrid import rrf_fuse
from app.core.flows import standard as standard_flow
from app.core.metrics.raglite import lite_metrics, delta_precision_lite
from app.core.retrievers.rerank import rerank_bge_topn, mmr_rerank, llm_rerank
from app.core.metrics.token_cost import estimate_tokens_cost
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
from app.core.embeddings import get_default_embedder

TEMPLATES = Path(__file__).resolve().parents[1] / "prompting" / "templates"
_env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape())


# Expose patchable aliases for tests (forwarding wrappers)
def retrieve_dense(query, cfg):
    return dense_retriever.retrieve_dense(query, cfg)


DenseConfig = dense_retriever.DenseConfig


def generate_answer(prompt, model, offline, docs):
    return generator.generate_answer(prompt, model, offline, docs)


def stream_answer(prompt, model, offline, docs):
    return generator.stream_answer(prompt, model, offline, docs)


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


def _generate_hyde_seed(question: str, model: str, offline: bool) -> str:
    """Generate a short hypothetical passage describing the answer/topic using hyde_seed.j2 template.
    Falls back to offline stub if OpenAI is unavailable or offline=True.
    """
    load_dotenv()
    tmpl = _env.get_template("hyde_seed.j2")
    prompt = tmpl.render(question=question)
    if offline or (OpenAI is None) or not os.getenv("OPENAI_API_KEY"):
        return f"Hypothetical paragraph about: {question}"
    client = OpenAI()
    sys = "You are helpful and concise."
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


@dataclass
class HyDEParams:
    k_base: int = 12
    k_hyde: int = 12
    top_k_final: int = 10
    rrf_k: int = 60
    rerank: bool = False
    rerank_top_n: int = 6
    auto_mode: bool = True  # Enable auto-skip for simple queries
    guardrails_config: Dict | None = None
    rerank_strategy: str = "mmr"  # none|mmr|cross_encoder|llm_judge
    mmr_lambda: float = 0.5
    llm_judge_model: str = "gpt-4o-mini"


class HyDEFlow(Flow):
    def __init__(
        self,
        offline: bool,
        gen_model: str,
        emb_st: str | None,
        emb_oa: str | None,
        params: HyDEParams,
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
                question, "hyde", self.params.guardrails_config
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
                    ),
                )
                bundle = fallback.run(question, session_state)
                # Add skip reason to extras
                bundle.extras["hyde_skipped"] = reason
                return bundle

        # Step A: HyDE seed generation (guardrails-aware)
        gr = self.params.guardrails_config or {}
        t_seed0 = time.time()
        # use a cheap model: reuse self.gen_model to keep config simple; can swap to mini via config
        seed = _generate_hyde_seed(question, self.gen_model, self.offline)
        # Cap seed tokens if configured (rough whitespace tokenization)
        try:
            max_seed = int(gr.get("hyde_seed_max_tokens", 0) or 0)
        except Exception:
            max_seed = 0
        if max_seed > 0:
            toks = seed.split()
            if len(toks) > max_seed:
                seed = " ".join(toks[:max_seed])
        t_hyde_seed = time.time() - t_seed0

        # Step B: retrieve on seed and base
        t0 = time.time()
        de_hyde, _ = retrieve_dense(
            seed,
            DenseConfig(
                top_k=self.params.k_hyde,
                offline=self.offline,
                st_model=self.emb_st,
                oa_model=self.emb_oa,
            ),
        )
        de_base, _ = retrieve_dense(
            question,
            DenseConfig(
                top_k=self.params.k_base,
                offline=self.offline,
                st_model=self.emb_st,
                oa_model=self.emb_oa,
            ),
        )
        fused = rrf_fuse(de_hyde, de_base, k=self.params.rrf_k)
        fused = _dedupe_by_chunk_id(fused)[: self.params.top_k_final]
        pre_docs = list(fused)
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

        # Build prompt with memory
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
        prompt = build_answer_prompt(
            question, memory=mem_text, docs=fused, format_hint=fmt_hint, persona_hint=per_hint
        )

        # Generate answer
        t1 = time.time()
        out = generate_answer(prompt, self.gen_model, self.offline, fused)
        t_generate = time.time() - t1

        timings = {"t_hyde_seed": t_hyde_seed, "t_retrieve": t_retrieve, "t_generate": t_generate}
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

    def run_stream(self, question: str, session_state: Dict):
        # Auto-skip guardrails
        if self.params.auto_mode:
            should_skip, reason = should_skip_advanced_flow(
                question, "hyde", self.params.guardrails_config
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
                    ),
                )
                it, ctx = fallback.run_stream(question, session_state)
                ctx["hyde_skipped"] = reason
                return it, ctx

        # HyDE seed generation (non-streamed, guardrails-aware)
        gr = self.params.guardrails_config or {}
        t_seed0 = time.time()
        seed = _generate_hyde_seed(question, self.gen_model, self.offline)
        try:
            max_seed = int(gr.get("hyde_seed_max_tokens", 0) or 0)
        except Exception:
            max_seed = 0
        if max_seed > 0:
            toks = seed.split()
            if len(toks) > max_seed:
                seed = " ".join(toks[:max_seed])
        t_hyde_seed = time.time() - t_seed0

        # Retrieve on seed and base
        t0 = time.time()
        de_hyde, _ = retrieve_dense(
            seed,
            DenseConfig(
                top_k=self.params.k_hyde,
                offline=self.offline,
                st_model=self.emb_st,
                oa_model=self.emb_oa,
            ),
        )
        de_base, _ = retrieve_dense(
            question,
            DenseConfig(
                top_k=self.params.k_base,
                offline=self.offline,
                st_model=self.emb_st,
                oa_model=self.emb_oa,
            ),
        )
        fused = rrf_fuse(de_hyde, de_base, k=self.params.rrf_k)
        fused = _dedupe_by_chunk_id(fused)[: self.params.top_k_final]
        gr = self.params.guardrails_config or {}
        do_rerank = bool(self.params.rerank) and not bool(gr.get("disable_rerank", False))
        if do_rerank:
            cap = int(gr.get("rerank_top_n_cap", self.params.rerank_top_n))
            old_rtn = self.params.rerank_top_n
            try:
                self.params.rerank_top_n = max(1, min(old_rtn, cap))
                fused, _ = self._apply_rerank(question, fused)
            finally:
                self.params.rerank_top_n = old_rtn
        t_retrieve = time.time() - t0

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
        prompt = build_answer_prompt(
            question, memory=mem_text, docs=fused, format_hint=fmt_hint, persona_hint=per_hint
        )
        it = stream_answer(prompt, self.gen_model, self.offline, fused)
        ctx = {
            "retrieved": fused,
            "prompt": prompt,
            "t_retrieve": t_retrieve,
            "t_hyde_seed": t_hyde_seed,
            "memory_text": mem_text,
            "format_hint": fmt_hint,
            "persona_hint": per_hint,
        }
        return it, ctx
