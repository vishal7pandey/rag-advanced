from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List

from app.core.flows.base import Flow
from app.core.generator import generate_answer, stream_answer
from app.core.prompting.builder import build_answer_prompt
from app.core.retrievers.bm25 import BM25Config, retrieve_bm25
from app.core.retrievers.dense import DenseConfig, retrieve_dense
from app.core.retrievers.hybrid import rrf_fuse, rrf_fuse_multi
from app.core.types import AnswerBundle, RetrievedDoc
from app.core.metrics.raglite import lite_metrics, delta_precision_lite
from app.core.retrievers.rerank import rerank_bge_topn, mmr_rerank, llm_rerank
from app.core.metrics.token_cost import estimate_tokens_cost
from app.core.embeddings import get_default_embedder

# Optional OpenAI import for multi-query rewrite
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    OpenAI = None  # type: ignore


@dataclass
class HybridParams:
    bm25_k: int = 20
    dense_k: int = 20
    rrf_k: int = 10
    rerank: bool = False
    rerank_top_n: int = 6
    rerank_strategy: str = "mmr"  # none|mmr|cross_encoder|llm_judge
    mmr_lambda: float = 0.5
    llm_judge_model: str = "gpt-4o-mini"
    guardrails_config: Dict | None = None
    # New: weighted fusion and multi-query expansion
    weight_bm25: float = 1.0
    weight_dense: float = 1.0
    multi_query_n: int = 1  # includes the base query, so n<=1 means disabled


class HybridFlow(Flow):
    def __init__(
        self,
        offline: bool,
        gen_model: str,
        emb_st: str | None,
        emb_oa: str | None,
        params: HybridParams,
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
            safe_cost = cost if isinstance(cost, dict) else {}
            info.update({"llm": self.params.llm_judge_model, **safe_cost})
            if safe_cost.get("error") or self.offline:
                head = rerank_bge_topn(question, docs, top_n=self.params.rerank_top_n)
                return head, {"strategy": "cross_encoder", "fallback_from": "llm_judge"}

            def _key(d: RetrievedDoc):
                return getattr(d.chunk, "id", None) or (d.chunk.meta.get("path", ""), d.chunk.ord)

            sel_keys = {_key(d) for d in rr}
            tail = [d for d in docs if _key(d) not in sel_keys]
            return rr + tail, info
        head = rerank_bge_topn(question, docs, top_n=self.params.rerank_top_n)
        return head, info

    def _dedupe_by_chunk_id(self, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        seen = set()
        out: List[RetrievedDoc] = []
        for d in docs:
            cid = getattr(d.chunk, "id", None) or f"{d.chunk.meta.get('path', '')}::{d.chunk.ord}"
            if cid in seen:
                continue
            seen.add(cid)
            out.append(d)
        return out

    def _generate_rewrites(self, question: str, n: int) -> List[str]:
        """Generate up to n rewritten variants of the question using OpenAI when available.
        Falls back to simple heuristic rewrites if offline/unavailable.
        """
        n = max(0, int(n))
        if n == 0:
            return []
        # Respect generator model for economy; use same model
        model = self.gen_model
        # Cheap fallback if offline or OpenAI not configured
        if self.offline or (OpenAI is None):
            base = question.strip()
            # lightweight heuristic variants
            variants = [
                base,
                f"{base} (key details)",
                f"{base} (step-by-step)",
                f"{base} (definition)",
                f"{base} (troubleshooting)",
            ]
            # Return up to n unique variants excluding the original which caller already includes
            uniq: List[str] = []
            for v in variants[1:]:
                if len(uniq) >= n:
                    break
                uniq.append(v)
            return uniq
        try:
            client = OpenAI()
            sys = (
                "You rewrite a user question into concise alternative phrasings that might retrieve different evidence. "
                "Return each rewrite on its own line; do not number or explain. Keep them short."
            )
            prompt = (
                f"Original: {question}\n"
                f"Write {n} alternative phrasings that broaden recall (synonyms, explicit entities, common variants).\n"
                "Only output the rewrites, one per line."
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
                temperature=0.3,
            )
            text = (resp.choices[0].message.content or "").strip()
            lines = [ln.strip(" -\t") for ln in text.splitlines() if ln.strip()]
            return lines[:n]
        except Exception:
            return []

    def run(self, question: str, session_state: Dict) -> AnswerBundle:
        t0 = time.perf_counter()
        # Multi-query expansion: base + rewrites
        queries: List[str] = [question]
        extra_n = max(0, int(self.params.multi_query_n) - 1)
        if extra_n > 0:
            rewrites = self._generate_rewrites(question, extra_n)
            # ensure uniqueness
            for r in rewrites:
                if r and r not in queries:
                    queries.append(r)

        bm_lists: List[List[RetrievedDoc]] = []
        de_lists: List[List[RetrievedDoc]] = []
        for q in queries:
            bm_lists.append(retrieve_bm25(q, BM25Config(top_k=self.params.bm25_k)))
            de_q, _ = retrieve_dense(
                q,
                DenseConfig(
                    top_k=self.params.dense_k,
                    offline=self.offline,
                    st_model=self.emb_st,
                    oa_model=self.emb_oa,
                ),
            )
            de_lists.append(de_q)

        if len(queries) > 1:
            fused = rrf_fuse_multi(
                bm_lists,
                de_lists,
                k=self.params.rrf_k,
                w_bm25=self.params.weight_bm25,
                w_dense=self.params.weight_dense,
            )
        else:
            fused = rrf_fuse(
                bm_lists[0],
                de_lists[0],
                k=self.params.rrf_k,
                w_bm25=self.params.weight_bm25,
                w_dense=self.params.weight_dense,
            )
        fused = self._dedupe_by_chunk_id(fused)[: max(self.params.bm25_k, self.params.dense_k)]
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
        t_retrieve = time.perf_counter() - t0
        if t_retrieve <= 0:
            t_retrieve = 1e-6
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
        t1 = time.perf_counter()
        out = generate_answer(prompt, self.gen_model, self.offline, fused)
        t_generate = time.perf_counter() - t1
        if t_generate <= 0:
            t_generate = 1e-6
        timings = {"t_retrieve": t_retrieve, "t_generate": t_generate}
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
        t0 = time.perf_counter()
        queries: List[str] = [question]
        extra_n = max(0, int(self.params.multi_query_n) - 1)
        if extra_n > 0:
            rewrites = self._generate_rewrites(question, extra_n)
            for r in rewrites:
                if r and r not in queries:
                    queries.append(r)
        bm_lists: List[List[RetrievedDoc]] = []
        de_lists: List[List[RetrievedDoc]] = []
        for q in queries:
            bm_lists.append(retrieve_bm25(q, BM25Config(top_k=self.params.bm25_k)))
            de_q, _ = retrieve_dense(
                q,
                DenseConfig(
                    top_k=self.params.dense_k,
                    offline=self.offline,
                    st_model=self.emb_st,
                    oa_model=self.emb_oa,
                ),
            )
            de_lists.append(de_q)
        if len(queries) > 1:
            fused = rrf_fuse_multi(
                bm_lists,
                de_lists,
                k=self.params.rrf_k,
                w_bm25=self.params.weight_bm25,
                w_dense=self.params.weight_dense,
            )
        else:
            fused = rrf_fuse(
                bm_lists[0],
                de_lists[0],
                k=self.params.rrf_k,
                w_bm25=self.params.weight_bm25,
                w_dense=self.params.weight_dense,
            )
        fused = self._dedupe_by_chunk_id(fused)[: max(self.params.bm25_k, self.params.dense_k)]
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
        t_retrieve = time.perf_counter() - t0
        if t_retrieve <= 0:
            t_retrieve = 1e-6
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
            "memory_text": mem_text,
            "format_hint": fmt_hint,
            "persona_hint": per_hint,
        }
        return it, ctx
