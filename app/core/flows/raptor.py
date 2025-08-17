from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Iterator, Tuple, Any

from app.core.flows.base import Flow
from app.core.generator import generate_answer
from app.core.prompting.builder import build_answer_prompt
from app.core.retrievers.dense import DenseConfig, retrieve_dense
from app.core.types import AnswerBundle, RetrievedDoc
from app.core.metrics.raglite import lite_metrics, delta_precision_lite
from app.core.retrievers.rerank import rerank_bge_topn, mmr_rerank, llm_rerank
from app.core.embeddings import get_default_embedder
from app.core.metrics.token_cost import estimate_tokens_cost


@dataclass
class RaptorParams:
    # Minimal stub params; full RAPTOR would add tree levels/fanout and internal summarization
    levels: int = 2
    fanout: int = 4
    top_k_final: int = 8
    rerank: bool = False
    rerank_top_n: int = 6
    guardrails_config: Dict | None = None
    rerank_strategy: str = "cross_encoder"  # none|mmr|cross_encoder|llm_judge
    mmr_lambda: float = 0.5
    llm_judge_model: str = "gpt-4o-mini"


class RaptorFlow(Flow):
    """Minimal RAPTOR scaffold.

    For now, performs dense retrieval (optionally reranked) to keep behavior sane,
    while allowing the orchestrator/registry to construct and run a RAPTOR flow.
    """

    def __init__(
        self,
        offline: bool,
        gen_model: str,
        emb_st: str | None,
        emb_oa: str | None,
        params: RaptorParams,
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
        t0 = time.time()
        # For stub: use dense retrieval with a modest top-k informed by top_k_final
        dense_cfg = DenseConfig(
            top_k=max(self.params.top_k_final, 1),
            offline=self.offline,
            st_model=self.emb_st,
            oa_model=self.emb_oa,
        )
        docs, _ = retrieve_dense(question, dense_cfg)
        pre_docs = list(docs)
        # Optional rerank (guardrails-aware)
        gr = self.params.guardrails_config or {}
        do_rerank = bool(self.params.rerank) and not bool(gr.get("disable_rerank", False))
        rr_info: Dict = {}
        if do_rerank:
            docs, rr_info = self._apply_rerank(question, docs)
        t_retrieve = time.time() - t0

        mem_text = ""
        if isinstance(session_state, dict):
            mem_text = session_state.get("memory_text", "")
        fmt_hint = None
        if isinstance(session_state, dict):
            valf = session_state.get("format_hint")
            fmt_hint = str(valf) if isinstance(valf, str) and valf.strip() else None
        per_hint = None
        if isinstance(session_state, dict):
            valp = session_state.get("persona_hint")
            per_hint = str(valp) if isinstance(valp, str) and valp.strip() else None
        prompt = build_answer_prompt(
            question, memory=mem_text, docs=docs, format_hint=fmt_hint, persona_hint=per_hint
        )

        t1 = time.time()
        out = generate_answer(prompt, self.gen_model, self.offline, docs)
        t_generate = time.time() - t1

        timings = {"t_retrieve": t_retrieve, "t_generate": t_generate}
        mets = lite_metrics(out["answer_md"], docs)

        # Metrics enrichment
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
        mets["reranked"] = _mk_list(docs)
        if do_rerank:
            mets["delta_precision_lite"] = delta_precision_lite(pre_docs, docs)
            old_map = {
                (d.chunk.meta.get("path", ""), d.chunk.ord): float(d.score) for d in pre_docs
            }
            new_map = {(d.chunk.meta.get("path", ""), d.chunk.ord): float(d.score) for d in docs}
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
            retrieved=docs,
            extras=extras,
        )

    def run_stream(
        self, question: str, session_state: Dict
    ) -> Tuple[Iterator[str], Dict[str, Any]]:
        """Fallback streaming that reuses run() and streams the final text."""
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
        if (bundle.extras or {}).get("rerank_info"):
            ctx["rerank_info"] = (bundle.extras or {}).get("rerank_info")
        return _iter(), ctx
