from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Literal, Iterator, cast
import re
import random

from app.core.utils import load_config
from app.core.types import StreamCtx, RetrievedDoc
from app.core.flows.registry import make_flow


FlowName = Literal["standard", "hybrid", "hyde", "multi_hop", "raptor"]


@dataclass
class RetrievalPlan:
    mode: str  # 'standard'|'hybrid'|'hyde'|'multi_hop'
    top_k: int
    rerank: bool
    rerank_top_n: int
    bm25_k: int | None = None
    dense_k: int | None = None
    rrf_k: int | None = None
    subq_max: int | None = None
    notes: List[str] = field(default_factory=list)


@dataclass
class GenerationPlan:
    model: str
    temperature: float = 0.2
    max_tokens: int | None = None
    format_hint: str | None = None
    persona_hint: str | None = None


@dataclass
class RunPlan:
    flow_name: FlowName  # 'standard'|'hybrid'|'hyde'|'multi_hop'|'raptor'
    retrieval: RetrievalPlan
    generation: GenerationPlan
    notes: List[str] = field(default_factory=list)


class Orchestrator:
    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        self.cfg = cfg or load_config()

    # ---- Heuristics -----------------------------------------------------
    def _classify(self, q: str) -> Dict[str, bool]:
        ql = q.lower()
        toks = [t for t in re.split(r"\s+", ql) if t]
        is_short_or_vague = len(toks) < 4
        is_multihop = bool(
            re.search(
                r"\b(then|after|before|first|second|third|combine|merge|compare|contrast|steps|and)\b",
                ql,
            )
        )
        return {
            "is_short_or_vague": is_short_or_vague,
            "is_multihop": is_multihop,
        }

    def _format_hint(self, q: str) -> str | None:
        """Lightweight heuristics to guide answer formatting."""
        ql = q.lower()
        # How-to / steps
        if re.search(r"\b(how to|how do i|steps|procedure|setup|install|configure|create)\b", ql):
            return (
                "Use concise numbered steps (1., 2., 3.). Each step <= 2 sentences. "
                "End with a short 'Next actions' bullet list."
            )
        # Comparison
        if re.search(
            r"\b(compare|comparison|difference|vs\.?|pros and cons|advantages|disadvantages)\b", ql
        ):
            return (
                "Provide a brief intro, then a 2-column Markdown table of key aspects. "
                "Follow with bullet-point pros and cons."
            )
        # Summary / overview
        if re.search(r"\b(summary|summarize|overview|highlights|key points)\b", ql):
            return "Provide 3–5 bullet points. Bold the lead-in term for each point."
        # Definition
        if re.search(r"\b(what is|define|definition)\b", ql):
            return "Start with a single concise paragraph definition, then 3 bullets with salient facts."
        # Code/sample
        if re.search(r"\b(example|sample|snippet|regex|command|sql|code)\b", ql):
            return "Show a minimal code block first, then a short explanation and 2–3 gotchas as bullets."
        # Lists
        if re.search(r"\b(list|top|best|checklist)\b", ql):
            return "Return a numbered list with at most 5 items, each one sentence."
        # Timeline
        if re.search(r"\b(timeline|chronology|history|roadmap)\b", ql):
            return "Provide a chronological bullet list with dates or phases in bold."
        return None

    def _persona_hint(self, q: str) -> str:
        """Random/automated persona selection with light heuristics.

        Personas:
        - Concise Expert
        - Supportive Mentor
        - Analytical Researcher
        - Playful Creator
        """
        ql = q.lower()
        # Base candidates
        personas = [
            (
                "Concise Expert",
                "Adopt a concise expert tone: clear, objective, minimal adjectives; avoid hedging; prioritize accuracy and brevity.",
            ),
            (
                "Supportive Mentor",
                "Adopt a supportive mentor tone: empathetic, encouraging, step-by-step guidance; avoid sarcasm; acknowledge the user's intent.",
            ),
            (
                "Analytical Researcher",
                "Adopt an analytical researcher tone: compare alternatives when relevant, cite evidence from the provided context, and highlight trade-offs with neutral language.",
            ),
            (
                "Playful Creator",
                "Adopt a playful creative tone: light humor and vivid analogies while keeping facts correct; avoid overusing emojis; be brief and engaging.",
            ),
        ]
        # Heuristic weighting
        weights = [1.0, 1.0, 1.0, 1.0]
        if re.search(r"\b(how to|steps|troubleshoot|fix|install|configure|setup)\b", ql):
            # Guidance oriented
            weights = [0.8, 1.8, 1.0, 0.9]
        elif re.search(r"\b(compare|vs\.?|pros and cons|trade[- ]?offs|difference)\b", ql):
            # Analytical
            weights = [1.0, 0.8, 1.8, 0.9]
        elif re.search(r"\b(idea|brainstorm|creative|story|analogy)\b", ql):
            # Creative
            weights = [0.9, 1.0, 0.9, 1.8]
        else:
            # Default balances to concise expert
            weights = [1.6, 1.0, 1.0, 0.9]

        # Randomly choose one with weights
        idx = random.choices(range(len(personas)), weights=weights, k=1)[0]
        name, desc = personas[idx]
        # Return a compact persona hint for templates
        return f"{name}: {desc}"

    def build_plan(self, question: str) -> RunPlan:
        rcfg = self.cfg.get("retrieval", {})
        mcfg = self.cfg.get("models", {})
        fcfg = self.cfg.get("flow", {})
        top_k = int(rcfg.get("top_k", 6))
        rerank = bool(rcfg.get("rerank", False))
        rerank_top_n = int(rcfg.get("rerank_top_n", 6))
        gen_model = str(mcfg.get("generation", "gpt-4o-mini"))

        triage = self._classify(question)
        notes: List[str] = []
        # Interpret Admin Settings for flow control
        hyde_mode = str(fcfg.get("hyde_mode", "auto")).lower()
        multi_mode = str(fcfg.get("multi_hop_mode", "auto")).lower()
        hyde_off = (hyde_mode == "off") or (fcfg.get("hyde_mode") is False)
        multi_off = (multi_mode == "off") or (fcfg.get("multi_hop_mode") is False)
        low_cost = bool(fcfg.get("low_cost_mode", False))

        # Choose flow
        flow_name: FlowName
        if low_cost:
            # Force economical default under low-cost mode
            flow_name = "hybrid"
            notes.append("hybrid: low_cost_mode")
            rplan = RetrievalPlan(
                mode="hybrid",
                top_k=top_k,
                rerank=rerank,
                rerank_top_n=rerank_top_n,
                bm25_k=top_k,
                dense_k=top_k,
                rrf_k=max(10, top_k),
                notes=["rrf fuse bm25+dense"],
            )
        elif triage["is_multihop"] and not multi_off:
            flow_name = "multi_hop"
            notes.append("multi_hop: connectors detected")
            rplan = RetrievalPlan(
                mode="multi_hop",
                top_k=top_k,
                rerank=rerank,
                rerank_top_n=min(10, top_k),
                subq_max=3,
                rrf_k=max(30, top_k * 5),
                notes=["rrf_k scaled for hops"],
            )
        elif triage["is_short_or_vague"] and not hyde_off:
            flow_name = "hyde"
            notes.append("hyde: short or vague query")
            rplan = RetrievalPlan(
                mode="hyde",
                top_k=top_k,
                rerank=rerank,
                rerank_top_n=min(10, top_k),
                rrf_k=max(30, top_k * 5),
                notes=["hyde seed + fusion"],
            )
        else:
            flow_name = "hybrid"
            notes.append("hybrid: default")
            rplan = RetrievalPlan(
                mode="hybrid",
                top_k=top_k,
                rerank=rerank,
                rerank_top_n=rerank_top_n,
                bm25_k=top_k,
                dense_k=top_k,
                rrf_k=max(10, top_k),
                notes=["rrf fuse bm25+dense"],
            )

        gplan = GenerationPlan(
            model=gen_model,
            temperature=0.2,
            format_hint=self._format_hint(question),
            persona_hint=self._persona_hint(question),
        )
        return RunPlan(flow_name=flow_name, retrieval=rplan, generation=gplan, notes=notes)

    # ---- Execute --------------------------------------------------------
    def plan_and_run(self, question: str, session_state: Dict[str, Any]) -> Tuple[Any, RunPlan]:
        plan = self.build_plan(question)
        mcfg = self.cfg.get("models", {})
        offline = str(mcfg.get("offline", "false")).lower() == "true"
        gen_model = str(mcfg.get("generation", "gpt-4o-mini"))
        emb_st = None
        emb_oa = str(mcfg.get("embedding", "text-embedding-3-small"))
        # Retrieve dynamic rerank and guardrails settings
        rcfg = self.cfg.get("retrieval", {})
        fcfg = self.cfg.get("flow", {})
        rerank_strategy = str(rcfg.get("rerank_strategy", "mmr"))
        try:
            mmr_lambda = float(rcfg.get("mmr_lambda", 0.5))
        except Exception:
            mmr_lambda = 0.5
        llm_judge_model = str(rcfg.get("llm_judge_model", gen_model))
        # Build guardrails config (supports low-cost mode)
        guardrails_cfg: Dict[str, Any] = {}
        try:
            if bool(fcfg.get("low_cost_mode", False)):
                # Lazy import to avoid any potential cycles
                from app.core.flows.guardrails import get_low_cost_mode_config  # type: ignore

                guardrails_cfg = get_low_cost_mode_config()
            else:
                if "min_query_tokens" in fcfg:
                    try:
                        guardrails_cfg["min_query_tokens"] = int(fcfg.get("min_query_tokens"))
                    except Exception:
                        pass
                if "max_cost_estimate" in fcfg:
                    try:
                        guardrails_cfg["max_cost_estimate"] = float(fcfg.get("max_cost_estimate"))
                    except Exception:
                        pass
                # Allow disabling specific advanced flows
                hm = str(fcfg.get("hyde_mode", "auto")).lower()
                if (hm == "off") or (fcfg.get("hyde_mode") is False):
                    guardrails_cfg["disable_hyde"] = True
                mm = str(fcfg.get("multi_hop_mode", "auto")).lower()
                if (mm == "off") or (fcfg.get("multi_hop_mode") is False):
                    guardrails_cfg["disable_multi_hop"] = True
        except Exception:
            # On any error, fall back to empty guardrails config
            guardrails_cfg = {}

        # Convert plan to flow params
        p = plan.retrieval
        if plan.flow_name == "standard":
            params = {
                "top_k": p.top_k,
                "rerank": p.rerank,
                "rerank_top_n": p.rerank_top_n,
                "rerank_strategy": rerank_strategy,
                "mmr_lambda": mmr_lambda,
                "llm_judge_model": llm_judge_model,
                "guardrails_config": guardrails_cfg,
            }
        elif plan.flow_name == "hybrid":
            params = {
                "bm25_k": p.bm25_k or p.top_k,
                "dense_k": p.dense_k or p.top_k,
                "rrf_k": p.rrf_k or max(10, p.top_k),
                "rerank": p.rerank,
                "rerank_top_n": p.rerank_top_n,
                "rerank_strategy": rerank_strategy,
                "mmr_lambda": mmr_lambda,
                "llm_judge_model": llm_judge_model,
                "guardrails_config": guardrails_cfg,
                # New: weights and multi-query expansion
                "weight_bm25": float(rcfg.get("rrf_weight_bm25", 1.0)),
                "weight_dense": float(rcfg.get("rrf_weight_dense", 1.0)),
                "multi_query_n": int(rcfg.get("multi_query_n", 1)),
            }
        elif plan.flow_name == "hyde":
            params = {
                "k_base": p.top_k,
                "k_hyde": p.top_k,
                "top_k_final": p.top_k,
                "rrf_k": p.rrf_k or max(30, p.top_k * 5),
                "rerank": p.rerank,
                "rerank_top_n": min(10, p.top_k),
                "rerank_strategy": rerank_strategy,
                "mmr_lambda": mmr_lambda,
                "llm_judge_model": llm_judge_model,
                "guardrails_config": guardrails_cfg,
            }
        elif plan.flow_name == "multi_hop":
            params = {
                "subq_max": p.subq_max or 3,
                "k_each": max(3, min(10, p.top_k)),
                "top_k_final": p.top_k,
                "rrf_k": p.rrf_k or max(30, p.top_k * 5),
                "rerank": p.rerank,
                "rerank_top_n": min(10, p.top_k),
                "rerank_strategy": rerank_strategy,
                "mmr_lambda": mmr_lambda,
                "llm_judge_model": llm_judge_model,
                "guardrails_config": guardrails_cfg,
            }
        else:
            params = {
                "top_k": p.top_k,
                "rerank": p.rerank,
                "rerank_top_n": p.rerank_top_n,
                "rerank_strategy": rerank_strategy,
                "mmr_lambda": mmr_lambda,
                "llm_judge_model": llm_judge_model,
                "guardrails_config": guardrails_cfg,
            }

        engine = make_flow(plan.flow_name, offline, gen_model, emb_st, emb_oa, params)
        # Propagate optional hints via session_state
        ss = dict(session_state or {})
        if plan.generation.format_hint:
            ss["format_hint"] = plan.generation.format_hint
        if plan.generation.persona_hint:
            ss["persona_hint"] = plan.generation.persona_hint
        bundle = engine.run(question, session_state=ss)
        try:
            # Attach plan to extras for transparency
            if hasattr(bundle, "extras") and isinstance(bundle.extras, dict):
                bundle.extras.update(
                    {
                        "plan": asdict(plan),
                        "flow": plan.flow_name,
                        "format_hint": plan.generation.format_hint,
                        "persona_hint": plan.generation.persona_hint,
                    }
                )
        except Exception:
            pass
        return bundle, plan

    def plan_and_stream(
        self, question: str, session_state: Dict[str, Any]
    ) -> Tuple[Iterator[str], RunPlan, StreamCtx]:
        """Plan and return a streaming iterator plus plan and minimal ctx.
        Falls back to non-streaming run if the flow does not support streaming.
        """
        plan = self.build_plan(question)
        mcfg = self.cfg.get("models", {})
        offline = str(mcfg.get("offline", "false")).lower() == "true"
        gen_model = str(mcfg.get("generation", "gpt-4o-mini"))
        emb_st = None
        emb_oa = str(mcfg.get("embedding", "text-embedding-3-small"))
        # Retrieve dynamic rerank and guardrails settings
        rcfg = self.cfg.get("retrieval", {})
        fcfg = self.cfg.get("flow", {})
        rerank_strategy = str(rcfg.get("rerank_strategy", "mmr"))
        try:
            mmr_lambda = float(rcfg.get("mmr_lambda", 0.5))
        except Exception:
            mmr_lambda = 0.5
        llm_judge_model = str(rcfg.get("llm_judge_model", gen_model))
        # Build guardrails config (supports low-cost mode)
        guardrails_cfg: Dict[str, Any] = {}
        try:
            if bool(fcfg.get("low_cost_mode", False)):
                from app.core.flows.guardrails import get_low_cost_mode_config  # type: ignore

                guardrails_cfg = get_low_cost_mode_config()
            else:
                if "min_query_tokens" in fcfg:
                    try:
                        guardrails_cfg["min_query_tokens"] = int(fcfg.get("min_query_tokens"))
                    except Exception:
                        pass
                if "max_cost_estimate" in fcfg:
                    try:
                        guardrails_cfg["max_cost_estimate"] = float(fcfg.get("max_cost_estimate"))
                    except Exception:
                        pass
                hm = str(fcfg.get("hyde_mode", "auto")).lower()
                if (hm == "off") or (fcfg.get("hyde_mode") is False):
                    guardrails_cfg["disable_hyde"] = True
                mm = str(fcfg.get("multi_hop_mode", "auto")).lower()
                if (mm == "off") or (fcfg.get("multi_hop_mode") is False):
                    guardrails_cfg["disable_multi_hop"] = True
        except Exception:
            guardrails_cfg = {}

        # Convert plan to flow params (same as plan_and_run)
        p = plan.retrieval
        if plan.flow_name == "standard":
            params = {
                "top_k": p.top_k,
                "rerank": p.rerank,
                "rerank_top_n": p.rerank_top_n,
                "rerank_strategy": rerank_strategy,
                "mmr_lambda": mmr_lambda,
                "llm_judge_model": llm_judge_model,
                "guardrails_config": guardrails_cfg,
            }
        elif plan.flow_name == "hybrid":
            params = {
                "bm25_k": p.bm25_k or p.top_k,
                "dense_k": p.dense_k or p.top_k,
                "rrf_k": p.rrf_k or max(10, p.top_k),
                "rerank": p.rerank,
                "rerank_top_n": p.rerank_top_n,
                "rerank_strategy": rerank_strategy,
                "mmr_lambda": mmr_lambda,
                "llm_judge_model": llm_judge_model,
                "guardrails_config": guardrails_cfg,
                # New: weights and multi-query expansion
                "weight_bm25": float(rcfg.get("rrf_weight_bm25", 1.0)),
                "weight_dense": float(rcfg.get("rrf_weight_dense", 1.0)),
                "multi_query_n": int(rcfg.get("multi_query_n", 1)),
            }
        elif plan.flow_name == "hyde":
            params = {
                "k_base": p.top_k,
                "k_hyde": p.top_k,
                "top_k_final": p.top_k,
                "rrf_k": p.rrf_k or max(30, p.top_k * 5),
                "rerank": p.rerank,
                "rerank_top_n": min(10, p.top_k),
                "rerank_strategy": rerank_strategy,
                "mmr_lambda": mmr_lambda,
                "llm_judge_model": llm_judge_model,
                "guardrails_config": guardrails_cfg,
            }
        elif plan.flow_name == "multi_hop":
            params = {
                "subq_max": p.subq_max or 3,
                "k_each": max(3, min(10, p.top_k)),
                "top_k_final": p.top_k,
                "rrf_k": p.rrf_k or max(30, p.top_k * 5),
                "rerank": p.rerank,
                "rerank_top_n": min(10, p.top_k),
                "rerank_strategy": rerank_strategy,
                "mmr_lambda": mmr_lambda,
                "llm_judge_model": llm_judge_model,
                "guardrails_config": guardrails_cfg,
            }
        else:
            params = {
                "top_k": p.top_k,
                "rerank": p.rerank,
                "rerank_top_n": p.rerank_top_n,
                "rerank_strategy": rerank_strategy,
                "mmr_lambda": mmr_lambda,
                "llm_judge_model": llm_judge_model,
                "guardrails_config": guardrails_cfg,
            }

        engine = make_flow(plan.flow_name, offline, gen_model, emb_st, emb_oa, params)
        ss = dict(session_state or {})
        if plan.generation.format_hint:
            ss["format_hint"] = plan.generation.format_hint
        if plan.generation.persona_hint:
            ss["persona_hint"] = plan.generation.persona_hint
        # Try streaming
        it: Iterator[str] | None = None
        ctx: StreamCtx = {}
        try:
            if hasattr(engine, "run_stream"):
                it_obj, ctx_obj = engine.run_stream(question, session_state=ss)  # type: ignore[attr-defined]
                it = cast(Iterator[str], it_obj)
                ctx = cast(StreamCtx, ctx_obj or {})
        except Exception:
            it = None
        if it is None:
            # Fallback: run() then stream the final text
            bundle = engine.run(question, session_state=ss)

            def _iter():
                for tok in (bundle.answer_md or "").split():
                    yield tok + " "

            it = _iter()
            ctx = {
                "retrieved": cast(List[RetrievedDoc], bundle.retrieved),
                "prompt": str((bundle.extras or {}).get("prompt", {}).get("text", "")),
                "t_retrieve": cast(float | None, (bundle.timings or {}).get("t_retrieve")),
                "t_generate": cast(float | None, (bundle.timings or {}).get("t_generate")),
            }
            # Surface multi-hop hops for UI if present
            try:
                hops = (bundle.extras or {}).get("hops")
                if hops:
                    ctx["hops"] = hops
            except Exception:
                pass
        return it, plan, ctx
