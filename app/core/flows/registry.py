from __future__ import annotations
from typing import Dict, Any, Literal, TypeVar, Union

from app.core.flows.standard import StandardFlow, StandardParams
from app.core.flows.hybrid import HybridFlow, HybridParams
from app.core.flows.hyde import HyDEFlow, HyDEParams
from app.core.flows.multi_hop import MultiHopFlow, MultiHopParams
from app.core.flows.raptor import RaptorFlow, RaptorParams

# Type variable for flow parameters
ParamsT = TypeVar("ParamsT", StandardParams, HybridParams, HyDEParams, MultiHopParams, RaptorParams)

# Type for the configuration dictionary
FlowConfig = Dict[str, Any]


def make_flow(
    name: Literal["standard", "hybrid", "hyde", "multi_hop", "raptor"],
    offline: bool,
    gen_model: str,
    emb_st: str | None,
    emb_oa: str | None,
    cfg: FlowConfig,
) -> Union[StandardFlow, HybridFlow, HyDEFlow, MultiHopFlow, RaptorFlow]:
    if name == "standard":
        params: StandardParams = StandardParams(
            top_k=cfg.get("top_k", 6),
            rerank=cfg.get("rerank", False),
            rerank_top_n=cfg.get("rerank_top_n", 6),
            rerank_strategy=cfg.get("rerank_strategy", "cross_encoder"),
            mmr_lambda=cfg.get("mmr_lambda", 0.5),
            llm_judge_model=cfg.get("llm_judge_model", "gpt--mini"),
            guardrails_config=cfg.get("guardrails_config"),
        )
        return StandardFlow(offline, gen_model, emb_st, emb_oa, params)
    if name == "hybrid":
        # support both flat weights and nested cfg.weights.{bm25,dense}
        weights = cfg.get("weights", {}) or {}
        w_bm25 = cfg.get("weight_bm25", weights.get("bm25", 1.0))
        w_dense = cfg.get("weight_dense", weights.get("dense", 1.0))
        hybrid_params: HybridParams = HybridParams(
            bm25_k=cfg.get("bm25_k", 20),
            dense_k=cfg.get("dense_k", 20),
            rrf_k=cfg.get("rrf_k", 10),
            rerank=cfg.get("rerank", False),
            rerank_top_n=cfg.get("rerank_top_n", 6),
            rerank_strategy=cfg.get("rerank_strategy", "cross_encoder"),
            mmr_lambda=cfg.get("mmr_lambda", 0.5),
            llm_judge_model=cfg.get("llm_judge_model", "gpt--mini"),
            guardrails_config=cfg.get("guardrails_config"),
            weight_bm25=float(w_bm25),
            weight_dense=float(w_dense),
            multi_query_n=int(cfg.get("multi_query_n", 1)),
        )
        return HybridFlow(offline, gen_model, emb_st, emb_oa, hybrid_params)
    if name == "hyde":
        hyde_params: HyDEParams = HyDEParams(
            k_base=cfg.get("k_base", 12),
            k_hyde=cfg.get("k_hyde", 12),
            top_k_final=cfg.get("top_k_final", 10),
            rrf_k=cfg.get("rrf_k", 60),
            rerank=cfg.get("rerank", False),
            rerank_top_n=cfg.get("rerank_top_n", 6),
            rerank_strategy=cfg.get("rerank_strategy", "cross_encoder"),
            mmr_lambda=cfg.get("mmr_lambda", 0.5),
            llm_judge_model=cfg.get("llm_judge_model", "gpt--mini"),
            guardrails_config=cfg.get("guardrails_config"),
        )
        return HyDEFlow(offline, gen_model, emb_st, emb_oa, hyde_params)
    if name == "multi_hop":
        multi_hop_params: MultiHopParams = MultiHopParams(
            subq_max=cfg.get("subq_max", 3),
            k_each=cfg.get("k_each", 5),
            top_k_final=cfg.get("top_k_final", 8),
            rrf_k=cfg.get("rrf_k", 60),
            rerank=cfg.get("rerank", False),
            rerank_top_n=cfg.get("rerank_top_n", 6),
            rerank_strategy=cfg.get("rerank_strategy", "cross_encoder"),
            mmr_lambda=cfg.get("mmr_lambda", 0.5),
            llm_judge_model=cfg.get("llm_judge_model", "gpt--mini"),
            guardrails_config=cfg.get("guardrails_config"),
        )
        return MultiHopFlow(offline, gen_model, emb_st, emb_oa, multi_hop_params)
    if name == "raptor":
        rparams: RaptorParams = RaptorParams(
            levels=cfg.get("levels", 2),
            fanout=cfg.get("fanout", 4),
            top_k_final=cfg.get("top_k_final", 8),
            rerank=cfg.get("rerank", False),
            rerank_top_n=cfg.get("rerank_top_n", 6),
            rerank_strategy=cfg.get("rerank_strategy", "cross_encoder"),
            mmr_lambda=cfg.get("mmr_lambda", 0.5),
            llm_judge_model=cfg.get("llm_judge_model", "gpt--mini"),
            guardrails_config=cfg.get("guardrails_config"),
        )
        return RaptorFlow(offline, gen_model, emb_st, emb_oa, rparams)
    # Default fallback
    default_params = StandardParams()
    return StandardFlow(offline, gen_model, emb_st, emb_oa, default_params)
