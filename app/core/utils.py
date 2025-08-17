from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict

import structlog
from omegaconf import OmegaConf
from app.core.logging_setup import configure_logging

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = APP_ROOT.parent / "data"
DB_PATH = APP_ROOT / "rag.db"
CONFIG_DIR = APP_ROOT / "config"

_logger = structlog.get_logger("rag")


def get_logger():
    # Centralized configuration (console + rotating JSONL file + redaction)
    configure_logging()
    return _logger


def load_config() -> Dict[str, Any]:
    # Load base config
    base_cfg = OmegaConf.load(CONFIG_DIR / "default.yaml")

    # Load environment variables first
    env_cfg = OmegaConf.create(
        {
            "models": {
                "generation": os.getenv("GEN_MODEL", "gpt-5-nano"),
                "embedding": os.getenv("EMB_MODEL", "text-embedding-3-small"),
                "offline": os.getenv("OFFLINE_MODE", "false").lower() == "true",
            }
        }
    )

    # Merge with base config
    merged_cfg = OmegaConf.merge(base_cfg, env_cfg)
    resolved: Dict[str, Any] = OmegaConf.to_container(merged_cfg, resolve=True)  # type: ignore[assignment]
    # Merge persisted settings from DB (if available)
    try:
        # Lazy import to avoid circulars
        from app.core.storage import init_db, get_settings

        init_db()
        settings = get_settings()

        # Map flat settings to nested config keys
        def set_nested(d: Dict[str, Any], path: list[str], value: Any):
            cur = d
            for p in path[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[path[-1]] = value

        mapping = {
            "models.generation": settings.get("models.generation"),
            "models.embedding": settings.get("models.embedding"),
            "models.offline": settings.get("models.offline"),
            "retrieval.top_k": settings.get("retrieval.top_k"),
            "retrieval.rerank": settings.get("retrieval.rerank"),
            "retrieval.rerank_top_n": settings.get("retrieval.rerank_top_n"),
            "retrieval.rerank_strategy": settings.get("retrieval.rerank_strategy"),
            "retrieval.mmr_lambda": settings.get("retrieval.mmr_lambda"),
            "retrieval.rerank_model": settings.get("retrieval.rerank_model"),
            "retrieval.rerank_device": settings.get("retrieval.rerank_device"),
            "retrieval.rerank_batch_size": settings.get("retrieval.rerank_batch_size"),
            "retrieval.llm_judge_model": settings.get("retrieval.llm_judge_model"),
            "retrieval.recency_filter_days": settings.get("retrieval.recency_filter_days"),
            "retrieval.recency_decay_lambda": settings.get("retrieval.recency_decay_lambda"),
            "memory.window_size": settings.get("memory.window_size"),
            "memory.summarize": settings.get("memory.summarize"),
            "flow.default": settings.get("flow.default"),
            "flow.low_cost_mode": settings.get("flow.low_cost_mode"),
            "flow.hyde_mode": settings.get("flow.hyde_mode"),
            "flow.multi_hop_mode": settings.get("flow.multi_hop_mode"),
            "flow.max_cost_estimate": settings.get("flow.max_cost_estimate"),
            "flow.min_query_tokens": settings.get("flow.min_query_tokens"),
            "metrics.ragas_enabled": settings.get("metrics.ragas_enabled"),
            "metrics.ragas_model": settings.get("metrics.ragas_model"),
        }
        for k, v in mapping.items():
            if v is None:
                continue
            set_nested(resolved, k.split("."), v)
    except Exception:
        # If DB not ready, continue with base config
        pass
    return resolved  # type: ignore[return-value]


def ensure_dirs():
    (APP_ROOT / "app").mkdir(parents=True, exist_ok=True)
    (APP_ROOT / "app" / "ui" / "pages").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "samples").mkdir(parents=True, exist_ok=True)


def pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
