from __future__ import annotations
import streamlit as st
from dotenv import load_dotenv

from app.core.utils import load_config, pretty_json
from app.core.storage import init_db, get_settings, set_settings

load_dotenv()

st.title("⚙️ Admin Settings")

cfg = load_config()
init_db()
saved = get_settings()

st.subheader("Models")
model = st.text_input(
    "Generation model", saved.get("models.generation", cfg["models"]["generation"])
)
emb = st.text_input("Embedding model", saved.get("models.embedding", cfg["models"]["embedding"]))
offline = st.toggle("Offline mode", bool(saved.get("models.offline", cfg["models"]["offline"])))

st.subheader("Generator")
gen_provider = st.selectbox(
    "Provider",
    ["openai", "ollama", "stub"],
    index=["openai", "ollama", "stub"].index(
        str(saved.get("generator.provider", cfg.get("generator", {}).get("provider", "openai")))
    ),
)
gen_model = st.text_input(
    "Default model (OpenAI)",
    str(saved.get("generator.model", cfg.get("generator", {}).get("model", "gpt-4o-mini"))),
)
gen_ollama_model = st.text_input(
    "Ollama model",
    str(
        saved.get(
            "generator.ollama_model", cfg.get("generator", {}).get("ollama_model", "llama3.1:8b")
        )
    ),
)
gen_timeout = st.number_input(
    "Timeout (s)",
    min_value=5,
    max_value=600,
    value=int(saved.get("generator.timeout_s", cfg.get("generator", {}).get("timeout_s", 30))),
    step=5,
)

st.subheader("Retrieval Defaults")
top_k = st.slider("Top-k", 1, 50, int(saved.get("retrieval.top_k", cfg["retrieval"]["top_k"])))

# Reranking Strategy
rerank_strategy = st.selectbox(
    "Rerank strategy",
    ["none", "mmr", "cross_encoder", "llm_judge"],
    index=["none", "mmr", "cross_encoder", "llm_judge"].index(
        str(
            saved.get(
                "retrieval.rerank_strategy", cfg.get("retrieval", {}).get("rerank_strategy", "mmr")
            )
        )
    ),
    help="None: no reranking | MMR: diversity via embeddings (API-only) | CrossEncoder: requires [offline] extra | LLM-judge: OpenAI scoring (costs tokens)",
)
rerank_top_n = st.slider("Rerank top N", 1, 20, int(saved.get("retrieval.rerank_top_n", 6)))

# MMR settings (shown when MMR selected)
if rerank_strategy == "mmr":
    mmr_lambda = st.slider(
        "MMR λ (relevance vs diversity)",
        0.0,
        1.0,
        float(saved.get("retrieval.mmr_lambda", 0.5)),
        step=0.1,
        help="1.0 = pure relevance, 0.0 = pure diversity",
    )
else:
    mmr_lambda = 0.5

# CrossEncoder settings (shown when cross_encoder selected)
if rerank_strategy == "cross_encoder":
    rr_model = st.text_input(
        "CrossEncoder model",
        str(
            saved.get(
                "retrieval.rerank_model",
                cfg["retrieval"].get("rerank_model", "BAAI/bge-reranker-v2-m3"),
            )
        ),
    )
    rr_device = st.selectbox(
        "Device",
        ["cpu", "cuda"],
        index=["cpu", "cuda"].index(
            str(saved.get("retrieval.rerank_device", cfg["retrieval"].get("rerank_device", "cpu")))
        ),
    )
    rr_bs = st.number_input(
        "Batch size",
        min_value=1,
        max_value=128,
        value=int(
            saved.get("retrieval.rerank_batch_size", cfg["retrieval"].get("rerank_batch_size", 16))
        ),
        step=1,
    )
else:
    rr_model = cfg["retrieval"].get("rerank_model", "BAAI/bge-reranker-v2-m3")
    rr_device = "cpu"
    rr_bs = 16

# LLM-judge settings (shown when llm_judge selected)
if rerank_strategy == "llm_judge":
    llm_judge_model = st.text_input(
        "LLM-judge model",
        str(saved.get("retrieval.llm_judge_model", "gpt-4o-mini")),
        help="OpenAI model for relevance scoring",
    )
else:
    llm_judge_model = "gpt-4o-mini"

# Legacy rerank toggle for backward compatibility
rerank = rerank_strategy != "none"
recency_days = st.number_input(
    "Recency filter (days; 0=off)",
    min_value=0,
    max_value=3650,
    value=int(
        saved.get("retrieval.recency_filter_days", cfg["retrieval"].get("recency_filter_days", 0))
    ),
    step=1,
)
recency_lambda = st.number_input(
    "Recency decay λ (0.0=off)",
    min_value=0.0,
    max_value=1.0,
    value=float(
        saved.get(
            "retrieval.recency_decay_lambda", cfg["retrieval"].get("recency_decay_lambda", 0.0)
        )
    ),
    step=0.01,
)

st.subheader("Memory Defaults")
window = st.slider(
    "Window size", 1, 20, int(saved.get("memory.window_size", cfg["memory"]["window_size"]))
)
summarize = st.toggle(
    "Enable summarization", bool(saved.get("memory.summarize", cfg["memory"]["summarize"]))
)

st.subheader("Flow Defaults")
flow_default = st.selectbox(
    "Default flow",
    ["standard", "hybrid", "hyde", "multi_hop"],
    index=["standard", "hybrid", "hyde", "multi_hop"].index(saved.get("flow.default", "standard")),
)

# Low-cost mode and guardrails
low_cost_mode = st.toggle(
    "Low-cost mode",
    bool(saved.get("flow.low_cost_mode", False)),
    help="Disables HyDE/Multi-hop and LLM-judge reranking to minimize API costs",
)

st.write("**Advanced Flow Guardrails**")
hyde_auto = st.selectbox(
    "HyDE mode",
    ["auto", "on", "off"],
    index=["auto", "on", "off"].index(str(saved.get("flow.hyde_mode", "auto"))),
    help="Auto: skip for simple queries | On: always use | Off: never use",
)
multi_hop_auto = st.selectbox(
    "Multi-hop mode",
    ["auto", "on", "off"],
    index=["auto", "on", "off"].index(str(saved.get("flow.multi_hop_mode", "auto"))),
    help="Auto: skip for simple queries | On: always use | Off: never use",
)

if not low_cost_mode:
    max_cost = st.number_input(
        "Max cost per query ($)",
        min_value=0.01,
        max_value=1.0,
        value=float(saved.get("flow.max_cost_estimate", 0.10)),
        step=0.01,
        format="%.3f",
    )
    min_tokens = st.number_input(
        "Min query tokens for advanced flows",
        min_value=1,
        max_value=10,
        value=int(saved.get("flow.min_query_tokens", 3)),
        step=1,
    )
else:
    max_cost = 0.05
    min_tokens = 4

st.subheader("Indexing")
semantic_chunking = st.toggle(
    "Enable semantic/layout chunking (unstructured)",
    bool(
        saved.get(
            "index.semantic_chunking_enabled", cfg["index"].get("semantic_chunking_enabled", False)
        )
    ),
)

st.subheader("Metrics (RAGAS)")
ragas_enabled = st.toggle("Enable RAGAS metrics", bool(saved.get("metrics.ragas_enabled", False)))
ragas_model = st.text_input("RAGAS model (OpenAI)", saved.get("metrics.ragas_model", "gpt-4o-mini"))

st.subheader("Flow Config Snapshot")
st.code(pretty_json(cfg), language="json")

if st.button("Save Settings"):
    set_settings(
        {
            "models.generation": model,
            "models.embedding": emb,
            "models.offline": offline,
            # Generator
            "generator.provider": gen_provider,
            "generator.model": gen_model,
            "generator.ollama_model": gen_ollama_model,
            "generator.timeout_s": int(gen_timeout),
            "retrieval.top_k": top_k,
            "retrieval.rerank": rerank,
            "retrieval.rerank_strategy": rerank_strategy,
            "retrieval.rerank_top_n": rerank_top_n,
            "retrieval.mmr_lambda": float(mmr_lambda),
            "retrieval.rerank_model": rr_model,
            "retrieval.rerank_device": rr_device,
            "retrieval.rerank_batch_size": int(rr_bs),
            "retrieval.llm_judge_model": llm_judge_model,
            "retrieval.recency_filter_days": int(recency_days),
            "retrieval.recency_decay_lambda": float(recency_lambda),
            "memory.window_size": window,
            "memory.summarize": summarize,
            "flow.default": flow_default,
            "flow.low_cost_mode": bool(low_cost_mode),
            "flow.hyde_mode": hyde_auto,
            "flow.multi_hop_mode": multi_hop_auto,
            "flow.max_cost_estimate": float(max_cost),
            "flow.min_query_tokens": int(min_tokens),
            "index.semantic_chunking_enabled": bool(semantic_chunking),
            "metrics.ragas_enabled": ragas_enabled,
            "metrics.ragas_model": ragas_model,
        }
    )
    st.success("Settings saved. They will be applied on next page load.")
