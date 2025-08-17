# Project Plan — Dynamic Rerank & Guardrails (RAG Advanced)

Last updated: 2025-08-16 19:13 IST

## Objectives
- Implement dynamic rerank strategy dispatch across flows: none | mmr | cross_encoder | llm_judge.
- Enforce guardrails to optionally disable rerank per flow and auto-skip advanced flows when appropriate.
- Propagate new params (`rerank_strategy`, `mmr_lambda`, `llm_judge_model`, `guardrails_config`) through orchestrator and `registry.py`.
- Include rerank info in `extras` and rerank deltas + precision metrics.
- Maintain correct fallbacks (e.g., StandardFlow, cross-encoder fallback when LLM judge unavailable/offline).
- Keep persona behavior automated and ensure `persona_hint` continues to propagate to prompts and extras.

## Current Status
- Standard flow: dynamic rerank + guardrails integrated.
  - File: `app/core/flows/standard.py`
- Hybrid flow: done previously.
  - File: `app/core/flows/hybrid.py`
- HyDE flow: dynamic rerank + guardrails integrated; auto-skip supported.
  - File: `app/core/flows/hyde.py`
  - Recent changes: forwarding wrappers added for retrieval (`retrieve_dense`) and generation (`generate_answer`, `stream_answer`) to improve test patchability.
- Multi-hop flow: dynamic rerank + guardrails integrated; auto-skip supported; fallback uses module import for testability.
  - File: `app/core/flows/multi_hop.py`
  - Recent changes: default `auto_mode=False` to avoid unintended auto-skip in basic tests; fallback references `app.core.flows.standard` module.
- RAPTOR flow: dynamic rerank + guardrails integrated.
  - File: `app/core/flows/raptor.py`
- Registry propagation complete.
  - File: `app/core/flows/registry.py`
- Rerankers and embeddings confirmed for MMR, cross-encoder, and LLM judge with fallback.
  - File: `app/core/retrievers/rerank.py`, `app/core/embeddings.py`

## Next Steps (High Priority)
- Tests: expand coverage for Multi-hop and RAPTOR rerank/guardrails
  - Add: `test_multi_hop_rerank_strategy_mmr_applies_and_metrics`
  - Add: `test_multi_hop_rerank_strategy_llm_judge_fallback_cross_encoder_offline`
  - Add: `test_multi_hop_guardrails_disable_rerank`
  - Add: `test_raptor_rerank_strategy_mmr`
  - Add: `test_raptor_rerank_strategy_llm_judge_fallback`
  - Add: `test_raptor_guardrails_disable_rerank`
- Tests: registry/orchestrator parameter propagation
  - Add: `test_orchestrator_propagates_rerank_strategy_to_flows`
  - Add: `test_registry_passes_guardrails_config`
- Run full test suite to catch regressions: `pytest -q`

## Nice-to-haves
- Add logging around guardrails decisions and rerank strategy selection for better observability.
- Extend metrics dashboard/visualization to display rerank deltas and precision improvements.

## End-to-End & UX
- Verify Admin Settings/UI sync for:
  - `rerank_strategy`, `mmr_lambda`, `llm_judge_model`, `guardrails_config.disable_rerank`.
- Ensure persona selection stays automated (no UI toggles) and `persona_hint` flows into prompt builders and `extras`.
  - References: `build_answer_prompt()` usage in flows and `extras["persona_hint"]` population.

## Risks & Fallbacks
- LLM judge rerank requires OpenAI API; offline/errored -> fallback to cross-encoder.
- Cross-encoder requires `sentence-transformers`; if missing, MMR remains available (uses default embedder).
- FAISS/DB access in tests: use proper patching/mocking to avoid real retrieval when not needed.

## Environment & Dependencies
- Env vars: `OPENAI_API_KEY` for LLM judge; offline mode bypasses.
- Packages: `sentence-transformers`, `openai`, `numpy`, `jinja2`, `python-dotenv`.

## Changelog (recent)
- Guardrails refined: short but clearly natural-language queries no longer auto-skip advanced flows; prevents unintended fallback to Standard on concise questions. File: `app/core/flows/guardrails.py`.
- HyDE testability: added forwarding wrappers for `retrieve_dense`, `generate_answer`, and `stream_answer` so patches on either HyDE or underlying modules are honored. File: `app/core/flows/hyde.py`.
- Multi-hop fallback changed to use `app.core.flows.standard` module import; `auto_mode` default set to `False`.
- RAPTOR params extended with dynamic rerank fields.
- Metrics and extras enriched across flows for rerank info and deltas.
