# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-08-12
### Added
- Streamlit UI with four pages: Upload/Index, Chat, Runs & Metrics, Admin Settings.
- Multiple RAG flows: standard, hybrid (RRF), HyDE, multi-hop.
- Embeddings: OpenAI + sentence-transformers (offline).
- Reranker: bge cross-encoder with toggle and top-N control.
- Memory: window + summary with resilient fallback summarizer.
- Indexing: FAISS + SQLite, content-hash dedup, disk embedding cache, re-embed changed, purge.
- Metrics: lite metrics + optional RAGAS wrapper with graceful skip.
- Persistence: runs, settings, messages, memory in SQLite.
- CLI tools: ingest, eval_offline.
- Unit tests for core modules and flows.
- Static checks: ruff and mypy.

### Changed
- Prompt templates including multi-hop merge for better synthesis with citations.

### Fixed
- Windows/FAISS and tokenizer resilience; tests skip or fallback gracefully.
