# rag-advanced-local

**API-first** modular RAG application with Streamlit UI. Uses OpenAI for embeddings and generation by default. Optional offline mode available with sentence-transformers (install `[offline]` extra). Config files live in `app/config/`.

## Quick Reference

- [Installation](#installation)
- [Usage: Online vs Offline](#usage-online-vs-offline)
- [Streamlit Pages](#streamlit-pages-screenshots)
- [Flow Comparison](#flow-comparison)
- [Offline Capability Matrix](#offline-capability-matrix)
- [Performance Tips](#performance-tips)
- [Hybrid retrieval configuration](#hybrid-retrieval-configuration)
- [Streaming multi-query behavior](#streaming-multi-query-behavior)
- [Streaming & UX](#streaming--ux)
- [Answer Formatting & Evidence Policy](#rag-answer-formatting--evidence-policy)
- [Citations & Run Debugger](#citations-merged-1-based--run-debugger)
- [Prompt Templates](#prompt-templates)
- [Caching & Performance Internals](#caching--performance-internals)
- [Maintenance](#maintenance-purge-and-re-embed)
- [Data & Storage](#data--storage)
- [Development](#development)
- [Offline testing (no network)](#offline-testing-no-network)
- [Troubleshooting](#troubleshooting)
- [Architecture & Internals](#architecture-and-internals)

## Installation

### Prerequisites
- **Python 3.12** (recommended) or 3.11+
- [UV](https://github.com/astral-sh/uv) - A fast Python package manager
- OpenAI API key (for default API-first mode)

### Setup with UV (Recommended)

1. Install UV (if not already installed). Follow the official instructions:
   https://docs.astral.sh/uv/getting-started/installation/

2. Create and activate a virtual environment:
   ```bash
   uv venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

3. Install the package in development mode:
   ```bash
   # API-first mode (default, lightweight)
   uv pip install -e ".[dev]"

   # With offline capabilities (adds sentence-transformers)
   uv pip install -e ".[dev,offline]"

   # With evaluation tools (adds RAGAS)
   uv pip install -e ".[dev,eval]"

   # Everything (offline + eval + experimental)
   uv pip install -e ".[dev,full]"
   ```

### Common UV Workflows

- **Install a new package**:
  ```bash
  uv pip install package-name
  ```

- **Update all dependencies**:
  ```bash
  uv pip install --upgrade -e "."
  ```

Note: This project uses `pyproject.toml` as the single source of truth for dependencies. A `requirements.txt` is not required.

### Alternative: Standard pip

If you prefer not to use UV:
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### Environment Setup (Required for OpenAI)

Create a `.env` file in the project root with any needed variables. For OpenAI features:

```
OPENAI_API_KEY=sk-...
```

Run UI:

```bash
streamlit run app/ui/streamlit_app.py
```

CLI ingestion:

```bash
rag-advanced-ingest --paths data/samples
```

Offline evaluation CLI (no OpenAI required):

```bash
rag-advanced-eval-offline --help
```

Quick setup scripts (optional):

```bash
# macOS/Linux
scripts/setup.sh
# Windows (PowerShell)
scripts/setup.bat
```

## Usage: Online vs Offline

- Online (OpenAI):
  - Admin Settings → Models: set Generation model (e.g., gpt-4o-mini), disable Offline.
  - RAGAS metrics can be enabled (requires `OPENAI_API_KEY` and `ragas`).

- Offline:
  - Admin Settings → enable Offline mode.
  - Uses sentence-transformers for embeddings and reranker; generator is a local stub.
  - RAGAS will be skipped automatically.
  - Tip: For large indexing runs, prefer Offline mode to avoid OpenAI rate limits, then switch Online for chat.

## Streamlit Pages (Screenshots)

- Upload & Index: add docs, build FAISS, re-embed changed, Purge Index & Cache, and a guarded Full Purge (Everything).
- Chat RAG: simple chat box; the Orchestrator automatically selects retrieval/generation flows and params. View answer, sources, metrics, and the selected plan.
  - Automated persona: on each turn, the Orchestrator selects a persona (e.g., Concise Expert, Supportive Mentor, Analytical Researcher, Playful Creator) automatically with light intent-aware randomness. There are no UI controls; the chosen persona is displayed in the Run Debugger (see `extras.persona_hint`).
  - Dynamic formatting: the system may apply automatic formatting hints (e.g., steps, tables, lists) based on your question; shown in Run Debugger as `extras.format_hint`.
  - Streaming: answers stream token-by-token. The UI shows micro-batched updates, a live elapsed timer, and status badges; token/cost estimates are computed post-stream.
- Runs & Metrics: see recent runs, metrics, and timings.
- Admin Settings: toggle offline, models, retrieval defaults, memory, RAGAS.
  - Index Maintenance: Re-embed missing, Build index, Purge current index & cache, and a guarded Full Purge (Everything). Mirrors maintenance actions from Upload & Index for convenience.
  - Retrieval extras: configure Recency filter (days) and Recency decay (λ) to prefer fresher chunks.

Placeholders (add your screenshots under `/docs/img/`):
- `docs/img/upload.png`
- `docs/img/chat.png`
- `docs/img/runs.png`
- `docs/img/settings.png`

## Flow Comparison

| Flow      | Retrieval              | Pros               | Cons             | When to use         |
|-----------|------------------------|--------------------|------------------|---------------------|
| standard  | dense                  | fastest, simple    | misses keywords  | general Q&A         |
| hybrid    | BM25 + dense (RRF)     | robust on keywords | slight latency   | code/docs w/ terms  |
| HyDE      | dense + seed           | better recall      | extra gen step   | sparse corpora      |
| multi-hop | per-subq + merge       | compositional Qs   | slower           | "A then B" queries  |
| RAPTOR    | hierarchical summaries + dense | robust long docs  | WIP scaffold     | large corpora       |

## Offline Capability Matrix

| Feature            | Offline |
|--------------------|---------|
| Embeddings (ST)    | ✅      |
| Rerank (bge)       | ✅ (CPU) |
| Generator          | ⚠️ stub (no real LLM) |
| RAGAS full         | ❌ (needs API) |
| Lite metrics       | ✅      |

## Performance Tips

- Chunking: tune `app/config/default.yaml` → `index.chunk_size` (e.g., 400–1000) and `index.overlap` (e.g., 80–200).
- Retrieval top-k: smaller is faster; hybrid/HyDE/multi-hop benefit from rerank.
- Reranker: enable for quality; set `rerank_top_n` to a modest number (6–10).
- Embedding cache: re-embed changed only via Upload & Index page.
- FAISS: normalize vectors (cosine) already handled in `build_faiss()`.
- Recency: use `recency_filter_days` to exclude too-old chunks entirely; use `recency_decay_lambda` to gently prefer newer material without hard filtering.

### Hybrid retrieval configuration

Keys live in `app/config/default.yaml` under `retrieval` and can be overridden via Admin Settings or env merge.

- `rrf_k`: Constant in Reciprocal Rank Fusion 1/(k + rank). Higher dampens tail influence.
- `rrf_weight_bm25` / `rrf_weight_dense`: Weights for lexical vs semantic lists. Increase one to bias fusion accordingly.
- `multi_query_n`: Total queries fused (original + rewrites). `1` disables multi-query. Offline uses simple heuristic rewrites; online may use OpenAI.

Example:

```yaml
retrieval:
  rrf_k: 10
  rrf_weight_bm25: 0.6
  rrf_weight_dense: 0.4
  multi_query_n: 3
```

#### How weighted fusion works

- Fusion uses Reciprocal Rank Fusion across BM25 and dense lists with source weights.
- Intuition: higher-ranked items contribute more; weights bias lexical vs semantic.

```python
# For each ranked list (BM25 and dense), accumulate per chunk id
score[id] += w_source * 1.0 / (rrf_k + rank + 1)
# Higher weight -> more influence; smaller rrf_k -> steeper decay of lower ranks
```

- Deduplication: chunks are keyed by `chunk.id`; first-seen metadata is preserved.
- Optional rerank is applied after fusion when `retrieval.rerank: true`.

#### Tuning presets (YAML snippets)

- Balanced (default-ish)

```yaml
retrieval:
  rrf_k: 10
  rrf_weight_bm25: 0.5
  rrf_weight_dense: 0.5
  multi_query_n: 1
```

- Lexical-heavy (code/docs with exact terms)

```yaml
retrieval:
  rrf_k: 8
  rrf_weight_bm25: 0.8
  rrf_weight_dense: 0.2
  multi_query_n: 1
```

- Semantic-heavy (conceptual queries)

```yaml
retrieval:
  rrf_k: 12
  rrf_weight_bm25: 0.3
  rrf_weight_dense: 0.7
  multi_query_n: 1
```

- High-recall (multi-query expansion)

```yaml
retrieval:
  rrf_k: 10
  rrf_weight_bm25: 0.5
  rrf_weight_dense: 0.5
  multi_query_n: 4  # original + 3 rewrites
```

#### Admin Settings and overrides

- Admin Settings exposes common knobs (top-k, rerank, rerank_top_n).
- Advanced hybrid knobs (`rrf_k`, weights, `multi_query_n`) live in `app/config/default.yaml` under `retrieval`.
- If your UI build does not expose these yet, edit the YAML and restart the app.

### OpenAI Embedding Rate-Limits (429) — Built-in Mitigations

- `app/core/embeddings.py` batches inputs per `embeddings.create()` call, throttles to a safe RPM, and retries with exponential backoff.
- Configure via environment variables:
  - `OPENAI_EMBED_RPM_CAP` (default: `90`) — target requests-per-minute throttle.
  - `OPENAI_EMBED_MAX_ITEMS` (default: `256`) — max items per batch request.
  - `OPENAI_EMBED_TPM_CAP` (optional) — tokens-per-minute budget for embeddings. Enables token-aware micro-batching and a TPM token-bucket limiter.
- For heavy indexing, you can also run Offline mode to bypass OpenAI entirely and use SentenceTransformers.

Notes:
- Token-aware micro-batching uses tiktoken (if available) to keep each request under a computed token budget (≈60% of TPM divided by RPM).
- Retries honor `Retry-After` headers and add jitter; only 429/5xx/timeouts/connection errors are retried. 400s (e.g., too-long input) fail fast with details.

## Streaming & UX

- Streaming answers: the Chat page renders tokens as they arrive from `generator.stream_answer` via `orchestrator.plan_and_stream()`.
- Micro-batching: the UI batches frequent token updates for smoothness without lag.
- Status: a live elapsed timer and header badges reflect the active flow and rerank status.
- Usage and cost: token usage/cost are estimated after the stream (uses `tiktoken` when available, else a heuristic) and persisted in runs.

### Streaming multi-query behavior

- When `retrieval.multi_query_n > 1`, the Hybrid flow expands the user query into rewrites and retrieves per-query.
- Fusion: `rrf_fuse_multi` merges BM25 and dense lists across all queries with weights (`rrf_weight_bm25`, `rrf_weight_dense`) and `rrf_k`.
- Generation streams over the fused context (optionally reranked) like single-query.
- Benefits: improved recall for ambiguous/underspecified prompts at minor latency cost.

## RAG Answer Formatting & Evidence Policy

### Contract: one message, two artifacts

- **Human message (Markdown)**
  - Direct answer (1–3 sentences) → Key points → Sources → Status footer.
  - Citations use 1-based footnotes that map to the Merged Context list (deduped, deterministic).
- **Machine envelope (JSON)**
  - Run/session IDs, models, retrieval params, usage/cost, timings, citation map, safety flags, memory hints, and exit state. Persona appears only in the envelope.

#### Skeleton (rendered)

```
**<direct answer>** [^1][^2]

**Key points**
- <point 1> [^2]
- <point 2> [^3]
- _Caveat:_ <optional, see below>

**Sources**
[^1] <Title> — <Author>, <Year> (loc). [doc:<short-id>#<chunk>]
[^2] <Title> — <Author>, <Year>. [doc:<short-id>#<chunk>]

— *Online • <model> • Flow: <type> (rerank=<strategy>) • <X.X>s • Cost ≈ $<0.000X>*
```

No path leakage: never show OS paths in the UI. Use titles + `[doc:<short-id>#chunk]`. Full provenance (paths/pages/scores) lives in the envelope.

### Post-stream consolidation (what users see)

- During streaming, the UI displays only the Direct answer line (with fixed footnote numbers).
- On completion, the streamed text swaps to the consolidated card: adds Key points, Sources, and the Status footer.
- “Show context” expander includes evidence snippets (below).

### No-answer & caveat behavior

- **No-answer template** (used when evidence is insufficient):

  ```
  I don’t know from the provided sources.

  How to proceed
  – Upload/select more relevant docs
  – Relax recency/filters
  – Try a broader/alternative phrasing
  ```

- **Gating (defaults)**
  - No-answer if `retrieved_count == 0`, or `(retrieved_count ≤ 1 and context_precision_lite < 0.50)`.
  - Caveat line if `groundedness_lite < 0.70` or `context_precision_lite < 0.80`.
  - These are conservative and can be adjusted; stronger faithfulness metrics can replace/augment later.

### “Show context” expander (evidence policy)

- Shows ≤ 120-char sanitized snippets per retrieved chunk.
- Highlights up to 6 query terms (bold), collapses whitespace, strips HTML/JS.
- Labels each item as: `**<Title> [doc:<short-id>#<chunk>]** — <snippet>`.
- Deterministic order matches the merged citations; low-relevance items may be omitted from the human view but remain in the envelope.
- Accessibility: list semantics, readable contrast, keyboard focusable.

### Deterministic citations

- Footnotes are assigned from the merged, deduped context before generation.
- Stable sort by rank → title, then map `[^i]` to `[doc:<short-id>#<chunk>]`.
- Short IDs derive from titles + a small hash; full paths stay in the envelope.

### Status footer

- Single line at the bottom: `Online • <model> • Flow: <type> (rerank=<strategy>) • <X.X>s • Cost ≈ $<0.000X>`.
- Keeps plumbing out of the way while preserving auditability.

### Envelope (logged & persisted)

```json
{
  "v": 1,
  "session_id": "…",
  "run_id": "…",
  "models": { "gen": "…", "embed": "…" },
  "flow": { "type": "hybrid", "top_k": 12, "rerank": "off", "rrf_k": 60 },
  "prompt": { "hash": "sha256:…", "chars_sample": "What is …" },
  "citations": [ { "id": 1, "title": "…", "doc_id_short": "…", "chunk_id": "…", "loc": "…", "full_path": "…", "rank": 1, "score": 0.82 } ],
  "retrieval": { "dense_k": 12, "bm25_k": 30, "hybrid_rrf_k": 60, "rank_deltas": [ … ] },
  "usage": { "tokens_in": 0, "tokens_out": 0, "embed_tokens": 0, "cost_usd": 0.0 },
  "timings_ms": { "retrieve": 0, "generate_ttfb": 0, "generate_total": 0 },
  "metrics": { "context_precision_lite": 0.0, "groundedness_lite": 0.0 },
  "safety": { "sanitized": true, "prompt_injection_flag": false },
  "memory": { "window_turns": 0, "summary_used": false },
  "exit": "ok"
}
```

Persona hint (and other internal routing metadata) appears only here.

### Config knobs (defaults)

```
ui.snippet.max_len = 120
ui.status.show_footer = true
ui.citations.hide_paths = true
policy.no_answer.min_sources = 1
policy.no_answer.min_context_precision = 0.50
policy.caveat.min_groundedness = 0.70
policy.caveat.min_context_precision = 0.80
logging.envelope.persist = true
logging.envelope.include_persona = true
```

### QA checklist (manual)

- Streaming swaps to consolidated card on completion; no flicker.
- No-answer triggers correctly; `exit="no_answer"`; no citations in envelope for no-answer runs.
- Caveat shows only under thresholds.
- Evidence snippets ≤120 chars, with term highlights, and no local paths.
- Envelope in logs/DB includes exit state, usage, timings, safety flags; persona not shown in UI.

### Silencing FAISS NumPy deprecation (optional)

Pick one:

- `pytest.ini` (recommended)

  ```ini
  [pytest]
  filterwarnings =
      ignore::DeprecationWarning:faiss.loader
  ```

- `pyproject.toml`

  ```toml
  [tool.pytest.ini_options]
  filterwarnings = [
    "ignore:.*numpy.core._multiarray_umath.*:DeprecationWarning",
    "ignore::DeprecationWarning:faiss.loader",
  ]
  ```

- `tests/conftest.py`

  ```python
  import warnings
  warnings.filterwarnings("ignore", category=DeprecationWarning, module="faiss.loader")
  ```

- Ad-hoc run

  ```bash
  pytest -q -W ignore::DeprecationWarning:faiss.loader
  ```

## Citations (Merged, 1-based) & Run Debugger

- Inline style: citations use inline footnotes like `[^i]` where `i = 1..N` maps to the numbered "Context" (or "Merged Context") list shown in the UI.
- Multi-hop: per-hop context lists are shown for reference, but citations must refer to the merged 1-based list. This yields consistent, auditable sources.
- Run Debugger: the Chat page's context panel displays "Merged Context (citation order: 1..N)" and shows retrieval/rerank details, persona and formatting hints, and prompt/memory snapshots for transparency.

## Prompt Templates

- Answer synthesis: `app/core/prompting/templates/answer.j2` — explicitly instructs `[^i]` mapping to the numbered Context list.
- Multi-hop merge: `app/core/prompting/templates/multi_hop_merge.j2` — renders a 1..N "Merged Context" section and clarifies per-hop indices are reference-only.
- Query condensing: `app/core/prompting/templates/condense_query.j2` — emphasizes preserving entities, dates, units, and numeric ranges (e.g., "between 2019 and 2021").
- HyDE seed: `app/core/prompting/templates/hyde_seed.j2` — retrieval-friendly phrasing for better recall.

## Caching & Performance Internals

- Model caching: SentenceTransformer and CrossEncoder models are cached (singleton-style) for reuse across requests.
- Retrieval cache: dense/BM25 retrieval may use an LRU cache to avoid repeated lookups for identical queries in a short window.
- Embedding on-disk cache: per `(model, text_sha)` `.npy` files prevent re-embedding unchanged chunks.

## Maintenance: Purge and Re-embed

- **Purge Index & Cache** (`app.core.indexer.purge_index()`)
  - Removes the FAISS index and embedding cache files for the active embedding tag.
  - Cleans index metadata so a fresh build can proceed cleanly.
  - Use when models/settings change or index seems stale.
  - Available in UI via Upload & Index and Admin Settings → Index Maintenance.

- **Full Purge (Everything)** (`app.core.indexer.full_purge()`)
  - Deletes all documents/chunks/indices and all embedding cache files.
  - Optional checkboxes allow deleting uploaded files (`data/uploads/`), runs/metrics, and chat history (messages + memory).
  - UI requires an explicit Confirm checkbox before enabling the button.

- **Re-embed Changed Only**
  - Efficiently updates only new/changed chunks based on content hashing; avoids reprocessing unchanged text.
  - Parallel ingestion: parsing/splitting can use multiple workers. On Windows, the default is workers=1 for safety; you can increase it in the Upload & Index page if resources allow.

> ⚠️ Caution: Full Purge is destructive. Ensure you have backups of any important uploads or metrics before proceeding.

## Data & Storage

- SQLite database: `app/rag.db` (+ WAL/SHM managed by SQLite).
- Uploads directory: `data/uploads/` (user files).
- Embedding cache: `data/emb_cache/` (per-model, per-text_sha `.npy`).
- FAISS indices: `data/indices/<tag>/` (per embedding model tag).
- Samples: `data/samples/` (example docs).

Ignored (via `.gitignore`): all runtime/generated artifacts above, including FAISS `*.faiss` files, SQLite temp files, `.ruff_cache/`, etc. Verified with `git status --ignored -s`.

## Development

Run quick checks:

```bash
scripts/dev_check.sh    # macOS/Linux
scripts/dev_check.ps1   # Windows
```

Run tests with coverage:

```bash
pytest --cov=app --cov-report=term-missing
```

### Offline testing (no network)

Run tests and the app fully offline using the local stub generator and Sentence-Transformers (if installed):

```powershell
$env:OFFLINE_MODE = "true"
uv run pytest -q

# Launch UI offline
uv run streamlit run app/ui/streamlit_app.py
```

## UV (Ultra-fast Python) Quickstart

Prefer UV for speed and reproducibility:

```bash
# Create a virtualenv (managed by UV)
uv venv

# Install project with dev extras
uv pip install -e .[dev]

# Run tests / coverage / static checks
uv run pytest --cov=app --cov-report=term-missing
uv run ruff check .
uv run mypy app

# Launch the UI
uv run streamlit run app/ui/streamlit_app.py
```

## Troubleshooting

- OpenAI API key not detected
  - Ensure `.env` exists in project root with `OPENAI_API_KEY=...`.
  - The Streamlit app loads `.env` explicitly in `app/ui/streamlit_app.py`.

- OpenAI 429 (Rate limit exceeded on embeddings)
  - Mitigations are built-in (micro-batching, throttling, backoff). If it persists:
    - Reduce RPM: set `OPENAI_EMBED_RPM_CAP` (e.g., 80).
    - Reduce batch size: set `OPENAI_EMBED_MAX_ITEMS` (e.g., 128).
    - Prefer Offline mode for indexing, then switch back Online for chat/rerank.

- RAGAS/LangChain deprecation warnings
  - Harmless; tests still pass. Suppress in `pytest.ini` with `filterwarnings` if desired.

- First run is slow on reranker
  - Cross-encoder (bge) downloads on first use; subsequent runs are fast and CPU-only by default.

- Windows parallel ingestion
  - Default workers=1 for safety; increase in Upload & Index page if resources allow.

- Windows + OneDrive hardlink issues with UV
  - If you see file locking or hardlink errors on OneDrive-backed folders, force copy mode:

    ```powershell
    $env:UV_LINK_MODE = "copy"
    uv run pytest -q
    ```
  - This avoids hardlinks that OneDrive may block.

- Offline mode behavior
  - Generator is a stub; RAGAS is skipped automatically. Expect different answer style than online models.

### Dev Check Minimal Output
- **Note**: `scripts/dev_check.ps1` and `scripts/dev_check.sh` run tests, ruff, and mypy with minimal output.
- **For detailed output**: Run individual commands (`pytest -v`, `ruff check --diff`, `mypy app/`).

## Architecture and Internals

### Capabilities

- **Document ingestion and indexing**
  - Upload PDFs/TXT/MD, chunk them, persist chunks + metadata to SQLite, and build a FAISS dense index.
  - Re-embed only changed/missing chunks and purge all indexes/caches.
- **Multiple retrieval/generation flows**
  - Standard dense, Hybrid (BM25 + dense via RRF), HyDE (hypothetical doc expansion), Multi-hop (sub-queries and merge).
  - RAPTOR (scaffold): hierarchical summarization tree (stubbed to dense for now) with final top-k.
  - Optional reranking via cross-encoder.
- **Chat with memory**
  - Window memory and optional summary memory; injects memory into prompts.
  - Automated persona selection with no UI controls; persona hint is threaded through orchestrator → flows → templates and logged in run `extras`.
- **Metrics**
  - Built-in “lite” metrics.
  - Optional RAGAS metrics (requires OpenAI + ragas).
- **Modes**
  - Online (OpenAI models for embeddings/generation).
  - Offline (Sentence-Transformers embeddings, local stub generator; RAGAS disabled automatically).
- **Persistence**
  - SQLite DB for messages, runs/metrics, and admin settings.
- **CLI tools**
  - Ingest paths (`rag-advanced-ingest`) and offline eval (`rag-advanced-eval-offline`).

### Persona Automation (No UI controls)

- Personas include: "Concise Expert", "Supportive Mentor", "Analytical Researcher", "Playful Creator".
- Selection is automatic per turn in `app/core/orchestrator.py` with light intent-aware weighting and randomness.
- The chosen `persona_hint` propagates orchestrator → flows (`app/core/flows/*.py`) → prompt builder (`app/core/prompting/builder.py`).
- There are no persona controls in the UI (`app/ui/pages/2_💬_Chat_RAG.py`). The active persona is visible in the Run Debugger under `extras.persona_hint`.

### Front-end (Streamlit)

- **Entry point**: `app/ui/streamlit_app.py`
  - Loads `.env` via `dotenv.load_dotenv()`.
  - Loads config using `app.core.utils.load_config()`.
  - Initializes DB (`app.core.storage.init_db()`).
  - Sets page config and provides a simple status sidebar.
- **Pages** (`app/ui/pages/`):
  - `1_📤_Upload_and_Index.py`
    - Upload files using `st.file_uploader`.
    - Calls:
      - `app.core.indexer.ingest_paths()` to chunk and persist.
      - `app.core.indexer.build_faiss()` to build/refresh FAISS index.
      - `app.core.indexer.reembed_changed_only()` to update only changed embeddings.
      - `app.core.indexer.purge_index()` ("Purge Index & Cache") to clear FAISS index, caches, and metadata for the active embedding tag.
      - `app.core.indexer.full_purge()` ("Full Purge (Everything)") to delete all documents/chunks/indices and embedding cache; optional deletion of uploads, runs/metrics, and chat history (messages + memory) with confirmation.
    - Uses `app.core.utils.DATA_ROOT` for `data/uploads/`.
  - `2_💬_Chat_RAG.py`
    - Auto mode: Orchestrator plans and runs the best flow and parameters; no manual flow/rerank toggles.
    - Memory is applied automatically per config (window size, optional summary cadence).
    - Persona is selected automatically per turn (no dropdowns or toggles); visible in Run Debugger under `extras.persona_hint`.
    - Hydrates chat history via `app.core.storage.get_recent_messages()`.
    - Memory:
      - Window memory: `app.core.memory.window.WindowMemory`.
      - Summary memory: `app.core.memory.summarizer.SummaryMemory` (optional).
      - Prompt memory text via `app.core.prompting.builder.build_memory_text()`.
    - Flow engine: selected via `app.core.orchestrator.Orchestrator.plan_and_run()`; under the hood builds the chosen flow and executes it.
    - Persists messages (`add_message`) and runs (`add_run`).
    - Metrics:
      - Uses lite metrics from the bundle.
      - Optional RAGAS via `app.core.metrics.ragas_wrap.eval_ragas()`.
    - Shows citations and retrieved context; displays timings and metrics.
  - `3_📈_Runs_&_Metrics.py`
    - Retrieves runs using `app.core.storage.get_recent_runs()` and displays a DataFrame (with CSV export).
  - `4_⚙️_Admin_Settings.py`
    - Loads config and DB; reads/writes persisted settings:
      - `get_settings()` / `set_settings()` from `app.core.storage`.
    - Controls:
      - Models (generation + embedding), Offline toggle.
      - Retrieval defaults (top-k, rerank, rerank_top_n).
      - Memory defaults (window size, summarize toggle).
      - Default flow.
      - RAGAS enable + model name.
    - Displays config snapshot using `app.core.utils.pretty_json()`.

    Admin Settings mapping (config → UI)

    - `models.generation` / `generator.model` → Generation model dropdown.
    - `models.embedding` → Embedding model/tag selection.
    - `models.offline` → Offline toggle.
    - `retrieval.top_k` → Retrieval top-k.
    - `retrieval.rerank` / `retrieval.rerank_top_n` → Rerank toggle and top-n.
    - Advanced hybrid knobs (edit YAML): `retrieval.rrf_k`, `retrieval.rrf_weight_bm25`, `retrieval.rrf_weight_dense`, `retrieval.multi_query_n`.

### Back-end (Core modules)

- **Config and utilities**: `app/core/utils.py`
  - `load_config()` loads YAML config with environment overrides (OmegaConf).
  - Provides helpers such as `DATA_ROOT`, `pretty_json()`, and logging (`get_logger()` used in `streamlit_app.py`).
- **Types**: `app/core/types.py`
  - Core dataclasses (e.g., `Chunk`, `RetrievedDoc`, `AnswerBundle`) used across indexing, retrieval, and generation.
- **Indexer**: `app/core/indexer.py`
  - `ingest_paths(paths, chunk_size, overlap)`: parses and splits documents into chunks; stores in SQLite with metadata.
  - `build_faiss(offline, emb_model_st, emb_model_oa)`: builds a FAISS index of embeddings.
  - `reembed_changed_only(offline, emb_model_st, emb_model_oa)`: hashes content, re-embeds only changed/missing chunks, updates caches.
  - `purge_index(offline, emb_model_st, emb_model_oa)`: removes FAISS index, clears embedding caches, and prunes metadata.
- **Embeddings**: `app/core/embeddings.py`
  - Hashing and cache path:
    - `_sha(text: str) -> str` for content hashing.
    - `_cache_path(model_name, text) -> Path` into `CACHE_DIR` (e.g., `emb_cache/`).
  - Adapters:
    - `embed_openai(texts: list[str], model_name=...) -> np.ndarray` (OpenAI client; uses on-disk cache, micro-batching, throttling, and backoff to reduce 429s).
    - `embed_st(texts: list[str], model_name=..., normalize_embeddings=True) -> np.ndarray` (uses `SentenceTransformer`).
  - Caching: per-text `.npy` files keyed by `(model, text_sha)`. In-memory model cache `_ST_MODELS`.
  - `get_default_embedder(offline, st_model, oa_model)` picks OpenAI vs Sentence-Transformers.
- **Retrievers**: `app/core/retrievers/`
  - `dense.py`: vector search over FAISS index.
  - `bm25.py`: lexical search via rank-bm25.
  - `hybrid.py`: merges multiple result lists using Reciprocal Rank Fusion (RRF).
  - `rerank.py`: optional reranking (bge cross-encoder); toggled in UI and flows.
- **Flows**: `app/core/flows/`
  - `base.py`: flow interface and common helpers.
  - `standard.py`: dense retrieval, optional rerank, then generation.
  - `hybrid.py`: BM25 + dense retrieval combined via RRF, optional rerank, then generation.
  - `hyde.py`: generates a hypothetical doc/seed, retrieves by seed + query, merges, optional rerank, then generation.
  - `multi_hop.py`: decomposes into sub-queries, retrieves per hop, merges, optional rerank, then generation.
  - `registry.py`: `make_flow(flow_type, offline, gen_model, emb_st, emb_oa, params)` builds and returns the selected flow engine.

  - Notes (guardrails and HyDE testability)
    - Guardrails short-query handling: `app/core/flows/guardrails.py` avoids auto-skipping advanced flows for short but clearly natural-language queries (e.g., ends with `?`, `!`, `.` or contains who/what/why/when/how). This prevents unintended fallback to Standard on concise questions.
    - HyDE forwarding wrappers: `app/core/flows/hyde.py` exposes thin wrappers for `retrieve_dense`, `generate_answer`, and `stream_answer` that delegate to their underlying modules. Test patches/mocks applied via either the HyDE import path or the source modules are honored.

- **Generation**: `app/core/generator.py`
  - Wrapper that supports:
    - Online: OpenAI generation (when `OPENAI_API_KEY` present).
    - Offline: local stub generator (deterministic/simple).
  - Returns `AnswerBundle` with `answer_md`, `citations`, `retrieved`, `timings`, and optional `usage` (token/cost).
- **Memory**: `app/core/memory/`
  - `window.py`: `WindowMemory` keeps recent turns; supports `.add(role, content)`, `.get(n)`.
  - `summarizer.py`: `SummaryMemory` stores/fetches a summary for long chats and can update it periodically.
- **Prompting**: `app/core/prompting/builder.py`
  - `build_memory_text(mem_window_texts, mem_summary_text, mem_hits)` composes memory context to include in prompts.
  - `build_answer_prompt(question, memory, docs, format_hint=None, persona_hint=None)` renders prompts with optional dynamic formatting and automated persona guidance.
  - Templates reside as package data (see `pyproject.toml` `[tool.setuptools.package-data]`).
- **Metrics**: `app/core/metrics/`
  - `raglite.py`: computes lite metrics (answer relevancy, context precision, groundedness, delta precision).
  - `ragas_wrap.py`: optional RAGAS metrics (`ragas_answer_relevancy`, `ragas_faithfulness`, `ragas_context_precision`), gracefully skips if missing deps or API key.
  - `token_cost.py`: token and cost estimation helpers for runs.
- **Storage**: `app/core/storage.py`
  - Initializes SQLite schema on connect (`init_db()`).
  - Messages: `add_message(session_id, role, content)`, `get_recent_messages(session_id, limit)`.
  - Runs: `add_run(session_id, flow, question, answer, citations, timings, metrics, cost)`, `get_recent_runs(limit)`.
  - Settings: `get_settings()`, `set_settings(dict)`; simple key-value via dotted keys.
- **CLI**: `app/scripts/`
  - `ingest.py`: Typer app exposed via `rag-advanced-ingest`.
  - `eval_offline.py`: Typer app exposed via `rag-advanced-eval-offline`.
- **App data and packaging**
  - `pyproject.toml` declares:
    - Dependencies (Streamlit, FAISS, rank-bm25, sentence-transformers, Jinja2, OmegaConf, structlog, pypdf, python-dotenv, tiktoken, typer, pandas, numpy, scikit-learn, SQLAlchemy, sqlite-utils, openai).
    - Optional extras: `full` (ragas, chromadb, llama-cpp-python), `dev` (pytest, pytest-cov, ruff, mypy).
    - Console scripts: `rag-advanced-ingest`, `rag-advanced-eval-offline`.
    - Package data: Jinja templates, YAML configs, etc.

### End-to-end Workflows

- **Ingestion & Indexing** (`1_📤_Upload_and_Index.py`)
  - User uploads files → written to `data/uploads/`.
  - `ingest_paths()`:
    - Reads files, splits into chunks (size/overlap from `cfg['index']`), stores chunk text + metadata in SQLite.
    - Tracks doc and chunk stats (added/updated/unchanged).
  - `build_faiss()`:
    - Chooses embedder via `get_default_embedder()` based on `cfg['models']['offline']`.
    - Embeds all chunks (using on-disk cache per (model, text_sha)).
    - Builds FAISS index; returns path and dimension.
  - `reembed_changed_only()`:
    - Uses content hashing to skip unchanged.
    - Embeds only new/changed chunks; updates cache and index if needed.
  - `purge_index()`:
    - Drops FAISS index, clears embedding cache files, prunes index metadata.
- **Chat & Answering** (`2_💬_Chat_RAG.py`)
  - On first load:
    - Loads config; initializes session ID, message list, and `WindowMemory()`.
    - Hydrates last N messages from SQLite (`get_recent_messages()`).
  - When user sends a prompt:
    - Updates memory window; may update/fetch summary via `SummaryMemory`.
    - Builds `memory_text` via `build_memory_text()`.
    - Selects flow using `make_flow(flow_type, offline, gen_model, emb_st, emb_oa, params)`.
    - Flow executes retrieve → optional rerank → generate (OpenAI or stub).
    - Displays answer, sources (citations), and optionally context blocks (retrieved chunks and per-hop artifacts).
    - Computes lite metrics and, if enabled, RAGAS. Token and cost estimates are displayed with each run (when available).
    - Run Debugger transparency: shows `extras` including `plan` (orchestrator decisions), `prompt.text`, `memory.text`, retrieval/rerank lists and deltas (when applicable), plus `format_hint` and `persona_hint` used for the turn.
    - Persists user and assistant messages; saves the run record (`add_run()`).
- **Runs & Metrics** (`3_📈_Runs_&_Metrics.py`)
  - Fetches last runs via `get_recent_runs()`; shows DataFrame with core and RAGAS metrics if available; CSV export.
- **Admin Settings** (`4_⚙️_Admin_Settings.py`)
  - Loads defaults + saved overrides.
  - On save: writes key-values via `set_settings()` (effective next reload; UI notes this).

### Interdependencies

- UI pages depend on:
  - `app.core.utils.load_config`, `pretty_json`, `DATA_ROOT`.
  - `app.core.storage` for settings, messages, runs, and DB init.
  - `app.core.indexer` for ingestion/index maintenance.
  - `app.core.flows.registry.make_flow` to construct flow engines.
  - `app.core.metrics.ragas_wrap.eval_ragas` for optional RAGAS.
- Flows depend on:
  - `app.core.retrievers.*` (dense FAISS, BM25, hybrid RRF).
  - `app.core.retrievers.rerank` when rerank toggle is enabled.
  - `app.core.generator` for answer synthesis and usage.
  - `app.core.embeddings` for embedding queries and/or passages.
  - `app.core.prompting.builder` and `app.core.memory.*` for prompt memory text.
- Storage is foundational:
  - Used by UI to persist settings, messages, runs.
  - Used by flows/generator indirectly through bundles/runs.
- Embeddings underpin:
  - Indexer (build/re-embed), dense retriever, and some flows.
  - Caching reduces repeated API/compute costs.

### Configuration and Environment

- **Config files**: `app/config/` (read by `load_config()`).
  - Includes defaults and flow parameters (YAML).
  - RAPTOR config: `app/config/flows/raptor.yaml`.
  - Environment variables override via OmegaConf merge.
- **Environment variables**:
  - `OPENAI_API_KEY` enables online OpenAI embeddings/generation and RAGAS.
  - `OPENAI_EMBED_RPM_CAP` caps embedding request rate (default 90 RPM). Lower if you observe 429s.
  - `OPENAI_EMBED_MAX_ITEMS` controls micro-batch size (default 256). Lower if you hit TPM or large input errors.
  - `OPENAI_EMBED_TPM_CAP` optional tokens-per-minute cap enabling token-aware batching and a global TPM token-bucket.
- **Offline mode**:
  - `cfg['models']['offline'] = True`:
    - Embeddings via `SentenceTransformer`.
    - Generator is a stub; RAGAS skipped.

### Persistence Model (SQLite)

- **Tables** (inferred from usage in `app/core/storage.py`):
  - Messages: `session_id`, `role`, `content`, timestamps.
  - Runs: `id`, `session_id`, `flow`, `question`, `answer`, `citations` (JSON), `timings` (JSON), `metrics` (JSON), `cost`, timestamps.
  - Settings: key-value (dotted-path keys like `models.generation`).
- **Init**: `init_db()` ensures schema-on-connect.

### Metrics

- **Lite metrics** (`app/core/metrics/raglite.py`):
  - Computes basic relevance/precision/groundedness and delta precision.
- **RAGAS** (`app/core/metrics/ragas_wrap.py`):
  - If enabled in settings and deps/API key present, returns `ragas_answer_relevancy`, `ragas_faithfulness`, `ragas_context_precision`. Otherwise returns `{}`.

### Performance and Caching

- **Embeddings cache**:
  - On-disk `.npy` per `(model, text_sha)` in `CACHE_DIR` managed by `app/core/embeddings.py`.
  - In-memory `SentenceTransformer` model cache (`_ST_MODELS`).
- **Index build**:
  - FAISS index persists to disk; rebuild only when needed (UI controls).
- **Reranking**:
  - Enable selectively; tune `rerank_top_n`.

### CLI Tools

 - **Ingestion**: `rag-advanced-ingest` → `app.scripts.ingest:app`
  - Point to directories/files (e.g., `--paths data/samples`) to batch-ingest.
 - **Offline evaluation**: `rag-advanced-eval-offline` → `app.scripts.eval_offline:app`
  - Evaluate flows/params without OpenAI.

### Development and Testing

- **Scripts**: `scripts/dev_check.sh` (macOS/Linux), `scripts/dev_check.ps1` (Windows) run tests, ruff, mypy.
- **Tests** in `tests/` cover:
  - Embeddings caching and adapters.
  - Indexer ingest/build logic.
  - Flows (standard, hybrid, HyDE, multi-hop).
  - Memory (window/summary), prompt builder, metrics (lite), RAGAS toggling.
  - Storage read/write behavior.
- **Coverage**: recent runs show ~84% total coverage (use `pytest --cov=app --cov-report=term-missing`).

### Testing Notes (HyDE patch points)

- To mock retrieval in HyDE, you can patch either `app.core.flows.hyde.retrieve_dense` or the underlying `app.core.retrievers.dense.retrieve` — the HyDE wrappers delegate correctly.
- To mock generation/streaming, patch either `app.core.flows.hyde.generate_answer` / `app.core.flows.hyde.stream_answer` or the underlying `app.core.generator.generate_answer` / `app.core.generator.stream_answer`.

## License

MIT
