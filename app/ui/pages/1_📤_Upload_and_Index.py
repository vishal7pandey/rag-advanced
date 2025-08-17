from __future__ import annotations
import streamlit as st

import platform
from pathlib import Path
from app.core.utils import load_config, DATA_ROOT
from app.core.indexer import (
    ingest_paths,
    ingest_paths_parallel,
    build_faiss,
    reembed_changed_only,
    purge_index,
    full_purge,
)

st.title("📤 Upload & Index")

cfg = load_config()

uploaded = st.file_uploader(
    "Upload PDF / TXT / MD", accept_multiple_files=True, type=["pdf", "txt", "md"]
)

# Parallel ingestion control (config-driven default; Windows caution still applies)
is_windows = platform.system() == "Windows"
try:
    default_workers_cfg = int(cfg.get("ingest", {}).get("default_workers", 4))
except Exception:
    default_workers_cfg = 4
default_workers = max(1, min(8, default_workers_cfg))
workers = st.slider(
    "Parallel ingest workers",
    min_value=1,
    max_value=8,
    value=default_workers,
    help=(
        "Use >1 to speed up parsing/splitting; a single writer ensures SQLite safety. "
        "On Windows, process pools can be less reliable in Streamlit—if you encounter issues, reduce workers to 1."
    ),
)
if is_windows and workers > 1:
    st.info(
        "Windows detected: parallel ingestion is experimental in Streamlit and may hang or use more memory. "
        "If you encounter issues, reduce workers to 1.",
    )

col1, col2, col3 = st.columns(3)
with col1:
    build = st.button("Build/Update Index")
with col2:
    purge = st.button("Purge Index & Cache")
with col3:
    rembed = st.button("Re-embed Changed Only")

uploads_dir = DATA_ROOT / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

if uploaded:
    # Save uploaded files first
    saved_paths: list[Path] = []
    for f in uploaded:
        out = uploads_dir / f.name
        out.write_bytes(f.getvalue())
        saved_paths.append(out)

    if workers > 1:
        # Parallel ingest across files
        with st.spinner(f"Parallel chunking & storing with {workers} workers…"):
            all_chunks, total = ingest_paths_parallel(
                saved_paths,
                chunk_size=cfg["index"]["chunk_size"],
                overlap=cfg["index"]["overlap"],
                workers=workers,
            )
        total_files = len(saved_paths)
        st.success(
            f"Stored {len(all_chunks)} chunks from {total_files} files. "
            f"docs_added={total['docs_added']} docs_updated={total['docs_updated']} docs_unchanged={total['docs_unchanged']} "
            f"chunks_added={total['chunks_added']} chunks_updated={total['chunks_updated']} chunks_skipped={total['chunks_skipped']}"
        )
    else:
        # Per-file ingest with progress (serial)
        st.info("Chunking & storing uploaded files…")
        prog = st.progress(0)
        status = st.empty()
        total_files = len(saved_paths)
        all_chunks = []
        total = {
            "docs_added": 0,
            "docs_updated": 0,
            "docs_unchanged": 0,
            "chunks_added": 0,
            "chunks_updated": 0,
            "chunks_skipped": 0,
        }
        for i, p in enumerate(saved_paths, start=1):
            status.markdown(f"Processing file {i}/{total_files}: `{p.name}` …")
            try:
                ch, st_stats = ingest_paths(
                    [p], chunk_size=cfg["index"]["chunk_size"], overlap=cfg["index"]["overlap"]
                )
                all_chunks.extend(ch)
                for k in total.keys():
                    total[k] += int(st_stats.get(k, 0))
            except Exception as e:
                st.error(f"Failed to ingest {p.name}: {e}")
            prog.progress(int(i * 100 / total_files))
        status.empty()
        st.success(
            f"Stored {len(all_chunks)} chunks from {total_files} files. "
            f"docs_added={total['docs_added']} docs_updated={total['docs_updated']} docs_unchanged={total['docs_unchanged']} "
            f"chunks_added={total['chunks_added']} chunks_updated={total['chunks_updated']} chunks_skipped={total['chunks_skipped']}"
        )

if build:
    with st.spinner("Building FAISS index…"):
        offline = bool(cfg["models"]["offline"])
        faiss_path, dim = build_faiss(
            offline=offline,
            emb_model_st=cfg["models"].get("embedding"),
            emb_model_oa=cfg["models"].get("embedding"),
        )
    st.success(f"Index built at {faiss_path} (dim={dim}).")

if rembed:
    with st.spinner("Embedding changed/missing chunks…"):
        offline = bool(cfg["models"]["offline"])
        res = reembed_changed_only(
            offline=offline,
            emb_model_st=cfg["models"].get("embedding"),
            emb_model_oa=cfg["models"].get("embedding"),
        )
    st.success(
        f"Re-embed done. total_chunks={res['total_chunks']} new_embeddings={res['new_embeddings']} skipped={res['skipped']}"
    )

if purge:
    with st.spinner("Purging index and caches…"):
        offline = bool(cfg["models"]["offline"])
        res = purge_index(
            offline=offline,
            emb_model_st=cfg["models"].get("embedding"),
            emb_model_oa=cfg["models"].get("embedding"),
        )
    st.warning(
        f"Purged tag={res['tag']}. removed_index={res['removed_index']} removed_cache_files={res['removed_cache_files']}"
    )

st.caption(f"Chunk size: {cfg['index']['chunk_size']} overlap: {cfg['index']['overlap']}")

# Danger Zone: Full Purge
st.divider()
st.markdown("#### Danger Zone")
with st.expander("Full Purge (Everything)"):
    st.warning(
        "This will delete all documents, chunks, indices, and embedding cache. "
        "Optionally it can also delete uploaded files, runs/metrics, and chat history/memory."
    )
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        opt_uploads = st.checkbox("Uploads", value=False, help="Delete files under data/uploads/")
    with c2:
        opt_runs = st.checkbox("Runs/Metrics", value=False)
    with c3:
        opt_msgs = st.checkbox("Chat History", value=False, help="messages + memory tables")
    with c4:
        confirm = st.text_input("Type DELETE to confirm", value="")
    do_full = st.button("Full Purge (Everything)", type="primary")
    if do_full:
        if confirm.strip().upper() == "DELETE":
            with st.spinner("Purging everything…"):
                res = full_purge(
                    purge_uploads=opt_uploads, purge_runs=opt_runs, purge_messages=opt_msgs
                )
            st.success(
                f"Purged. removed_faiss={res['removed_faiss']} "
                f"removed_cache_files={res['removed_cache_files']} removed_uploads={res['removed_uploads']} "
                f"db_before={res['db_counts_before']}"
            )
        else:
            st.error("Confirmation failed. Please type DELETE to proceed.")
