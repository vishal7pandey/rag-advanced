from __future__ import annotations
import hashlib
import json
import uuid
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timezone

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from app.core.splitter import split_text, SplitConfig
from app.core.embeddings import get_default_embedder
from app.core.storage import connect
from app.core.types import Chunk
from app.core.utils import DATA_ROOT, load_config, get_logger
import time

INDEX_DIR = DATA_ROOT / "indices"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
EMB_CACHE_DIR = DATA_ROOT / "emb_cache"
EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Central logger
logger = get_logger()


def _safe_text(s: str) -> str:
    """Return a UTF-8 safe string by replacing invalid surrogates.
    This avoids SQLite utf-8 encode errors when text contains unpaired surrogates
    (occasionally produced by PDF extraction).
    """
    try:
        # Fast path: if it encodes cleanly, return as-is
        s.encode("utf-8")
        return s
    except Exception:
        return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _sha_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # lazy import
        except Exception as e:
            raise RuntimeError(
                "Reading PDFs requires 'pypdf' to be installed. pip install pypdf"
            ) from e
        reader = PdfReader(str(path))
        txt = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return _safe_text(txt)
    return path.read_text(encoding="utf-8", errors="ignore")


def _semantic_partition(path: Path) -> Optional[List[tuple[str, str]]]:
    """Try to partition document into semantic elements using unstructured.
    Returns list of (text, element_type) or None if unavailable/fails.
    """
    try:
        # Lazy import; do not make this a hard dependency
        from unstructured.partition.auto import partition  # type: ignore
    except Exception:
        return None
    try:
        elements = partition(filename=str(path))
        out: List[tuple[str, str]] = []
        for el in elements:
            # Elements typically have .text and .category or .type
            text = getattr(el, "text", None) or ""
            if not isinstance(text, str) or not text.strip():
                continue
            etype = getattr(el, "category", None) or getattr(el, "type", None) or "Element"
            out.append((text, str(etype)))
        return out
    except Exception:
        return None


def _prepare_doc_for_ingest(
    path_str: str,
    chunk_size: int,
    overlap: int,
    semantic_enabled: bool,
    existing_sha_by_path: dict[str, str],
) -> dict:
    """Worker-side preparation for a single document.
    Returns a dict with keys:
      - path: str
      - sha: str
      - unchanged: bool
      - meta_base: dict
      - parts_with_type: Optional[List[Tuple[str, str]]] when changed
    """
    p = Path(path_str)
    if not p.exists() or not p.is_file():
        return {"path": path_str, "skip": True, "reason": "missing"}
    sha = _sha_file(p)
    # Meta base: path + mtime (ISO UTC)
    try:
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        mtime = datetime.now(tz=timezone.utc).isoformat()
    meta_base = {"path": path_str, "mtime": mtime}
    # If unchanged, avoid heavy parsing and splitting
    if existing_sha_by_path.get(path_str) == sha:
        return {"path": path_str, "sha": sha, "unchanged": True, "meta_base": meta_base}
    # Changed: perform parsing/partitioning/splitting
    if semantic_enabled:
        elems = _semantic_partition(p)
        if elems:
            parts_with_type: List[tuple[str, str]] = elems
        else:
            text = _read_file(p)
            parts_plain = split_text(text, SplitConfig(chunk_size=chunk_size, overlap=overlap))
            parts_with_type = [(t, "Text") for t in parts_plain]
    else:
        text = _read_file(p)
        parts_plain = split_text(text, SplitConfig(chunk_size=chunk_size, overlap=overlap))
        parts_with_type = [(t, "Text") for t in parts_plain]
    return {
        "path": path_str,
        "sha": sha,
        "unchanged": False,
        "meta_base": meta_base,
        "parts_with_type": parts_with_type,
    }


def ingest_paths(
    paths: List[Path], chunk_size: int = 800, overlap: int = 120
) -> tuple[List[Chunk], dict]:
    conn = connect()
    chunks: List[Chunk] = []
    stats = {
        "docs_added": 0,
        "docs_updated": 0,
        "docs_unchanged": 0,
        "chunks_added": 0,
        "chunks_updated": 0,
        "chunks_skipped": 0,
    }
    # Read semantic chunking config (optional)
    semantic_enabled = False
    try:
        cfg = load_config()
        semantic_enabled = bool(cfg.get("index", {}).get("semantic_chunking_enabled", False))
    except Exception:
        pass
    for p in paths:
        if not p.exists() or not p.is_file():
            continue
        sha = _sha_file(p)
        cur = conn.execute("SELECT id, sha FROM documents WHERE path=?", (str(p),))
        row = cur.fetchone()
        if row and row[1] == sha:
            # unchanged; skip re-chunk
            cur = conn.execute(
                "SELECT id, doc_id, ord, text, meta_json FROM chunks WHERE doc_id=? ORDER BY ord",
                (row[0],),
            )
            for cid, did, ordv, text, meta_json in cur.fetchall():
                chunks.append(
                    Chunk(
                        id=cid, doc_id=did, ord=ordv, text=text, meta=json.loads(meta_json or "{}")
                    )
                )
                stats["chunks_skipped"] += 1
            stats["docs_unchanged"] += 1
            continue
        doc_id = row[0] if row else str(uuid.uuid4())
        # Choose chunking strategy
        # Base metadata: path + file modification time (ISO UTC)
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
        except Exception:
            mtime = datetime.now(tz=timezone.utc).isoformat()
        meta_base = {"path": str(p), "mtime": mtime}
        if semantic_enabled:
            elems = _semantic_partition(p)
            if elems:
                parts_with_type: List[tuple[str, str]] = elems
                parts_plain: List[str] = [t for (t, _) in parts_with_type]
            else:
                # Fallback to plain read/split
                text = _read_file(p)
                parts_plain = split_text(text, SplitConfig(chunk_size=chunk_size, overlap=overlap))
                parts_with_type = [(t, "Text") for t in parts_plain]
        else:
            text = _read_file(p)
            parts_plain = split_text(text, SplitConfig(chunk_size=chunk_size, overlap=overlap))
            parts_with_type = [(t, "Text") for t in parts_plain]
        # Begin a short, per-document write transaction to minimize lock duration
        conn.execute("BEGIN IMMEDIATE")
        try:
            if row:
                conn.execute(
                    "UPDATE documents SET sha=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (sha, doc_id),
                )
                conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
                stats["docs_updated"] += 1
            else:
                conn.execute(
                    "INSERT INTO documents(id, path, sha) VALUES (?, ?, ?)", (doc_id, str(p), sha)
                )
                stats["docs_added"] += 1
            for i, (part, etype) in enumerate(parts_with_type):
                cid = str(uuid.uuid4())
                meta = dict(meta_base)
                # Include semantic element type when available
                if etype:
                    meta["element_type"] = etype
                conn.execute(
                    "INSERT INTO chunks(id, doc_id, ord, text, meta_json) VALUES (?, ?, ?, ?, ?)",
                    (cid, doc_id, i, part, json.dumps(meta)),
                )
                chunks.append(Chunk(id=cid, doc_id=doc_id, ord=i, text=part, meta=meta))
                if row:
                    stats["chunks_updated"] += 1
                else:
                    stats["chunks_added"] += 1
            # Commit the per-document transaction promptly to release any write locks
            conn.commit()
        except Exception:
            # Rollback to avoid leaving the DB in a locked state
            conn.rollback()
            raise
    conn.commit()
    conn.close()
    return chunks, stats


def ingest_paths_parallel(
    paths: List[Path],
    chunk_size: int = 800,
    overlap: int = 120,
    workers: int = 4,
) -> tuple[List[Chunk], dict]:
    """Parallelize CPU-heavy preprocessing with processes, then apply DB writes with a single writer.
    Returns (chunks, stats) like ingest_paths().
    """
    # Normalize and filter input paths
    todo_paths: List[Path] = [p for p in paths if p.exists() and p.is_file()]
    if not todo_paths:
        return [], {
            "docs_added": 0,
            "docs_updated": 0,
            "docs_unchanged": 0,
            "chunks_added": 0,
            "chunks_updated": 0,
            "chunks_skipped": 0,
        }
    # Config: semantic chunking
    semantic_enabled = False
    try:
        cfg = load_config()
        semantic_enabled = bool(cfg.get("index", {}).get("semantic_chunking_enabled", False))
    except Exception:
        pass
    # Preload existing mapping to avoid per-doc reads and help workers skip unchanged parsing
    conn = connect()
    try:
        cur = conn.execute("SELECT path, id, sha FROM documents")
        rows = cur.fetchall()
    finally:
        conn.close()
    existing_id_by_path: dict[str, str] = {r[0]: r[1] for r in rows}
    existing_sha_by_path: dict[str, str] = {r[0]: r[2] for r in rows}
    # Stage 1: preprocess in processes
    results: List[dict] = []
    path_strs = [str(p) for p in todo_paths]
    with ProcessPoolExecutor(max_workers=int(max(1, workers))) as ex:
        futs = [
            ex.submit(
                _prepare_doc_for_ingest,
                path_str,
                int(chunk_size),
                int(overlap),
                bool(semantic_enabled),
                existing_sha_by_path,
            )
            for path_str in path_strs
        ]
        for f in as_completed(futs):
            results.append(f.result())
    # Stage 2: single-writer apply
    conn = connect()
    chunks: List[Chunk] = []
    stats = {
        "docs_added": 0,
        "docs_updated": 0,
        "docs_unchanged": 0,
        "chunks_added": 0,
        "chunks_updated": 0,
        "chunks_skipped": 0,
    }
    for item in results:
        path_str = item.get("path")
        if not path_str:
            continue
        if item.get("skip"):
            # Missing or invalid file
            continue
        # Existing doc info if any
        has_existing = path_str in existing_id_by_path
        doc_id = existing_id_by_path.get(path_str, str(uuid.uuid4()))
        if bool(item.get("unchanged")):
            # Load existing chunks to return, mark stats
            if has_existing:
                cur = conn.execute(
                    "SELECT id, doc_id, ord, text, meta_json FROM chunks WHERE doc_id=? ORDER BY ord",
                    (doc_id,),
                )
                rows = cur.fetchall()
                for cid, did, ordv, text, meta_json in rows:
                    chunks.append(
                        Chunk(
                            id=cid,
                            doc_id=did,
                            ord=ordv,
                            text=text,
                            meta=json.loads(meta_json or "{}"),
                        )
                    )
                    stats["chunks_skipped"] += 1
            stats["docs_unchanged"] += 1
            continue
        # Changed doc: write updates
        sha = str(item.get("sha"))
        parts_with_type: List[tuple[str, str]] = item.get("parts_with_type", [])  # type: ignore
        meta_base = dict(item.get("meta_base", {}))
        conn.execute("BEGIN IMMEDIATE")
        try:
            if has_existing:
                conn.execute(
                    "UPDATE documents SET sha=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (sha, doc_id),
                )
                conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
                stats["docs_updated"] += 1
            else:
                conn.execute(
                    "INSERT INTO documents(id, path, sha) VALUES (?, ?, ?)", (doc_id, path_str, sha)
                )
                stats["docs_added"] += 1
                # update cache map so later duplicates (if any) treat as existing
                existing_id_by_path[path_str] = doc_id
                existing_sha_by_path[path_str] = sha
            for i, (part, etype) in enumerate(parts_with_type):
                cid = str(uuid.uuid4())
                meta = dict(meta_base)
                if etype:
                    meta["element_type"] = etype
                # Sanitize text to avoid utf-8 surrogate encode errors in SQLite
                _part = _safe_text(part)
                conn.execute(
                    "INSERT INTO chunks(id, doc_id, ord, text, meta_json) VALUES (?, ?, ?, ?, ?)",
                    (cid, doc_id, i, _part, json.dumps(meta)),
                )
                chunks.append(Chunk(id=cid, doc_id=doc_id, ord=i, text=_part, meta=meta))
                if has_existing:
                    stats["chunks_updated"] += 1
                else:
                    stats["chunks_added"] += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    conn.commit()
    conn.close()
    return chunks, stats


def _text_sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _model_tag(offline: bool, emb_model_st: str | None, emb_model_oa: str | None) -> str:
    return (emb_model_oa if not offline else emb_model_st) or ("st" if offline else "openai")


def _cache_dir_for_model(offline: bool, emb_model_st: str | None, emb_model_oa: str | None) -> Path:
    tag = _model_tag(offline, emb_model_st, emb_model_oa)
    d = EMB_CACHE_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def reembed_changed_only(offline: bool, emb_model_st: str | None, emb_model_oa: str | None) -> dict:
    """Embed only chunks missing from cache. Returns counts dict."""
    t0 = time.monotonic()
    model_tag = _model_tag(offline, emb_model_st, emb_model_oa)
    try:
        logger.info(
            "index.reembed_start",
            offline=bool(offline),
            model_tag=model_tag,
        )
    except Exception:
        pass
    conn = connect()
    try:
        cur = conn.execute("SELECT id, text FROM chunks ORDER BY doc_id, ord")
        rows = cur.fetchall()
    finally:
        conn.close()
    cache_dir = _cache_dir_for_model(offline, emb_model_st, emb_model_oa)
    embedder = get_default_embedder(offline=offline, st_model=emb_model_st, oa_model=emb_model_oa)
    # Determine which shas are missing in cache
    shas: list[str] = []
    texts_by_sha: dict[str, str] = {}
    missing: list[str] = []
    for _, text in rows:
        s = _text_sha(text)
        shas.append(s)
        if s not in texts_by_sha:
            texts_by_sha[s] = text
        if not (cache_dir / f"{s}.npy").exists():
            if s not in missing:
                missing.append(s)
    added = 0
    if missing:
        # Let embedder handle token-aware batching internally
        missing_texts = [texts_by_sha[s] for s in missing]
        X = embedder(missing_texts)
        for i, s in enumerate(missing):
            np.save(cache_dir / f"{s}.npy", X[i])
        added = len(missing)
    out = {"total_chunks": len(rows), "new_embeddings": added, "skipped": len(rows) - added}
    t1 = time.monotonic()
    try:
        logger.info(
            "index.reembed_end",
            offline=bool(offline),
            model_tag=model_tag,
            total_chunks=int(len(rows)),
            missing=int(len(missing)),
            new_embeddings=int(added),
            duration_ms=int((t1 - t0) * 1000),
        )
    except Exception:
        pass
    return out


def build_faiss(
    offline: bool, emb_model_st: str | None, emb_model_oa: str | None
) -> Tuple[Path, int]:
    # Try FAISS; if unavailable (e.g., Python 3.13), fall back to NumPy-based index
    t0 = time.monotonic()
    model_tag = _model_tag(offline, emb_model_st, emb_model_oa)
    try:
        logger.info(
            "index.build_start",
            offline=bool(offline),
            model_tag=model_tag,
        )
    except Exception:
        pass
    try:
        import faiss  # type: ignore

        have_faiss = True
    except Exception:
        faiss = None  # type: ignore
        have_faiss = False
    conn = connect()
    cur = conn.execute("SELECT id, text FROM chunks ORDER BY doc_id, ord")
    rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No chunks found. Ingest documents first.")
    # Prepare embeddings using cache; compute missing
    cache_dir = _cache_dir_for_model(offline, emb_model_st, emb_model_oa)
    embedder = get_default_embedder(offline=offline, st_model=emb_model_st, oa_model=emb_model_oa)
    shas = [_text_sha(r[1]) for r in rows]
    texts_by_sha = {s: t for s, t in zip(shas, [r[1] for r in rows])}
    # dedupe order for computing missing
    unique_order: list[str] = []
    seen: set[str] = set()
    for s in shas:
        if s not in seen:
            unique_order.append(s)
            seen.add(s)
    missing = [s for s in unique_order if not (cache_dir / f"{s}.npy").exists()]
    if missing:
        # Let embedder handle token-aware batching internally
        missing_texts = [texts_by_sha[s] for s in missing]
        Xb = embedder(missing_texts)
        for i, s in enumerate(missing):
            np.save(cache_dir / f"{s}.npy", Xb[i])
    # Load embeddings in chunk order
    embs = [np.load(cache_dir / f"{s}.npy") for s in shas]
    X = np.vstack(embs)
    dim = X.shape[1]
    model_tag = _model_tag(offline, emb_model_st, emb_model_oa)
    if have_faiss:
        index = faiss.IndexFlatIP(dim)  # type: ignore
        # normalize for cosine similarity
        faiss.normalize_L2(X)  # type: ignore
        index.add(X)
        # persist FAISS
        out = INDEX_DIR / f"faiss_{model_tag}.faiss"
        faiss.write_index(index, str(out))  # type: ignore
        # record index metadata
        conn.execute(
            "INSERT OR REPLACE INTO indices(index_name, faiss_path, dim) VALUES (?, ?, ?)",
            (f"faiss_{model_tag}", str(out), int(dim)),
        )
    else:
        # NumPy fallback: store L2-normalized matrix for cosine search
        # Normalize (avoid divide-by-zero)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        out = INDEX_DIR / f"np_{model_tag}.npz"
        np.savez_compressed(out, X=Xn.astype(np.float32))
        # record pseudo-index metadata; we still use the same table/columns
        conn.execute(
            "INSERT OR REPLACE INTO indices(index_name, faiss_path, dim) VALUES (?, ?, ?)",
            (f"np_{model_tag}", str(out), int(dim)),
        )
    conn.commit()
    conn.close()
    t1 = time.monotonic()
    try:
        logger.info(
            "index.build_end",
            offline=bool(offline),
            model_tag=model_tag,
            total_chunks=int(len(rows)),
            missing=int(len(missing)),
            dim=int(dim),
            have_faiss=bool(have_faiss),
            out=str(out),
            duration_ms=int((t1 - t0) * 1000),
        )
    except Exception:
        pass
    return out, dim


def purge_index(offline: bool, emb_model_st: str | None, emb_model_oa: str | None) -> dict:
    """Drop FAISS index files for current model, clear embedding cache for that model, and remove indices metadata."""
    tag = _model_tag(offline, emb_model_st, emb_model_oa)
    # Remove FAISS/NumPy index file(s) for tag
    faiss_path = INDEX_DIR / f"faiss_{tag}.faiss"
    np_path = INDEX_DIR / f"np_{tag}.npz"
    removed_index = False
    # Try remove FAISS index
    if faiss_path.exists():
        try:
            faiss_path.unlink()
            removed_index = True
        except Exception:
            pass
    # Try remove NumPy fallback index
    if np_path.exists():
        try:
            np_path.unlink()
            removed_index = True
        except Exception:
            pass
    # Remove cache dir for tag
    cache_dir = EMB_CACHE_DIR / tag
    removed_cache_files = 0
    if cache_dir.exists():
        for f in cache_dir.glob("*.npy"):
            try:
                f.unlink()
                removed_cache_files += 1
            except Exception:
                pass
        try:
            cache_dir.rmdir()
        except Exception:
            pass
    # Clean indices metadata (both FAISS and NumPy tags)
    conn = connect()
    try:
        conn.execute(
            "DELETE FROM indices WHERE index_name IN (?, ?)",
            (f"faiss_{tag}", f"np_{tag}"),
        )
        conn.commit()
    finally:
        conn.close()
    return {"removed_index": removed_index, "removed_cache_files": removed_cache_files, "tag": tag}


def full_purge(
    *, purge_uploads: bool = False, purge_runs: bool = False, purge_messages: bool = False
) -> dict:
    """Delete all retrieval artifacts and optionally uploads/runs/messages.

    - Deletes: all FAISS index files, all embedding cache files, and all rows in
      documents/chunks/indices tables.
    - Optionally deletes: uploads files, runs table rows, messages+memory rows.
    Returns counts summary.
    """
    # File system deletions
    removed_faiss = 0
    for f in INDEX_DIR.glob("faiss_*.faiss"):
        try:
            f.unlink()
            removed_faiss += 1
        except Exception:
            pass
    removed_cache_files = 0
    if EMB_CACHE_DIR.exists():
        for p in EMB_CACHE_DIR.rglob("*.npy"):
            try:
                p.unlink()
                removed_cache_files += 1
            except Exception:
                pass
        # clean empty tag dirs
        for d in sorted(
            {p.parent for p in EMB_CACHE_DIR.rglob("*.npy")},
            key=lambda x: len(str(x)),
            reverse=True,
        ):
            try:
                if d.exists() and not any(d.iterdir()):
                    d.rmdir()
            except Exception:
                pass
    removed_uploads = 0
    uploads_dir = DATA_ROOT / "uploads"
    if purge_uploads and uploads_dir.exists():
        for f in uploads_dir.glob("*"):
            try:
                if f.is_file():
                    f.unlink()
                    removed_uploads += 1
                elif f.is_dir():
                    removed_uploads += sum(1 for _ in f.rglob("*"))
                    shutil.rmtree(f, ignore_errors=True)
            except Exception:
                pass
    # Database deletions
    conn = connect()
    try:
        # capture counts before delete
        def _count(tbl: str) -> int:
            try:
                cur = conn.execute(f"SELECT COUNT(1) FROM {tbl}")
                return int(cur.fetchone()[0])
            except Exception:
                return 0

        counts_before = {
            "documents": _count("documents"),
            "chunks": _count("chunks"),
            "indices": _count("indices"),
            "runs": _count("runs") if purge_runs else None,
            "messages": _count("messages") if purge_messages else None,
            "memory": _count("memory") if purge_messages else None,
        }
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM indices")
        if purge_runs:
            conn.execute("DELETE FROM runs")
        if purge_messages:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM memory")
        conn.commit()
    finally:
        conn.close()
    return {
        "removed_faiss": removed_faiss,
        "removed_cache_files": removed_cache_files,
        "removed_uploads": removed_uploads,
        "db_counts_before": counts_before,
    }
