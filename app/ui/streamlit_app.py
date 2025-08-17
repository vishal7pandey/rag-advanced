from __future__ import annotations
import streamlit as st
from dotenv import load_dotenv

from app.core.utils import load_config, get_logger
from app.core.storage import init_db
from app.core.logging_setup import log_context_bind

from pathlib import Path
import uuid

# Ensure .env is loaded from the project root regardless of CWD
# Project root is two levels up from this file: app/ui/ -> project/
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
logger = get_logger()

st.set_page_config(page_title="RAG Advanced", layout="wide")

if "cfg" not in st.session_state:
    st.session_state.cfg = load_config()

# Bind a per-session logging context so all logs include session_id
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
log_context_bind(app="rag-advanced", session_id=st.session_state.session_id)

init_db()

try:
    logger.info(
        "app.ui_start",
        session_id=st.session_state.session_id,
        offline=bool(st.session_state.cfg["models"]["offline"]),
    )
except Exception:
    pass

st.title("RAG Advanced (Local)")
st.write(
    "Use the left sidebar to navigate pages: Upload & Index, Chat RAG, Runs & Metrics, Admin Settings."
)

with st.sidebar:
    st.markdown("### Status")
    st.write(f"Offline mode: {st.session_state.cfg['models']['offline']}")
    st.write("DB initialized ✔️")
