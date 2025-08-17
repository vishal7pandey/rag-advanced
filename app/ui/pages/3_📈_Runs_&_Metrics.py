from __future__ import annotations
import streamlit as st
import pandas as pd
from app.core.storage import get_recent_runs
from app.core.utils import pretty_json

st.title("📈 Runs & Metrics")

runs = get_recent_runs(limit=100)
if not runs:
    st.info("No runs yet. Ask a question in Chat to create runs.")
else:
    rows = []
    for r in runs:
        m = r.get("metrics", {}) or {}
        rows.append(
            {
                "id": r.get("id"),
                "flow": r.get("flow"),
                "answer_relevancy_lite": m.get("answer_relevancy_lite"),
                "context_precision_lite": m.get("context_precision_lite"),
                "groundedness_lite": m.get("groundedness_lite"),
                "delta_precision_lite": m.get("delta_precision_lite"),
                "ragas_answer_relevancy": m.get("ragas_answer_relevancy"),
                "ragas_faithfulness": m.get("ragas_faithfulness"),
                "ragas_context_precision": m.get("ragas_context_precision"),
                "cost": r.get("cost"),
                "created_at": r.get("created_at"),
                "question": r.get("question"),
            }
        )
    df = pd.DataFrame(rows)
    # Quick filters: flow/time window
    fcol1, fcol2, fcol3 = st.columns([2, 2, 1])
    with fcol1:
        all_flows = (
            sorted([f for f in df.get("flow", []).dropna().unique().tolist()])
            if not df.empty
            else []
        )
        selected_flows = st.multiselect("Flow", options=all_flows, default=all_flows)
    with fcol2:
        time_window = st.selectbox("Time window", options=["All", "24h", "7d", "30d"], index=0)
    with fcol3:
        st.write("")
        st.write("")
        st.caption("Filters apply to table and debugger")

    df_f = df.copy()
    if selected_flows:
        df_f = df_f[df_f["flow"].isin(selected_flows)]
    # Parse timestamps and apply time filter
    if not df_f.empty:
        df_f["_dt"] = pd.to_datetime(df_f["created_at"], errors="coerce")
        if time_window != "All":
            now = pd.Timestamp.utcnow()
            if time_window == "24h":
                cutoff = now - pd.Timedelta(hours=24)
            elif time_window == "7d":
                cutoff = now - pd.Timedelta(days=7)
            else:  # 30d
                cutoff = now - pd.Timedelta(days=30)
            df_f = df_f[df_f["_dt"] >= cutoff]

    st.dataframe(df_f.drop(columns=["_dt"], errors="ignore"), use_container_width=True)
    st.download_button(
        "Export CSV",
        data=df_f.drop(columns=["_dt"], errors="ignore").to_csv(index=False),
        file_name="runs.csv",
    )

    st.divider()
    st.subheader("Run Debugger")
    st.caption(
        "Expand a run to inspect plan, retrieval, rerank, memory, prompts, timings, and costs. Rendering adapts to available data."
    )

    # Render per-run debuggers (filtered)
    id_keep: set = set(df_f["id"].tolist()) if not df_f.empty else set()
    for r in [r for r in runs if (not id_keep or r.get("id") in id_keep)]:
        rid = r.get("id")
        created = r.get("created_at")
        flow = r.get("flow") or "auto"
        m = r.get("metrics", {}) or {}
        t = r.get("timings", {}) or {}
        cits = r.get("citations", []) or []
        q = r.get("question", "")
        ans = r.get("answer", "")
        cost = r.get("cost", 0.0) or 0.0

        with st.expander(f"Run {rid} · {created} · flow={flow}"):
            cols = st.columns([2, 1])
            with cols[0]:
                st.markdown("**Question**")
                st.write(q)
                st.markdown("**Answer**")
                st.markdown(ans)
            with cols[1]:
                st.markdown("**Summary**")
                st.write(
                    {
                        "flow": flow,
                        "created_at": created,
                        "cost": float(cost),
                    }
                )
                if any(
                    k in m
                    for k in (
                        "answer_relevancy_lite",
                        "context_precision_lite",
                        "groundedness_lite",
                    )
                ):
                    st.markdown("**Lite Metrics**")
                    st.write(
                        {
                            "answer_relevancy_lite": m.get("answer_relevancy_lite"),
                            "context_precision_lite": m.get("context_precision_lite"),
                            "groundedness_lite": m.get("groundedness_lite"),
                            "delta_precision_lite": m.get("delta_precision_lite"),
                        }
                    )
                if any(
                    k in m
                    for k in (
                        "ragas_answer_relevancy",
                        "ragas_faithfulness",
                        "ragas_context_precision",
                    )
                ):
                    st.markdown("**RAGAS**")
                    st.write(
                        {
                            "answer_relevancy": m.get("ragas_answer_relevancy"),
                            "faithfulness": m.get("ragas_faithfulness"),
                            "context_precision": m.get("ragas_context_precision"),
                        }
                    )

            # Plan / Orchestrator rationale (if present)
            plan_dict = m.get("plan")
            with st.expander("Selected plan & rationale", expanded=bool(plan_dict)):
                if plan_dict:
                    st.code(pretty_json(plan_dict), language="json")
                else:
                    st.caption("Plan details unavailable in this run.")

            # Retrieval and rerank debug
            with st.expander("Retrieval & Rerank Debug"):
                # Expect optional shapes in metrics if provided by orchestrator/retrievers
                retrieved = m.get("retrieved") or []  # list of {path, ord, score, source}
                reranked = m.get("reranked") or []  # list of {path, ord, score}
                deltas = m.get("rerank_deltas") or []  # list of {path, delta}
                if retrieved:
                    st.markdown("**Candidates (pre-rerank)**")
                    for i, it in enumerate(retrieved, start=1):
                        try:
                            st.write(
                                {
                                    "rank": i,
                                    "path": it.get("path"),
                                    "ord": it.get("ord"),
                                    "score": it.get("score"),
                                    "source": it.get("source"),
                                }
                            )
                        except Exception:
                            st.write(it)
                else:
                    st.caption("No retrieval candidate list recorded for this run.")
                if reranked:
                    st.markdown("**Reranked**")
                    for i, it in enumerate(reranked, start=1):
                        try:
                            st.write(
                                {
                                    "rank": i,
                                    "path": it.get("path"),
                                    "ord": it.get("ord"),
                                    "score": it.get("score"),
                                }
                            )
                        except Exception:
                            st.write(it)
                if deltas:
                    st.markdown("**Rerank Δscores**")
                    st.write(deltas)

            # Memory & prompt snapshot (if present)
            with st.expander("Memory & Prompt Snapshot"):
                mem = m.get("memory") or {}
                prompt = m.get("prompt") or {}
                if mem:
                    st.markdown("**Memory**")
                    st.code(pretty_json(mem), language="json")
                else:
                    st.caption("No memory snapshot recorded for this run.")
                if prompt:
                    st.markdown("**Prompt**")
                    st.code(pretty_json(prompt), language="json")
                else:
                    st.caption("No prompt snapshot recorded for this run.")

            # Timings and sources
            with st.expander("Timings & Sources", expanded=False):
                if t:
                    st.markdown("**Timings**")
                    st.code(pretty_json(t), language="json")
                else:
                    st.caption("No timings recorded.")
                if cits:
                    st.markdown("**Sources**")
                    for i, c in enumerate(cits, start=1):
                        try:
                            st.write(
                                {
                                    "rank": i,
                                    "path": c.get("path"),
                                    "ord": c.get("ord"),
                                    "marker": c.get("marker"),
                                }
                            )
                        except Exception:
                            st.write(c)
                else:
                    st.caption("No citations recorded.")
