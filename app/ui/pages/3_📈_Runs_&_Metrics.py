from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Any, Dict, Optional
from app.core.storage import get_recent_runs, get_counts
from app.core.utils import pretty_json

st.title("📈 Runs & Metrics")

runs = get_recent_runs(limit=200)
if not runs:
    st.info("No runs yet. Ask a question in Chat to create runs.")
else:
    rows = []
    for r in runs:
        m = r.get("metrics", {}) or {}
        t = r.get("timings", {}) or {}
        env: Dict[str, Any] = {}
        try:
            env = m.get("envelope", {}) or {}
        except Exception:
            env = {}
        usage: Dict[str, Any] = env.get("usage", {}) if isinstance(env, dict) else {}
        exit_state: Optional[str] = None
        try:
            exit_state = (env.get("exit", {}) or {}).get("state")
        except Exception:
            exit_state = None
        tr = t.get("t_retrieve") if isinstance(t, dict) else None
        tg = t.get("t_generate") if isinstance(t, dict) else None
        tt: Optional[float]
        try:
            tt = float(tr or 0.0) + float(tg or 0.0)
        except Exception:
            tt = None
        rows.append(
            {
                "id": r.get("id"),
                "flow": r.get("flow"),
                "t_retrieve": tr,
                "t_generate": tg,
                "t_total": tt,
                "answer_relevancy_lite": m.get("answer_relevancy_lite"),
                "context_precision_lite": m.get("context_precision_lite"),
                "groundedness_lite": m.get("groundedness_lite"),
                "delta_precision_lite": m.get("delta_precision_lite"),
                "ragas_answer_relevancy": m.get("ragas_answer_relevancy"),
                "ragas_faithfulness": m.get("ragas_faithfulness"),
                "ragas_context_precision": m.get("ragas_context_precision"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "exit_state": exit_state,
                "cost": r.get("cost"),
                "created_at": r.get("created_at"),
                "question": r.get("question"),
            }
        )
    df = pd.DataFrame(rows)
    # Sidebar thresholds for KPIs
    with st.sidebar:
        st.subheader("Dashboard thresholds")
        thr_faith = st.slider("Faithfulness threshold", 0.0, 1.0, 0.6, 0.05)
        thr_latency = st.number_input("Latency budget (s)", value=5.0, min_value=0.0, step=0.5)
        thr_cost = st.number_input("Cost budget per run ($)", value=0.02, min_value=0.0, step=0.01)
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

    # KPIs
    if not df_f.empty:
        lat_cols = st.columns(4)
        with lat_cols[0]:
            st.metric(
                "Avg total latency (s)",
                f"{pd.to_numeric(df_f['t_total'], errors='coerce').mean():.3f}",
            )
        with lat_cols[1]:
            st.metric(
                "Avg retrieve (s)",
                f"{pd.to_numeric(df_f['t_retrieve'], errors='coerce').mean():.3f}",
            )
        with lat_cols[2]:
            st.metric(
                "Avg generate (s)",
                f"{pd.to_numeric(df_f['t_generate'], errors='coerce').mean():.3f}",
            )
        with lat_cols[3]:
            st.metric("Avg cost ($)", f"{pd.to_numeric(df_f['cost'], errors='coerce').mean():.4f}")

        tok_cols = st.columns(3)
        with tok_cols[0]:
            st.metric(
                "Avg prompt tokens",
                f"{pd.to_numeric(df_f['prompt_tokens'], errors='coerce').mean():.0f}",
            )
        with tok_cols[1]:
            st.metric(
                "Avg completion tokens",
                f"{pd.to_numeric(df_f['completion_tokens'], errors='coerce').mean():.0f}",
            )

        # Rates
        def _faith_val(row):
            rf = row.get("ragas_faithfulness")
            gl = row.get("groundedness_lite")
            return rf if pd.notna(rf) else (gl if pd.notna(gl) else None)

        faith_series = df_f.apply(_faith_val, axis=1)
        denom = faith_series.notna().sum()
        halluc_rate = float(
            ((faith_series.dropna() < float(thr_faith)).sum() / denom) if denom else 0.0
        )
        no_answer_rate = float(
            (df_f.get("exit_state") == "no_answer").mean() if "exit_state" in df_f else 0.0
        )
        slow_rate = float(
            (pd.to_numeric(df_f.get("t_total"), errors="coerce") > float(thr_latency)).mean()
        )
        high_cost_rate = float(
            (pd.to_numeric(df_f.get("cost"), errors="coerce") > float(thr_cost)).mean()
        )
        with tok_cols[2]:
            st.metric("Hallucination rate", f"{halluc_rate * 100:.1f}%")

    # Table
    st.dataframe(df_f.drop(columns=["_dt"], errors="ignore"), use_container_width=True)
    st.download_button(
        "Export CSV",
        data=df_f.drop(columns=["_dt"], errors="ignore").to_csv(index=False),
        file_name="runs.csv",
    )

    st.divider()
    # Trends
    st.subheader("Trends")
    if not df_f.empty:
        plot_df = df_f.copy()
        plot_df = plot_df.sort_values("_dt")
        tcols = [c for c in ["t_total", "t_retrieve", "t_generate"] if c in plot_df]
        if tcols:
            st.line_chart(plot_df.set_index("_dt")[tcols])
        tokcols = [c for c in ["prompt_tokens", "completion_tokens"] if c in plot_df]
        if tokcols:
            st.line_chart(plot_df.set_index("_dt")[tokcols])
        mcols = [
            c
            for c in ["answer_relevancy_lite", "context_precision_lite", "groundedness_lite"]
            if c in plot_df
        ]
        if mcols:
            st.line_chart(plot_df.set_index("_dt")[mcols])
        if "cost" in plot_df:
            st.line_chart(plot_df.set_index("_dt")["cost"], height=150)

    st.divider()
    # Notable cases
    st.subheader("Notable cases")
    if not df_f.empty:
        col_na, col_h, col_slow, col_cost = st.columns(4)
        with col_na:
            st.caption("No Answer")
            try:
                st.dataframe(
                    df_f[df_f.get("exit_state") == "no_answer"][
                        ["id", "flow", "created_at", "t_total", "question"]
                    ].tail(10),
                    use_container_width=True,
                )
            except Exception:
                st.caption("None")
        with col_h:
            st.caption("Low faithfulness/groundedness")
            try:
                faith_vals = df_f.assign(
                    _faith=df_f["ragas_faithfulness"].fillna(df_f["groundedness_lite"])  # type: ignore
                )
                st.dataframe(
                    faith_vals[
                        pd.to_numeric(faith_vals["_faith"], errors="coerce") < float(thr_faith)
                    ][["id", "flow", "created_at", "_faith", "question"]].tail(10),
                    use_container_width=True,
                )
            except Exception:
                st.caption("None")
        with col_slow:
            st.caption("High latency")
            try:
                st.dataframe(
                    df_f[pd.to_numeric(df_f["t_total"], errors="coerce") > float(thr_latency)][
                        ["id", "flow", "created_at", "t_total", "question"]
                    ].tail(10),
                    use_container_width=True,
                )
            except Exception:
                st.caption("None")
        with col_cost:
            st.caption("High cost")
            try:
                st.dataframe(
                    df_f[pd.to_numeric(df_f["cost"], errors="coerce") > float(thr_cost)][
                        ["id", "flow", "created_at", "cost", "question"]
                    ].tail(10),
                    use_container_width=True,
                )
            except Exception:
                st.caption("None")

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

    st.divider()
    # Index health
    st.subheader("Index health")
    try:
        cnt = get_counts()
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.metric("Documents", f"{cnt.get('documents', 0)}")
        with ic2:
            st.metric("Chunks", f"{cnt.get('chunks', 0)}")
        with ic3:
            st.metric("Indices", f"{cnt.get('indices', 0)}")
    except Exception:
        st.caption("Counts unavailable.")
