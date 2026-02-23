"""
dashboard.py
------------
Streamlit dashboard for real-time fatigue monitoring.

Run:
    streamlit run phase4/dashboard/dashboard.py

Features:
    - Live risk score gauges per player
    - Risk timeline chart (rolling window)
    - Alert feed with colour-coded severity
    - Model agreement panel (LR vs LSTM vs Transformer)
    - Top-at-risk leaderboard
    - SHAP feature contribution bar (if shap_values.npy exists)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# __file__ = fatigue_predictor/phase4/dashboard/dashboard.py
# .parent   = fatigue_predictor/phase4/dashboard/
# .parent.parent = fatigue_predictor/phase4/
# .parent.parent.parent = fatigue_predictor/   ← project root
# Walk up from this file until we find the folder that contains phase4/outputs/
# Works regardless of whether dashboard.py is in phase4/dashboard/ or dashboard/
_here = Path(__file__).resolve()
ROOT  = _here.parent  # start one level up from dashboard.py
while not (ROOT / "phase4" / "outputs").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Fatigue Predictor",
    page_icon   = "⚡",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@400;700;800&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

.main { background: #070910; }
h1, h2, h3 { font-family: 'Oxanium', sans-serif !important; letter-spacing: 0.05em; }

.risk-card {
    background: #0d1117;
    border: 1px solid #161b24;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    margin-bottom: 8px;
}
.risk-val { font-family: 'Oxanium', sans-serif; font-size: 2.4rem; font-weight: 800; line-height: 1; }
.risk-name { font-size: 0.75rem; color: #3a4a5e; letter-spacing: 0.1em; margin-top: 4px; }

.alert-red   { border-left: 4px solid #f03a7a; background: rgba(240,58,122,0.07); padding: 10px 14px; border-radius: 4px; margin: 4px 0; }
.alert-amber { border-left: 4px solid #f0c93a; background: rgba(240,201,58,0.07); padding: 10px 14px; border-radius: 4px; margin: 4px 0; }
.alert-green { border-left: 4px solid #3af0a0; background: rgba(58,240,160,0.05); padding: 10px 14px; border-radius: 4px; margin: 4px 0; }

.metric-box {
    background: #0d1117;
    border: 1px solid #161b24;
    border-radius: 8px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Load simulation data ───────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_sim_data() -> pd.DataFrame | None:
    csv_path = ROOT / "phase4" / "outputs" / "simulation_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_alerts() -> list[dict]:
    log_path = ROOT / "phase4" / "outputs" / "alerts.jsonl"
    if not log_path.exists():
        return []
    alerts = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    alerts.append(json.loads(line))
                except Exception:
                    pass
    return alerts


@st.cache_data
def load_shap() -> tuple[np.ndarray, list[str]] | None:
    shap_path  = ROOT / "phase3" / "outputs" / "shap_values.npy"
    feats_path = ROOT / "phase3" / "outputs" / "feature_names.json"
    if shap_path.exists() and feats_path.exists():
        shap_vals = np.load(str(shap_path))
        with open(feats_path) as f:
            feat_names = json.load(f)
        return shap_vals, feat_names
    return None


@st.cache_data
def load_comparison() -> list[dict] | None:
    for phase in ["phase3", "phase2"]:
        p = ROOT / phase / "outputs" / "model_comparison.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Fatigue Predictor")
    st.markdown("---")

    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()

    risk_threshold = st.slider("Alert threshold", 0.0, 1.0, 0.65, 0.05)
    top_n = st.number_input("Players shown", 5, 30, 11, 1)

    st.markdown("---")
    st.markdown("**Alert levels**")
    st.markdown("🟢 GREEN  < 0.40")
    st.markdown("🟡 AMBER  0.40–0.65")
    st.markdown("🔴 RED    > 0.65")

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ── Main content ───────────────────────────────────────────────────────────────
st.markdown("# ⚡ FATIGUE PREDICTOR")
st.markdown("##### Real-time athlete performance drop monitoring")
st.markdown("---")

df = load_sim_data()

if df is None:
    csv_path = ROOT / "phase4" / "outputs" / "simulation_results.csv"
    st.warning(
        f"No simulation data found.\n\n"
        f"**Looking for:** `{csv_path}`\n\n"
        f"Run the pipeline first:\n```\npython phase4/pipeline.py\n```\n\n"
        f"Then click **Refresh data** in the sidebar."
    )
    st.info(f"Dashboard ROOT resolved to: `{ROOT}`")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

latest = df.sort_values("match_time_s").groupby("player_id").last().reset_index()
n_red   = (latest["alert_level"] == "RED").sum()
n_amber = (latest["alert_level"] == "AMBER").sum()
avg_lat = df["latency_ms"].mean()
total_alerts = df["alert_fired"].sum()

with kpi1:
    st.metric("🔴 RED Alerts",   int(n_red))
with kpi2:
    st.metric("🟡 AMBER Alerts", int(n_amber))
with kpi3:
    st.metric("⚡ Avg Latency",  f"{avg_lat:.1f} ms")
with kpi4:
    st.metric("📢 Total Alerts", int(total_alerts))

st.markdown("---")

# ── Risk leaderboard ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.4, 1])

with col_left:
    st.markdown("### 🏃 Player Risk Leaderboard")
    latest_sorted = latest.sort_values("risk_score", ascending=False).head(int(top_n))

    for _, row in latest_sorted.iterrows():
        score  = row["risk_score"]
        level  = row["alert_level"]
        name   = row.get("player_name", f"Player {row['player_id']:.0f}")
        colour = {"RED": "#f03a7a", "AMBER": "#f0c93a", "GREEN": "#3af0a0"}.get(level, "#3af0a0")
        bar_w  = int(score * 100)

        st.markdown(f"""
        <div class="risk-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:#c8d6e5;font-size:0.85rem;">{name}</span>
                <span class="risk-val" style="color:{colour};">{score:.2f}</span>
            </div>
            <div style="background:#161b24;border-radius:4px;height:6px;margin-top:8px;">
                <div style="background:{colour};width:{bar_w}%;height:6px;border-radius:4px;"></div>
            </div>
            <div class="risk-name">{level}</div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("### 📢 Recent Alerts")
    alerts = load_alerts()
    if alerts:
        for alert in reversed(alerts[-15:]):
            level   = alert.get("alert_level", "GREEN")
            css_cls = f"alert-{level.lower()}"
            ts      = alert.get("timestamp_s", 0)
            mins    = int(ts // 60)
            secs    = int(ts % 60)
            name    = alert.get("player_name", "Unknown")
            action  = alert.get("action", "")
            score   = alert.get("risk_score", 0)
            st.markdown(f"""
            <div class="{css_cls}">
                <strong>{name}</strong>
                <span style="float:right;color:#3a4a5e;font-size:0.75rem;">{mins:02d}:{secs:02d}</span><br>
                <span style="font-size:0.8rem;color:#8ea8c3;">{action}</span><br>
                <span style="font-size:0.75rem;color:#3a4a5e;">risk={score:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No alerts yet.")

st.markdown("---")

# ── Timeline chart ─────────────────────────────────────────────────────────────
st.markdown("### 📈 Risk Timeline")

player_options = sorted(df["player_name"].unique()) if "player_name" in df.columns \
                 else [f"Player {p:.0f}" for p in sorted(df["player_id"].unique())]

selected_players = st.multiselect(
    "Select players", player_options,
    default=player_options[:min(3, len(player_options))],
)

if selected_players:
    filter_col = "player_name" if "player_name" in df.columns else "player_id"
    filtered = df[df[filter_col].isin(selected_players)].copy()
    filtered["match_time_min"] = filtered["match_time_s"] / 60

    chart_df = filtered.pivot_table(
        index="match_time_min",
        columns=filter_col,
        values="risk_score",
        aggfunc="mean",
    )
    st.line_chart(chart_df, height=280)

st.markdown("---")

# ── Model agreement panel ─────────────────────────────────────────────────────
st.markdown("### 🤖 Model Agreement")
mcol1, mcol2, mcol3 = st.columns(3)

for col, model_col, label in [
    (mcol1, "lr_score",          "Logistic Regression"),
    (mcol2, "lstm_score",        "LSTM"),
    (mcol3, "transformer_score", "Transformer"),
]:
    if model_col in df.columns:
        with col:
            latest_model = latest[model_col].mean()
            st.metric(label, f"{latest_model:.3f}")
            hist_df = df.groupby("match_time_s")[model_col].mean().reset_index()
            hist_df.columns = ["time", model_col]
            st.line_chart(hist_df.set_index("time"), height=140)

st.markdown("---")

# ── SHAP panel ────────────────────────────────────────────────────────────────
shap_data = load_shap()
if shap_data:
    st.markdown("### 🔍 Feature Importance (SHAP)")
    shap_vals, feat_names = shap_data
    top_k   = 15
    order   = np.argsort(shap_vals)[::-1][:top_k]
    shap_df = pd.DataFrame({
        "Feature":    [feat_names[i] for i in order],
        "Importance": shap_vals[order],
    }).set_index("Feature")
    st.bar_chart(shap_df, height=320)

# ── Model comparison ──────────────────────────────────────────────────────────
comparison = load_comparison()
if comparison:
    st.markdown("### 📊 Model Comparison")
    comp_df = pd.DataFrame(comparison).set_index("name")
    st.dataframe(comp_df.style.highlight_max(axis=0, color="#1a2a1a"), use_container_width=True)

st.markdown("---")
st.caption("AI Fatigue Predictor · Phase 4 · Built with StatsBomb open data")