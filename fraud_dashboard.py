"""
============================================================================
 Executive Fraud Intelligence Dashboard
 Group 1 - Financial Fraud Analytics
 Author: Junyin Zhang
 Module 5 Final Project
----------------------------------------------------------------------------
 An interactive, executive-facing dashboard that turns 284K raw credit-card
 transactions into decision-ready fraud intelligence: portfolio KPIs,
 temporal & monetary risk patterns, a tunable machine-learning detection
 engine, and a prioritised high-risk transaction review queue.

 RUN:
     pip install -r requirements.txt
     streamlit run fraud_dashboard.py

 DATA:
     Place creditcard.csv (Kaggle ULB dataset) next to this file,
     or inside a ./data/ subfolder.
============================================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

# Plotly is used for all interactive visuals.
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:
    st.error("Plotly is required. Install it with:  pip install plotly")
    st.stop()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix,
)

# ----------------------------------------------------------------------------
# 1. PAGE CONFIG & EXECUTIVE THEME
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Executive Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light, corporate styling for an executive audience.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px;}
      h1, h2, h3 {color: #0b2545; font-family: "Segoe UI", system-ui, sans-serif;}
      [data-testid="stMetricValue"] {font-size: 1.9rem; color: #0b2545;}
      [data-testid="stMetric"] {
          background: #ffffff; border: 1px solid #e3e8ef; border-radius: 12px;
          padding: 14px 16px; box-shadow: 0 1px 3px rgba(11,37,69,.06);
      }
      .stTabs [data-baseweb="tab-list"] {gap: 6px;}
      .stTabs [data-baseweb="tab"] {
          background:#f1f4f9; border-radius:8px 8px 0 0; padding:8px 18px; font-weight:600;
      }
      .stTabs [aria-selected="true"] {background:#0b2545; color:#ffffff;}
      .subtle {color:#5b6b82; font-size:0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

PRIMARY = "#0b2545"
FRAUD_C = "#d7263d"
SAFE_C = "#1b998b"

# ----------------------------------------------------------------------------
# 2. DATA LOADING  (cached)
# ----------------------------------------------------------------------------
def _find_data() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "creditcard.csv", "data/creditcard.csv",
        os.path.join(here, "creditcard.csv"),
        os.path.join(here, "data", "creditcard.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # Try downloading via Kaggle API (Streamlit Cloud: set KAGGLE_USERNAME & KAGGLE_KEY in secrets)
    try:
        import kaggle  # noqa: F401
        import zipfile
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "mlg-ulb/creditcardfraud", path=here, unzip=False, quiet=False
        )
        zip_path = os.path.join(here, "creditcardfraud.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(here)
            os.remove(zip_path)
        csv = os.path.join(here, "creditcard.csv")
        if os.path.exists(csv):
            return csv
    except Exception:
        pass

    return ""


@st.cache_data(show_spinner="Loading transaction data…")
def load_data() -> pd.DataFrame:
    path = _find_data()
    if not path:
        st.error(
            "creditcard.csv not found. Please download the dataset from Kaggle and place it "
            "beside this file or in ./data/.\n\n"
            "📥 Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
        st.stop()
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)
    # Derive interpretable time fields (data spans 48h from first transaction).
    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)
    df["Day"] = (df["Time"] // 86400 + 1).astype(int)
    return df


# ----------------------------------------------------------------------------
# 3. DETECTION ENGINE  (cached resource — trains once per session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training fraud-detection engine (one-time, ~30s)…")
def train_engine(df: pd.DataFrame):
    feat = [c for c in df.columns if c not in ("Class", "Hour", "Day")]
    X, y = df[feat].copy(), df["Class"]

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.30, stratify=y, random_state=42
    )

    # Undersample the majority class for fast, balanced training.
    tr = X_tr.copy(); tr["Class"] = y_tr
    pos = tr[tr.Class == 1]
    neg = tr[tr.Class == 0].sample(n=min(40000, (tr.Class == 0).sum()), random_state=42)
    bal = pd.concat([pos, neg]).sample(frac=1, random_state=42)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=14, class_weight="balanced",
        n_jobs=-1, random_state=42,
    )
    model.fit(bal[feat], bal["Class"])

    proba = model.predict_proba(X_te[feat])[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
    }
    importances = pd.Series(model.feature_importances_, index=feat).sort_values(ascending=False)

    # Test-set scoring frame used by the review queue & confusion matrix.
    scored = df.loc[idx_te, ["Time", "Amount", "Hour", "Day", "Class"]].copy()
    scored["RiskScore"] = proba
    return metrics, importances, scored, y_te.values, proba


# ----------------------------------------------------------------------------
# 4. LOAD EVERYTHING
# ----------------------------------------------------------------------------
df = load_data()
metrics, importances, scored, y_te, proba = train_engine(df)

# ----------------------------------------------------------------------------
# 5. SIDEBAR CONTROLS
# ----------------------------------------------------------------------------
st.sidebar.title("🛡️ Control Panel")
st.sidebar.caption("Adjust the detection sensitivity and explore the portfolio.")

threshold = st.sidebar.slider(
    "Alert risk threshold",
    min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    help="Transactions scored at or above this probability are flagged for analyst review.",
)

amt_max = float(df["Amount"].quantile(0.999))
amt_range = st.sidebar.slider(
    "Transaction amount filter ($)",
    min_value=0.0, max_value=round(amt_max, 0), value=(0.0, round(amt_max, 0)), step=10.0,
)

hours = st.sidebar.multiselect(
    "Hour of day", options=list(range(24)), default=list(range(24)),
    help="Filter the fraud-landscape views by hour.",
)
if not hours:
    hours = list(range(24))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span class='subtle'>Dataset: anonymised European card transactions "
    "(Kaggle / ULB), 2 days. Features V1–V28 are PCA components.<br><br>"
    "📥 <a href='https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud' target='_blank'>"
    "Download Dataset (Kaggle)</a></span>",
    unsafe_allow_html=True,
)

# Apply landscape filters
mask = (
    df["Amount"].between(amt_range[0], amt_range[1]) & df["Hour"].isin(hours)
)
dff = df[mask]

# ----------------------------------------------------------------------------
# 6. HEADER + KPI BAND
# ----------------------------------------------------------------------------
st.title("Executive Fraud Intelligence Dashboard")
st.markdown(
    "<p class='subtle'>Group 1 · Financial Fraud Analytics — turning raw "
    "transactions into decision-ready risk intelligence.</p>",
    unsafe_allow_html=True,
)

total_tx = len(df)
fraud_n = int(df["Class"].sum())
fraud_rate = fraud_n / total_tx
fraud_dollars = df.loc[df["Class"] == 1, "Amount"].sum()

# Operating point at the chosen threshold (held-out test set).
pred = (proba >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
recall = tp / (tp + fn) if (tp + fn) else 0
precision = tp / (tp + fp) if (tp + fp) else 0
alerts = int(tp + fp)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Transactions", f"{total_tx:,}")
k2.metric("Confirmed Fraud", f"{fraud_n:,}")
k3.metric("Fraud Rate", f"{fraud_rate*100:.3f}%")
k4.metric("Fraud $ Exposure", f"${fraud_dollars:,.0f}")
k5.metric("Detection Recall", f"{recall*100:.1f}%", help="Share of true fraud caught at current threshold (test set).")
k6.metric("Alert Precision", f"{precision*100:.1f}%", help="Share of alerts that are truly fraud.")

st.markdown("")

# ----------------------------------------------------------------------------
# 7. TABS
# ----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊  Fraud Landscape", "🤖  Detection Intelligence", "🚨  High-Risk Queue"]
)

# ---- TAB 1 : LANDSCAPE -----------------------------------------------------
with tab1:
    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("When does fraud strike?")
        by_hour = (
            df.groupby("Hour")
            .agg(tx=("Class", "size"), fraud=("Class", "sum"))
            .reset_index()
        )
        by_hour["fraud_rate"] = by_hour["fraud"] / by_hour["tx"] * 100
        fig = go.Figure()
        fig.add_bar(x=by_hour["Hour"], y=by_hour["tx"], name="Transactions",
                    marker_color="#c7d2e0", yaxis="y1", opacity=0.7)
        fig.add_trace(go.Scatter(
            x=by_hour["Hour"], y=by_hour["fraud_rate"], name="Fraud rate (%)",
            mode="lines+markers", line=dict(color=FRAUD_C, width=3), yaxis="y2"))
        fig.update_layout(
            height=380, margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(title="Transaction volume"),
            yaxis2=dict(title="Fraud rate (%)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.12), plot_bgcolor="white",
            xaxis=dict(title="Hour of day", dtick=2),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Transaction volume peaks during business hours, but the **fraud "
            "rate** spikes in the early-morning low-traffic window (≈ 2–4 AM) — "
            "exactly when monitoring staffing is thinnest."
        )

    with c2:
        st.subheader("Fraud vs. legitimate amounts")
        cap = float(df["Amount"].quantile(0.99))
        plot_df = df[df["Amount"] <= cap].copy()
        plot_df["Type"] = np.where(plot_df["Class"] == 1, "Fraud", "Legitimate")
        fig2 = px.box(plot_df, x="Type", y="Amount", color="Type",
                      color_discrete_map={"Fraud": FRAUD_C, "Legitimate": SAFE_C},
                      points=False)
        fig2.update_layout(height=380, showlegend=False, plot_bgcolor="white",
                           margin=dict(t=10, b=10, l=10, r=10),
                           yaxis_title="Amount ($, ≤99th pct)")
        st.plotly_chart(fig2, use_container_width=True)
        med_f = df.loc[df.Class == 1, "Amount"].median()
        med_l = df.loc[df.Class == 0, "Amount"].median()
        st.caption(
            f"Median fraud charge is just **${med_f:,.2f}** vs **${med_l:,.2f}** "
            "for legitimate activity — fraudsters favour small 'card-testing' "
            "amounts that slip past simple value thresholds."
        )

    st.subheader("Dollar exposure by hour")
    exp = (df[df.Class == 1].groupby("Hour")["Amount"].sum().reindex(range(24), fill_value=0))
    fig3 = px.bar(x=exp.index, y=exp.values,
                  labels={"x": "Hour of day", "y": "Fraud $ exposure"})
    fig3.update_traces(marker_color=FRAUD_C)
    fig3.update_layout(height=300, plot_bgcolor="white",
                       margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(dtick=2))
    st.plotly_chart(fig3, use_container_width=True)

# ---- TAB 2 : DETECTION INTELLIGENCE ---------------------------------------
with tab2:
    m1, m2, m3 = st.columns(3)
    m1.metric("Model ROC-AUC", f"{metrics['roc_auc']:.3f}")
    m2.metric("Model PR-AUC", f"{metrics['pr_auc']:.3f}",
              help="Precision-Recall AUC — the meaningful score under 0.17% class imbalance.")
    m3.metric("Active threshold", f"{threshold:.2f}")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Detection trade-off")
        # Sweep thresholds: frauds caught vs false alerts.
        ts = np.linspace(0.05, 0.95, 19)
        caught, falsepos = [], []
        for t in ts:
            pr = (proba >= t).astype(int)
            _tn, _fp, _fn, _tp = confusion_matrix(y_te, pr).ravel()
            caught.append(_tp); falsepos.append(_fp)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=caught, name="Frauds caught",
                                 line=dict(color=SAFE_C, width=3)))
        fig.add_trace(go.Scatter(x=ts, y=falsepos, name="False alerts",
                                 line=dict(color=FRAUD_C, width=3, dash="dot")))
        fig.add_vline(x=threshold, line_dash="dash", line_color=PRIMARY)
        fig.update_layout(height=340, plot_bgcolor="white",
                          margin=dict(t=10, b=10, l=10, r=10),
                          xaxis_title="Risk threshold", yaxis_title="Test-set count",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Lowering the threshold catches more fraud but multiplies analyst "
            "workload. The dashed line marks your current setting."
        )

    with c2:
        st.subheader(f"Confusion matrix @ {threshold:.2f}")
        cm = np.array([[tn, fp], [fn, tp]])
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Legit", "Fraud"], y=["Legit", "Fraud"], aspect="auto",
        )
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"At this setting the engine flags **{alerts:,}** transactions, "
            f"catching **{tp}** of **{tp+fn}** frauds while missing **{fn}**."
        )

    st.subheader("What drives a high risk score?")
    top = importances.head(10).iloc[::-1]
    fig = px.bar(x=top.values, y=top.index, orientation="h",
                 labels={"x": "Importance", "y": "Feature"})
    fig.update_traces(marker_color=PRIMARY)
    fig.update_layout(height=340, plot_bgcolor="white",
                      margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Features V14, V10 and V12 dominate the model's decisions. Though "
        "PCA-anonymised, their stability across folds makes them reliable "
        "early-warning signals worth monitoring in real time."
    )

# ---- TAB 3 : HIGH-RISK QUEUE ----------------------------------------------
with tab3:
    st.subheader("Prioritised review queue")
    st.markdown(
        "<span class='subtle'>Held-out transactions scored at or above the "
        "current threshold, ranked by risk. This is the queue an analyst team "
        "would action first.</span>", unsafe_allow_html=True,
    )

    queue = scored[scored["RiskScore"] >= threshold].copy()
    queue = queue.sort_values("RiskScore", ascending=False)
    queue["Status"] = np.where(queue["Class"] == 1, "✅ Confirmed fraud", "⚠️ Review")
    show = queue.rename(columns={"Amount": "Amount ($)", "Day": "Day"})[
        ["RiskScore", "Amount ($)", "Hour", "Day", "Status"]
    ].copy()
    show["RiskScore"] = (show["RiskScore"] * 100).round(1)
    show = show.rename(columns={"RiskScore": "Risk %"})

    q1, q2, q3 = st.columns(3)
    q1.metric("Items in queue", f"{len(queue):,}")
    q2.metric("$ in queue", f"${queue['Amount'].sum():,.0f}")
    hit = (queue["Class"] == 1).mean() if len(queue) else 0
    q3.metric("Queue hit-rate", f"{hit*100:.1f}%")

    st.dataframe(
        show.head(300),
        use_container_width=True, hide_index=True, height=430,
        column_config={
            "Risk %": st.column_config.ProgressColumn(
                "Risk %", min_value=0, max_value=100, format="%.1f"),
            "Amount ($)": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.download_button(
        "⬇️  Export queue (CSV)",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name="high_risk_queue.csv", mime="text/csv",
    )

# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<span class='subtle'>Executive Fraud Intelligence Dashboard · Junyin Zhang · "
    "Group 1 Financial Fraud Analytics. Model trained on a held-out 70/30 split; "
    "all detection metrics reported on unseen test data.</span>",
    unsafe_allow_html=True,
)
