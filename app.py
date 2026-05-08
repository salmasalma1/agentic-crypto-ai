# app.py

import os
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv

from pipeline.ingestion import load_data
from pipeline.model_persistence import get_artifact_status
from pipeline.pipeline_runner import run_crypto_ai_pipeline
from utils.config import ACTION_LOG_PATH, SUPPORTED_CIPHER_TYPES

load_dotenv()

# =========================================================
# PLOTLY CYBER DARK TEMPLATE
# =========================================================

CYBER_BG = "rgba(6, 14, 24, 0.96)"
CYBER_PAPER = "rgba(8, 18, 30, 0.96)"
CYBER_GRID = "rgba(0, 229, 255, 0.10)"
CYBER_TEXT = "#EAFBFF"
CYBER_CYAN = "#00E5FF"
CYBER_GREEN = "#7CFFB2"
CYBER_PURPLE = "#A970FF"
CYBER_ORANGE = "#FFC857"
CYBER_RED = "#FF4D6D"

pio.templates["crypto_cyber_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=CYBER_PAPER,
        plot_bgcolor=CYBER_BG,
        font=dict(color=CYBER_TEXT, family="Inter, sans-serif"),
        title=dict(font=dict(color=CYBER_TEXT, size=18), x=0.02),
        xaxis=dict(
            gridcolor=CYBER_GRID,
            zerolinecolor=CYBER_GRID,
            linecolor="rgba(0,229,255,0.25)",
            tickfont=dict(color=CYBER_TEXT),
            title_font=dict(color=CYBER_TEXT),
        ),
        yaxis=dict(
            gridcolor=CYBER_GRID,
            zerolinecolor=CYBER_GRID,
            linecolor="rgba(0,229,255,0.25)",
            tickfont=dict(color=CYBER_TEXT),
            title_font=dict(color=CYBER_TEXT),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CYBER_TEXT)),
        colorway=[CYBER_CYAN, CYBER_GREEN, CYBER_PURPLE, CYBER_ORANGE, CYBER_RED],
        margin=dict(l=40, r=30, t=55, b=45),
    )
)

px.defaults.template = "crypto_cyber_dark"
px.defaults.color_discrete_sequence = [
    CYBER_CYAN,
    CYBER_GREEN,
    CYBER_PURPLE,
    CYBER_ORANGE,
    CYBER_RED,
]

st.set_page_config(
    page_title="CRYPTO AI | SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# HELPERS
# =========================================================


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def fmt_score(value, default: float = 0.0) -> str:
    return f"{safe_float(value, default):.4f}"


def threat_level(score: float) -> tuple[str, str]:
    score = safe_float(score)

    if score >= 0.85:
        return "CRITICAL", "critical"
    if score >= 0.75:
        return "HIGH", "high"
    if score >= 0.55:
        return "ELEVATED", "medium"
    return "LOW", "low"


def reliability_kind(level: str, score: float) -> str:
    level = str(level or "").lower()
    score = safe_float(score)

    if level == "high" or score >= 0.85:
        return "good"
    if level == "medium" or score >= 0.65:
        return "warn"
    if level == "fallback":
        return "warn"
    return "bad"


def status_badge(text: str, kind: str = "info") -> str:
    return f'<span class="status-badge {kind}">{text}</span>'


def kpi_card(title: str, value: str, subtitle: str = "", kind: str = "cyan") -> str:
    return f"""
    <div class="kpi-card {kind}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """


def json_debug_block(title: str, payload, expanded: bool = False):
    with st.expander(title, expanded=expanded):
        st.code(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            language="json",
        )


# =========================================================
# ENTERPRISE CYBERSECURITY UI THEME
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    :root {
        --bg: #03070d;
        --panel: rgba(7, 17, 28, 0.92);
        --panel-2: rgba(11, 24, 39, 0.92);
        --panel-3: rgba(6, 14, 24, 0.96);
        --cyan: #00e5ff;
        --cyan-soft: rgba(0, 229, 255, 0.14);
        --green: #7cffb2;
        --green-soft: rgba(124, 255, 178, 0.12);
        --purple: #a970ff;
        --red: #ff4d6d;
        --orange: #ffc857;
        --text: #eafcff;
        --muted: #9dc9d8;
        --faint: #6d9aaa;
        --border: rgba(0, 229, 255, 0.18);
        --border-strong: rgba(0, 229, 255, 0.34);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        color: var(--text);
        background:
            radial-gradient(circle at 12% 16%, rgba(0, 229, 255, 0.12), transparent 25%),
            radial-gradient(circle at 85% 12%, rgba(124, 255, 178, 0.10), transparent 22%),
            radial-gradient(circle at 45% 85%, rgba(169, 112, 255, 0.10), transparent 26%),
            linear-gradient(180deg, #02060a 0%, #03101a 38%, #06111d 100%);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(0, 255, 170, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 229, 255, 0.030) 1px, transparent 1px);
        background-size: 34px 34px;
        opacity: 0.33;
        animation: gridMove 18s linear infinite;
        z-index: 0;
    }

    @keyframes gridMove {
        from { background-position: 0 0; }
        to { background-position: 34px 34px; }
    }

    .stApp::after {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
            repeating-linear-gradient(
                to bottom,
                rgba(255, 255, 255, 0.018) 0px,
                rgba(255, 255, 255, 0.018) 1px,
                transparent 2px,
                transparent 6px
            );
        opacity: 0.10;
        z-index: 0;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        position: relative;
        z-index: 1;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 0.45px;
        color: var(--text);
    }

    h1 {
        font-size: 2.35rem !important;
        font-weight: 800 !important;
    }

    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.08rem !important; }

    p, li, label, span, div {
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(3, 9, 16, 0.98), rgba(7, 17, 28, 0.98));
        border-right: 1px solid rgba(0, 229, 255, 0.18);
        box-shadow: 0 0 34px rgba(0, 229, 255, 0.08);
    }

    section[data-testid="stSidebar"] * { color: var(--text) !important; }

    textarea, input, .stTextArea textarea, .stTextInput input {
        background: rgba(8, 18, 30, 0.96) !important;
        color: #eafcff !important;
        border-radius: 14px !important;
        border: 1px solid rgba(0, 229, 255, 0.24) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stTextArea textarea::placeholder { color: rgba(157, 201, 216, 0.58) !important; }

    .stButton button, .stDownloadButton button {
        background:
            linear-gradient(135deg, rgba(0, 229, 255, 0.20), rgba(124, 255, 178, 0.12)) !important;
        border: 1px solid rgba(0, 229, 255, 0.32) !important;
        color: white !important;
        border-radius: 14px !important;
        font-weight: 800 !important;
        box-shadow: 0 0 18px rgba(0, 229, 255, 0.12) !important;
        transition: all 0.25s ease !important;
    }

    .stButton button:hover, .stDownloadButton button:hover {
        transform: translateY(-1px);
        border-color: rgba(124, 255, 178, 0.55) !important;
        box-shadow: 0 0 28px rgba(124, 255, 178, 0.18) !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(0, 229, 255, 0.12), rgba(124, 255, 178, 0.06)),
            linear-gradient(180deg, rgba(8, 18, 30, 0.96), rgba(6, 14, 24, 0.92));
        border: 1px solid var(--border-strong);
        border-radius: 24px;
        padding: 1.35rem 1.45rem 1.15rem;
        margin-bottom: 1rem;
        box-shadow:
            0 0 38px rgba(0, 229, 255, 0.12),
            inset 0 0 28px rgba(0, 229, 255, 0.035);
    }

    .hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.12), transparent);
        animation: sweep 6.5s linear infinite;
        transform: translateX(-100%);
        opacity: 0.55;
    }

    @keyframes sweep {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    .hero-content { position: relative; z-index: 2; }
    .hero-sub { color: var(--muted); line-height: 1.65; font-size: 1rem; }
    .hero-title-glow { text-shadow: 0 0 10px rgba(0, 229, 255, 0.25), 0 0 24px rgba(0, 229, 255, 0.14); }

    .pill {
        display: inline-block;
        padding: 0.33rem 0.72rem;
        margin: 0.22rem 0.32rem 0.1rem 0;
        border-radius: 999px;
        background: rgba(0, 229, 255, 0.08);
        border: 1px solid rgba(0, 229, 255, 0.20);
        color: #e9fdff;
        font-size: 0.82rem;
        font-weight: 800;
        box-shadow: inset 0 0 12px rgba(0, 229, 255, 0.035);
    }

    .soc-card {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(180deg, rgba(8, 18, 30, 0.95), rgba(10, 21, 35, 0.93));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1rem 1rem 0.9rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 26px rgba(0, 229, 255, 0.075);
    }

    .soc-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: -30%;
        height: 1px;
        width: 60%;
        background: linear-gradient(90deg, transparent, var(--cyan), transparent);
        animation: borderScan 4s ease-in-out infinite;
        opacity: 0.65;
    }

    @keyframes borderScan { 0% { left: -40%; } 100% { left: 110%; } }
    .soc-card-inner { position: relative; z-index: 2; }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin-bottom: 1rem;
    }

    .kpi-card {
        background:
            linear-gradient(180deg, rgba(8, 18, 30, 0.96), rgba(10, 21, 35, 0.92));
        border: 1px solid rgba(0, 229, 255, 0.18);
        border-radius: 18px;
        padding: 0.9rem 0.95rem;
        box-shadow: 0 0 24px rgba(0, 229, 255, 0.08);
        min-height: 112px;
        position: relative;
        overflow: hidden;
        animation: cardIn 0.55s ease both;
    }

    @keyframes cardIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .kpi-card::after {
        content: "";
        position: absolute;
        right: -26px;
        top: -26px;
        width: 84px;
        height: 84px;
        border-radius: 50%;
        background: rgba(0, 229, 255, 0.065);
        box-shadow: 0 0 28px rgba(0, 229, 255, 0.10);
    }

    .kpi-card.green::after { background: rgba(124, 255, 178, 0.08); }
    .kpi-card.purple::after { background: rgba(169, 112, 255, 0.08); }
    .kpi-card.red::after { background: rgba(255, 77, 109, 0.08); }
    .kpi-card.orange::after { background: rgba(255, 200, 87, 0.08); }

    .kpi-title {
        color: var(--muted);
        font-weight: 800;
        font-size: 0.86rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.45rem;
    }

    .kpi-value {
        font-family: 'Orbitron', sans-serif;
        color: var(--cyan);
        font-size: 1.65rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        word-break: break-word;
    }

    .kpi-card.green .kpi-value { color: var(--green); }
    .kpi-card.purple .kpi-value { color: var(--purple); }
    .kpi-card.red .kpi-value { color: var(--red); }
    .kpi-card.orange .kpi-value { color: var(--orange); }

    .kpi-subtitle { color: var(--faint); font-size: 0.83rem; line-height: 1.35; }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 900;
        letter-spacing: 0.4px;
        border: 1px solid rgba(0, 229, 255, 0.18);
        background: rgba(0, 229, 255, 0.07);
        color: var(--text);
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .status-badge::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--cyan);
        box-shadow: 0 0 10px var(--cyan);
        animation: pulseDot 1.6s ease-in-out infinite;
    }

    @keyframes pulseDot {
        0%, 100% { opacity: 0.55; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.2); }
    }

    .status-badge.good::before { background: var(--green); box-shadow: 0 0 10px var(--green); }
    .status-badge.warn::before { background: var(--orange); box-shadow: 0 0 10px var(--orange); }
    .status-badge.bad::before { background: var(--red); box-shadow: 0 0 10px var(--red); }
    .status-badge.critical::before { background: var(--red); box-shadow: 0 0 14px var(--red); }

    .status-badge.good { border-color: rgba(124, 255, 178, 0.28); background: rgba(124, 255, 178, 0.08); }
    .status-badge.warn { border-color: rgba(255, 200, 87, 0.28); background: rgba(255, 200, 87, 0.08); }
    .status-badge.bad, .status-badge.critical { border-color: rgba(255, 77, 109, 0.28); background: rgba(255, 77, 109, 0.08); }

    div[data-testid="stMetric"] {
        background:
            linear-gradient(180deg, rgba(8, 18, 30, 0.96), rgba(10, 21, 35, 0.92));
        border: 1px solid rgba(0, 229, 255, 0.18);
        border-radius: 16px;
        padding: 0.65rem 0.8rem;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.06);
    }

    div[data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 800 !important; }
    div[data-testid="stMetricValue"] { color: var(--cyan) !important; font-weight: 900 !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 0.45rem; }

    .stTabs [data-baseweb="tab"] {
        background: rgba(8, 18, 30, 0.94);
        border: 1px solid rgba(0, 229, 255, 0.16);
        border-radius: 14px 14px 0 0;
        color: #eafcff;
        padding: 0.55rem 1rem;
        font-weight: 800;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(0, 229, 255, 0.12) !important;
        color: white !important;
        border-color: rgba(0, 229, 255, 0.34) !important;
    }

    div[data-testid="stDataFrame"] {
        background: linear-gradient(180deg, rgba(8,18,30,0.96), rgba(10,21,35,0.94)) !important;
        border: 1px solid rgba(0, 229, 255, 0.20) !important;
        border-radius: 18px !important;
        overflow: hidden !important;
        box-shadow: 0 0 24px rgba(0, 229, 255, 0.08) !important;
    }

    div[data-testid="stPlotlyChart"] {
        background: linear-gradient(180deg, rgba(8,18,30,0.96), rgba(10,21,35,0.94)) !important;
        border: 1px solid rgba(0, 229, 255, 0.18) !important;
        border-radius: 18px !important;
        padding: 0.75rem !important;
        box-shadow: 0 0 24px rgba(0, 229, 255, 0.08) !important;
    }

    div[data-testid="stCodeBlock"] {
        background: linear-gradient(180deg, rgba(3,9,16,0.98), rgba(8,18,30,0.96)) !important;
        border: 1px solid rgba(124,255,178,0.18) !important;
        border-radius: 18px !important;
        overflow: hidden !important;
        box-shadow: 0 0 18px rgba(124,255,178,0.08) !important;
    }

    div[data-testid="stCodeBlock"] pre {
        background: transparent !important;
        color: #9DFFD1 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    pre, code {
        background-color: rgba(3,9,16,0.92) !important;
        color: #9DFFD1 !important;
        border-radius: 10px !important;
    }

    .terminal-box {
        background: linear-gradient(180deg, rgba(3, 9, 16, 0.98), rgba(8, 18, 30, 0.96));
        border: 1px solid rgba(124, 255, 178, 0.18);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        color: #9dffd1;
        font-size: 0.9rem;
        line-height: 1.55;
        box-shadow: 0 0 18px rgba(124, 255, 178, 0.08);
        white-space: pre-wrap;
    }

    .streamlit-expanderHeader { font-weight: 900 !important; color: #eafcff !important; }
    details { background: rgba(8,18,30,0.50) !important; border-radius: 16px !important; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 229, 255, 0.22); border-radius: 999px; }
    ::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.025); }

    @media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 700px) { .kpi-grid { grid-template-columns: 1fr; } }


    /* Six monitoring zones */
    .zone-title {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-family: 'Orbitron', sans-serif;
        font-weight: 800;
        letter-spacing: 0.45px;
        color: #EAFBFF;
        margin-bottom: 0.35rem;
    }

    .zone-tag {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 72px;
        padding: 0.28rem 0.6rem;
        border-radius: 999px;
        background: rgba(0, 229, 255, 0.10);
        border: 1px solid rgba(0, 229, 255, 0.28);
        color: #00E5FF;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 900;
        box-shadow: 0 0 14px rgba(0, 229, 255, 0.10);
    }

    .route-card {
        background: linear-gradient(180deg, rgba(3,9,16,0.98), rgba(8,18,30,0.94));
        border: 1px solid rgba(0,229,255,0.20);
        border-radius: 18px;
        padding: 1rem;
        margin: 0.4rem 0 0.9rem 0;
        box-shadow: 0 0 22px rgba(0,229,255,0.07);
    }

    .route-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 800;
        color: #7CFFB2;
        margin-bottom: 0.35rem;
    }

    .route-body {
        color: #9DC9D8;
        line-height: 1.55;
        font-size: 0.92rem;
    }

    .small-note {
        color: #6D9AAA;
        font-size: 0.82rem;
        line-height: 1.45;
        margin-top: 0.25rem;
    }

    .decoded-box {
        background: linear-gradient(180deg, rgba(3, 9, 16, 0.98), rgba(8, 18, 30, 0.96));
        border: 1px solid rgba(124, 255, 178, 0.28);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin: 0.75rem 0 1rem 0;
        color: #9DFFD1;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.94rem;
        line-height: 1.65;
        white-space: pre-wrap;
        word-break: break-word;
        box-shadow: 0 0 26px rgba(124, 255, 178, 0.10);
    }

    .decoded-label {
        font-family: 'Orbitron', sans-serif;
        color: #7CFFB2;
        font-weight: 900;
        letter-spacing: 0.4px;
        margin-bottom: 0.35rem;
    }

    .validation-box {
        background: rgba(0, 229, 255, 0.065);
        border: 1px solid rgba(0, 229, 255, 0.20);
        border-radius: 16px;
        padding: 0.85rem 1rem;
        margin: 0.65rem 0 1rem 0;
        color: #EAFBFF;
        line-height: 1.55;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-content">
            <h1 class="hero-title-glow" style="margin:0 0 0.45rem 0;">🛡️ CRYPTO AI — Enterprise Cybersecurity Dashboard</h1>
            <div class="hero-sub">
                Autonomous <strong>Sense–Reason–Act</strong> pipeline for suspicious encoded message analysis,
                cipher-state detection, explainability, forecasting, and secure API response.
            </div>
            <div style="margin-top:0.85rem;">
                <span class="pill">SOC Monitoring</span>
                <span class="pill">Cipher Intelligence</span>
                <span class="pill">SVM-RBF Live Inference</span>
                <span class="pill">SHAP XAI</span>
                <span class="pill">SHAP Cosine Reliability</span>
                <span class="pill">SARIMA Drift Watch</span>
                <span class="pill">ReAct Routing</span>
                <span class="pill">Explicit RAG Vote</span>
                <span class="pill">Hex + ASCII Support</span>
                <span class="pill">Dual Drift</span>
                <span class="pill">Outcome Tracking</span>
                <span class="pill">Autonomous Alerting</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🧭 Mission Control")

artifact_status = get_artifact_status()
artifacts_ready = artifact_status.get("artifacts_exist", False)

st.sidebar.markdown("### 1) Model Artifacts")
if artifacts_ready:
    st.sidebar.success("Local trained artifacts detected")
else:
    st.sidebar.error("Artifacts not found")

st.sidebar.caption(
    "Run once before live inference:\n"
    "`python train_local_pipeline.py`"
)

st.sidebar.markdown("### 2) Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload dataset or use data/raw/dataset.csv",
    type=["csv", "xlsx", "xls"],
)

try:
    preview_df = load_data(uploaded_file)
    st.sidebar.success(
        f"Dataset ready: {preview_df.shape[0]} rows × {preview_df.shape[1]} columns"
    )
except Exception as e:
    preview_df = None
    st.sidebar.warning(f"Preview unavailable: {e}")

st.sidebar.markdown("### 3) Live Threat Message")
messy_text = st.sidebar.text_area(
    "Suspicious text containing encoded/cipher token",
    value="",
    placeholder=(
        "Paste suspicious encoded message here...\n\n"
        "Base64 example:\n"
        "Please analyze this suspicious Base64 encoded message: VEhFIFNPQyBURUFNIERFVEVDVEVEIEEgSElHSCBSSVNLIEVYSElMVFJBVElPTiBDT01NQU5E and run it through CRYPTO AI.\n\n"
        "Base32 example:\n"
        "Please analyze this suspicious Base32 encoded message: JBSWY3DPEBLW64TMMQ====== and run it through CRYPTO AI.\n\n"
        "Morse example:\n"
        "Please analyze this suspicious Morse encoded message: .... . .-.. .-.. --- / .-- --- .-. .-.. -.. and run it through CRYPTO AI.\n\n"
        "Hex example:\n"
        "Please analyze this suspicious Hex encoded message: 54484520534f4320444554454354454420412048494444454e20434f4d4d414e44 and run it through CRYPTO AI.\n\n"
        "ASCII example:\n"
        "Please analyze this suspicious ASCII encoded message: 83 79 67 32 65 76 69 82 84 32 84 82 73 71 71 69 82 69 68 and run it through CRYPTO AI."
    ),
    height=205,
    key="enterprise_cyber_live_message_input_v3",
)

st.sidebar.markdown("### 4) Demo Controls")
execute_api_action = st.sidebar.checkbox("Allow API action / logging", value=True)
force_api_action = st.sidebar.checkbox("Force API action for demo", value=True)
enable_dynamic_reclustering = st.sidebar.checkbox(
    "Enable dynamic reclustering if drift is detected",
    value=True,
)
max_rows_per_type = st.sidebar.slider(
    "Training sample rows per cipher type",
    min_value=100,
    max_value=1200,
    value=500,
    step=100,
)

run_button = st.sidebar.button("▶ Launch Live Analysis", type="primary")

st.sidebar.markdown("### 5) Integrations")
if os.getenv("AZURE_OPENAI_API_KEY"):
    st.sidebar.success("LLM configured")
else:
    st.sidebar.warning("LLM missing — fallback explanation/report will be used")

if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
    st.sidebar.success("Telegram API configured")
else:
    st.sidebar.info("Telegram credentials missing — action will be logged but not sent")

# =========================================================
# ARCHITECTURE / MODE
# =========================================================

with st.expander("Architecture & Operating Mode", expanded=True):
    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown(
            """
### Sense
- Streamlit live input
- Dataset upload/default ingestion
- LLM/Regex extracts `cypher_text`
- Supports Base64, Base32, Morse, Hex, ASCII
- LLM does **not** classify/decode
"""
        )

    with a2:
        st.markdown(
            """
### Reason
- Saved local scaler + SVM
- SVM-RBF live classification
- SHAP explainability
- SHAP cosine reliability to cluster mean
- SHAP re-check loop
- Explicit RAG majority vote
- ReAct routing engine
- SARIMA + dual drift intelligence
"""
        )

    with a3:
        st.markdown(
            """
### Act
- Exact five-signal Intelligent Score
- Adaptive ReAct action
- Outcome tracking
- Outer-loop rebuild guard
- Telegram Bot API action
- Safe simulated log if credentials are missing
"""
        )

# =========================================================
# ARTIFACT CHECK
# =========================================================

if not artifacts_ready:
    st.error(
        "Saved local models are missing. Run the offline training command first:\n\n"
        "`python train_local_pipeline.py`"
    )

    json_debug_block("Missing Artifact Details", artifact_status, expanded=True)
    st.stop()

# =========================================================
# PRE-RUN OVERVIEW
# =========================================================

st.markdown('<div class="soc-card"><div class="soc-card-inner">', unsafe_allow_html=True)
st.subheader("📡 System Readiness")

st.markdown(
    f"""
    <div>
        {status_badge("ARTIFACTS READY", "good")}
        {status_badge("LIVE INFERENCE MODE", "good")}
        {status_badge("SVM NOT RETRAINED PER RUN", "good")}
        {status_badge("SUPPORTED: " + ", ".join(SUPPORTED_CIPHER_TYPES), "info")}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="kpi-grid">
    """
    + kpi_card("Artifacts", "READY", "Saved local models detected", "green")
    + kpi_card("Operating Mode", "LIVE", "Loads model from disk", "cyan")
    + kpi_card("SVM Training", "OFF", "No retraining during dashboard run", "purple")
    + kpi_card("Cipher Families", str(len(SUPPORTED_CIPHER_TYPES)), ", ".join(SUPPORTED_CIPHER_TYPES), "orange")
    + """
    </div>
    """,
    unsafe_allow_html=True,
)

if preview_df is not None:
    st.caption(
        f"Dataset preview: {preview_df.shape[0]} rows × {preview_df.shape[1]} columns"
    )
    st.dataframe(preview_df.head(5), use_container_width=True)
else:
    st.info("No preview dataset available, but saved training artifacts are ready.")

st.markdown("</div></div>", unsafe_allow_html=True)

# =========================================================
# WAIT BEFORE RUN
# =========================================================

if not run_button:
    st.info("Paste a suspicious encoded message in the sidebar, then click **Launch Live Analysis**.")
    st.stop()

if not str(messy_text or "").strip():
    st.warning("Please enter a suspicious encoded message before running the live analysis.")
    st.stop()

# =========================================================
# RUN PIPELINE
# =========================================================

with st.spinner("Running enterprise live threat analysis pipeline..."):
    try:
        result = run_crypto_ai_pipeline(
            messy_text=messy_text,
            uploaded_file=uploaded_file,
            execute_api_action=execute_api_action,
            force_api_action=force_api_action,
            enable_dynamic_reclustering=enable_dynamic_reclustering,
            max_rows_per_type=max_rows_per_type,
        )
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        raise

# =========================================================
# EXTRACT COMMON OUTPUTS
# =========================================================

def ensure_dict(value):
    return value if isinstance(value, dict) else {}


def short_text(value, limit: int = 180) -> str:
    value = str(value or "")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def yes_no(value) -> str:
    return "YES" if bool(value) else "NO"


def zone_header(zone_no: int, title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="zone-title">
            <span class="zone-tag">ZONE {zone_no}</span>
            <span>{title}</span>
        </div>
        <div class="small-note">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def as_dataframe(rows):
    return pd.DataFrame(rows if isinstance(rows, list) else [])


total_score = safe_float(result.get("total_action_score", 0.0))
level_text, level_kind = threat_level(total_score)

prediction = result.get("extracted_prediction", "N/A")
svm_original_prediction = result.get("svm_original_prediction", prediction)
confidence = safe_float(result.get("extracted_confidence", 0.0))
xai_score = safe_float(result.get("xai_score", 0.0))
action_required = bool(result.get("action_required", False))
decoded_plain_text = str(result.get("decoded_plain_text", "") or "")
decode_result = ensure_dict(result.get("decode_result", {}))
cipher_validation_result = ensure_dict(result.get("cipher_validation_result", {}))
validation_override = bool(cipher_validation_result.get("override_applied", False))
validation_confidence = safe_float(cipher_validation_result.get("deterministic_confidence", 0.0))

live_df = result.get("live_df", pd.DataFrame())
training_labeled_df = result.get("training_labeled_df", pd.DataFrame())

shap_result = ensure_dict(result.get("shap_result", {}))
shap_reliability_result = ensure_dict(
    result.get(
        "shap_reliability_result",
        shap_result.get("shap_reliability_result", {}),
    )
)
shap_recheck_result = ensure_dict(result.get("shap_recheck_result", {}))
shap_cosine_similarity = safe_float(
    result.get(
        "shap_cosine_similarity",
        shap_result.get(
            "shap_cosine_similarity",
            shap_reliability_result.get("shap_cosine_similarity", 0.0),
        ),
    )
)
shap_reliability_level = shap_reliability_result.get("reliability_level", "unknown")
shap_reliability_badge_kind = reliability_kind(shap_reliability_level, shap_cosine_similarity)
importance_df = shap_result.get("importance_df", pd.DataFrame())
shap_english_result = ensure_dict(result.get("shap_english_result", {}))

action_score = ensure_dict(result.get("action_score_result", {}))
score_signals = ensure_dict(action_score.get("signals", {}))

dynamic_severity_result = ensure_dict(result.get("dynamic_severity_result", {}))
dynamic_severity_score = safe_float(result.get("dynamic_severity_score", dynamic_severity_result.get("dynamic_severity", score_signals.get("S_class_severity", 0.0))))
dynamic_severity_level = result.get("dynamic_severity_level", dynamic_severity_result.get("severity_level", "N/A"))

sarima = ensure_dict(result.get("sarima_result", {}))
drift = ensure_dict(result.get("drift_result", {}))
dual_drift_result = ensure_dict(result.get("dual_drift_result", drift.get("dual_drift_result", {})))
outer_loop_result = ensure_dict(
    result.get("outer_react_loop_result")
    or result.get("outer_loop_result")
    or result.get("outer_rebuild_result")
    or result.get("outer_react_result")
    or {}
)

rag_result = ensure_dict(result.get("rag_result", {}))
rag_decision_result = ensure_dict(result.get("rag_decision_result", {}))
react_routing_result = ensure_dict(result.get("react_routing_result", {}))
adaptive_action = ensure_dict(result.get("adaptive_action", react_routing_result.get("adaptive_action", {})))
api_result = ensure_dict(result.get("api_result", {}))
api_channel = str(api_result.get("channel", "telegram" if api_result.get("telegram_attempted") else "N/A"))
telegram_attempted = bool(api_result.get("telegram_attempted", False))
telegram_sent = bool(api_result.get("telegram_sent", False))
telegram_parts = api_result.get("message_parts", api_result.get("telegram_message_parts", api_result.get("parts_sent", "N/A")))
action_logged = bool(api_result.get("action_logged", False))
api_error = str(api_result.get("error", "") or "")
api_message = str(api_result.get("message", "") or "")
outcome_tracking_result = ensure_dict(
    result.get("outcome_tracking_result")
    or result.get("outcome_tracker_result")
    or result.get("outcome_result")
    or result.get("between_cycle_adaptation_result")
    or {}
)

# =========================================================
# TOP KPI STRIP
# =========================================================

st.markdown('<div class="soc-card"><div class="soc-card-inner">', unsafe_allow_html=True)
st.subheader("🚨 Threat Intelligence Summary")

badge_kind = "critical" if level_kind in ["critical", "high"] else "warn" if level_kind == "medium" else "good"
react_route = react_routing_result.get("route_id", result.get("final_react_route", "N/A"))
rag_case_type = rag_decision_result.get("case_type", "N/A")
dual_drift_flag = bool(dual_drift_result.get("dual_drift_detected", dual_drift_result.get("outer_loop_required", False)))
outer_triggered = bool(outer_loop_result.get("triggered", outer_loop_result.get("outer_loop_required", False)))

st.markdown(
    f"""
    <div style="margin-bottom:0.7rem;">
        {status_badge("THREAT LEVEL: " + level_text, badge_kind)}
        {status_badge("PREDICTION: " + str(prediction), "info")}
        {status_badge("VALIDATION OVERRIDE" if validation_override else "VALIDATED TYPE", "warn" if validation_override else "good")}
        {status_badge("DECODED" if decoded_plain_text else "NOT DECODED", "good" if decoded_plain_text else "warn")}
        {status_badge("FINAL ROUTE: " + str(react_route), "warn" if "REVIEW" in str(react_route).upper() or "RECHECK" in str(react_route).upper() else "good" if action_required else "info")}
        {status_badge("ACTION REQUIRED" if action_required else "MONITOR / REVIEW", "bad" if action_required else "good")}
        {status_badge("SHAP RELIABILITY: " + str(shap_reliability_level).upper(), shap_reliability_badge_kind)}
        {status_badge("DUAL DRIFT" if dual_drift_flag else "NO DUAL DRIFT", "bad" if dual_drift_flag else "good")}
        {status_badge("OUTER LOOP TRIGGERED" if outer_triggered else "OUTER LOOP STABLE", "bad" if outer_triggered else "good")}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="kpi-grid">
    """
    + kpi_card("Predicted Cipher", str(prediction), "Saved SVM-RBF live inference", "cyan")
    + kpi_card("SVM Confidence", f"{confidence:.4f}", "Gap(SVM) / classifier certainty", "green")
    + kpi_card("SHAP Cosine", f"{shap_cosine_similarity:.4f}", "Reliability to cluster mean", "purple")
    + kpi_card("Total Action Score", f"{total_score:.4f}", f"Threat level: {level_text}", "red" if action_required else "orange")
    + """
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 🔓 Decoded Plain Text")
if decoded_plain_text:
    st.markdown(
        f"""
        <div class="decoded-box">
            <div class="decoded-label">DETERMINISTIC DECODER OUTPUT</div>
            {decoded_plain_text}
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning(decode_result.get("message", "The cipher was not decoded in this run."))

st.markdown("</div></div>", unsafe_allow_html=True)

# =========================================================
# SIX MONITORING ZONES
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Zone 1 · Cluster Health",
        "Zone 2 · Live Scoring",
        "Zone 3 · Action Dispatch",
        "Zone 4 · SHAP Reliability",
        "Zone 5 · Outcome Tracking",
        "Zone 6 · System Alerts",
        "Mission Timeline",
        "Technical Audit",
    ]
)

# =========================================================
# ZONE 1 — CLUSTER HEALTH
# =========================================================

with tab1:
    st.markdown('<div class="soc-card"><div class="soc-card-inner">', unsafe_allow_html=True)
    zone_header(1, "Cluster Health", "DBSCAN/FCM live context, membership strength, nearest training-row assignment, and centroid-movement drift evidence.")

    z1a, z1b, z1c, z1d = st.columns(4)

    dbscan_cluster = live_df.get("dbscan_cluster", pd.Series(["N/A"])).iloc[0] if isinstance(live_df, pd.DataFrame) and not live_df.empty else "N/A"
    fcm_cluster = live_df.get("fcm_cluster", pd.Series(["N/A"])).iloc[0] if isinstance(live_df, pd.DataFrame) and not live_df.empty else "N/A"
    fcm_membership = live_df.get("fcm_membership_score", pd.Series([0.0])).iloc[0] if isinstance(live_df, pd.DataFrame) and not live_df.empty else 0.0
    nearest_distance = live_df.get("nearest_training_distance", pd.Series([0.0])).iloc[0] if isinstance(live_df, pd.DataFrame) and not live_df.empty else 0.0

    z1a.metric("DBSCAN Cluster", dbscan_cluster)
    z1b.metric("FCM Cluster", fcm_cluster)
    z1c.metric("FCM μ(predicted)", fmt_score(fcm_membership))
    z1d.metric("Nearest Distance", fmt_score(nearest_distance))

    live_cols = [
        c for c in [
            "cypher_text",
            "dbscan_cluster",
            "fcm_cluster",
            "fcm_membership_score",
            "risk_label",
            "business_risk_label",
            "svm_original_prediction",
            "svm_predicted_cipher_type",
            "validated_cipher_type",
            "cipher_validation_override",
            "cipher_validation_confidence",
            "svm_confidence",
            "nearest_training_row_index",
            "nearest_training_distance",
        ]
        if isinstance(live_df, pd.DataFrame) and not live_df.empty and c in live_df.columns
    ]

    st.markdown("### Live Cluster / Label Context")
    if isinstance(live_df, pd.DataFrame) and not live_df.empty and live_cols:
        st.dataframe(live_df[live_cols], use_container_width=True)
    else:
        st.info("No live cluster/label preview available.")

    st.markdown("### Centroid Movement Drift Evidence")
    centroid_rows = [
        {"Item": "Centroid Movement Detected", "Value": yes_no(dual_drift_result.get("centroid_movement_detected", False))},
        {"Item": "Centroid Distance", "Value": dual_drift_result.get("centroid_distance", dual_drift_result.get("live_centroid_distance", "N/A"))},
        {"Item": "Centroid Threshold", "Value": dual_drift_result.get("centroid_threshold", "N/A")},
        {"Item": "Dual Drift Decision", "Value": dual_drift_result.get("decision", "N/A")},
    ]
    st.dataframe(pd.DataFrame(centroid_rows), use_container_width=True)

    json_debug_block("Technical Debug: Dual Drift Details", dual_drift_result, expanded=False)
    st.markdown("</div></div>", unsafe_allow_html=True)

# =========================================================
# ZONE 2 — LIVE SCORING FEED
# =========================================================

with tab2:
    st.markdown('<div class="soc-card"><div class="soc-card-inner">', unsafe_allow_html=True)
    zone_header(2, "Live Scoring Feed", "Exact five-signal Intelligent Score: S, Gap(SVM), μ(predicted), ARIMA, and SHAP reliability.")

    z2a, z2b, z2c, z2d, z2e = st.columns(5)
    z2a.metric("S Dynamic Severity", fmt_score(score_signals.get("S_class_severity", dynamic_severity_score)), str(dynamic_severity_level))
    z2b.metric("Gap(SVM)", fmt_score(score_signals.get("Gap_SVM", confidence)))
    z2c.metric("μ(predicted)", fmt_score(score_signals.get("Mu_predicted_FCM", fcm_membership)))
    z2d.metric("ARIMA Trend", fmt_score(score_signals.get("ARIMA_trend", sarima.get("forecast_risk_score", 0.0))))
    z2e.metric("SHAP Reliability", fmt_score(score_signals.get("SHAP_reliability", shap_cosine_similarity)))

    st.markdown("### Exact Five-Signal Score Breakdown")
    score_rows = [
        {"Signal": "S — Dynamic Class Severity", "Value": score_signals.get("S_class_severity", dynamic_severity_score), "Meaning": "Case severity computed dynamically from prediction, text, SHAP, FCM, ARIMA, and drift."},
        {"Signal": "Gap(SVM)", "Value": score_signals.get("Gap_SVM", confidence), "Meaning": "Classifier confidence/separation signal."},
        {"Signal": "μ(predicted) — FCM Membership", "Value": score_signals.get("Mu_predicted_FCM", fcm_membership), "Meaning": "Geometric membership from FCM context."},
        {"Signal": "ARIMA Trend", "Value": score_signals.get("ARIMA_trend", sarima.get("forecast_risk_score", 0.0)), "Meaning": "Forecast/cluster trend multiplier."},
        {"Signal": "SHAP Reliability", "Value": score_signals.get("SHAP_reliability", shap_cosine_similarity), "Meaning": "Cosine similarity to cluster mean."},
        {"Signal": "XAI Score", "Value": action_score.get("xai_score", xai_score), "Meaning": "SVM Confidence × SHAP Importance."},
        {"Signal": "Total Action Score", "Value": action_score.get("total_action_score", total_score), "Meaning": "Final composite decision score."},
    ]
    st.dataframe(pd.DataFrame(score_rows), use_container_width=True)

    st.markdown("### Dynamic Severity Audit Trail")
    severity_rows = []
    for idx, reason in enumerate(dynamic_severity_result.get("reason_codes", []), start=1):
        severity_rows.append({"#": idx, "Reason": reason})
    if severity_rows:
        st.dataframe(pd.DataFrame(severity_rows), use_container_width=True)
    else:
        st.info("No dynamic severity reasons available.")

    json_debug_block("Technical Debug: Dynamic Severity Result", dynamic_severity_result, expanded=False)
    st.markdown("</div></div>", unsafe_allow_html=True)

# =========================================================
# ZONE 3 — ACTION DISPATCH
# =========================================================

with tab3:
    st.markdown('<div class="soc-card"><div class="soc-card-inner">', unsafe_allow_html=True)
    zone_header(3, "Action Dispatch", "Final ReAct route, explicit RAG vote, adaptive action, Telegram API status, timing, and intensity.")

    route_kind = "bad" if action_required else "warn" if str(react_routing_result.get("recommended_route", "")).lower() in {"human_review", "shap_recheck"} else "good"
    st.markdown(
        f"""
        <div style="margin-bottom:0.7rem;">
            {status_badge("FINAL ROUTE: " + str(react_routing_result.get("route_id", "N/A")), route_kind)}
            {status_badge("RAG CASE: " + str(rag_decision_result.get("case_type", "N/A")), "info")}
            {status_badge("RAG USED" if rag_decision_result.get("rag_used") else "RAG LOGGED / NOT BORDERLINE", "warn" if rag_decision_result.get("rag_used") else "good")}
            {status_badge("TELEGRAM SENT" if telegram_sent else "TELEGRAM NOT SENT", "good" if telegram_sent else "bad" if telegram_attempted else "warn")}
            {status_badge("CHANNEL: " + api_channel.upper(), "info")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    z3a, z3b, z3c, z3d = st.columns(4)
    z3a.metric("Recommended Route", react_routing_result.get("recommended_route", "N/A"))
    z3b.metric("Action Type", adaptive_action.get("action_type", "N/A"))
    z3c.metric("Timing", adaptive_action.get("timing", "N/A"))
    z3d.metric("Intensity", adaptive_action.get("intensity", "N/A"))

    st.markdown("### 🔓 Decoded Plain Text")
    if decoded_plain_text:
        st.markdown(
            f"""
            <div class="decoded-box">
                <div class="decoded-label">{str(prediction).upper()} → PLAINTEXT</div>
                {decoded_plain_text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(decode_result.get("message", "No decoded output available."))

    st.markdown("### Cipher Validation Guardrail")
    st.markdown(
        f"""
        <div class="validation-box">
            <strong>Original SVM Prediction:</strong> {svm_original_prediction}<br/>
            <strong>Validated Final Prediction:</strong> {prediction}<br/>
            <strong>Override Applied:</strong> {yes_no(validation_override)}<br/>
            <strong>Deterministic Confidence:</strong> {validation_confidence:.4f}<br/>
            <strong>Reason:</strong> {short_text(cipher_validation_result.get('reason', 'No validation result available.'), 900)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Adaptive ReAct Action")
    st.markdown(
        f"""
        <div class="route-card">
            <div class="route-title">{react_routing_result.get('route_name', 'Final ReAct Route')}</div>
            <div class="route-body">{short_text(react_routing_result.get('reason', 'No ReAct reason available.'), 800)}</div>
            <br/>
            <div class="route-body"><strong>Adaptive message:</strong> {short_text(adaptive_action.get('personalized_message', 'No adaptive message available.'), 1200)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Explicit RAG Majority Vote")
    rag_rows = [
        {"Item": "Case Type", "Value": rag_decision_result.get("case_type", "N/A")},
        {"Item": "RAG Used", "Value": yes_no(rag_decision_result.get("rag_used", False))},
        {"Item": "Majority Vote", "Value": rag_decision_result.get("majority_vote", "N/A")},
        {"Item": "Vote Counts", "Value": rag_decision_result.get("vote_counts", {})},
        {"Item": "Tie Detected", "Value": yes_no(rag_decision_result.get("tie_detected", False))},
        {"Item": "Zero Success", "Value": yes_no(rag_decision_result.get("zero_success", False))},
        {"Item": "Recommended Route", "Value": rag_decision_result.get("recommended_route", "N/A")},
        {"Item": "ARIMA Alignment", "Value": ensure_dict(rag_decision_result.get("arima_alignment", {})).get("arima_alignment", "N/A")},
        {"Item": "Reason", "Value": rag_decision_result.get("reason", "")},
    ]
    st.dataframe(pd.DataFrame(rag_rows), use_container_width=True)

    evidence_rows = rag_decision_result.get("evidence_rows", [])
    if evidence_rows:
        st.markdown("### RAG Evidence Rows")
        st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True)

    st.markdown("### Telegram API Action")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Action Required", yes_no(action_required))
    a2.metric("API Success", yes_no(api_result.get("success", False)))
    a3.metric("Channel", api_channel)
    a4.metric("Telegram Sent", yes_no(telegram_sent))
    a5.metric("Action Logged", yes_no(action_logged))

    st.markdown(
        f"""
        <div style="margin-bottom:0.7rem;">
            {status_badge("CHANNEL: " + api_channel.upper(), "info")}
            {status_badge("TELEGRAM ATTEMPTED" if telegram_attempted else "TELEGRAM NOT ATTEMPTED", "good" if telegram_attempted else "warn")}
            {status_badge("TELEGRAM SENT" if telegram_sent else "TELEGRAM NOT SENT", "good" if telegram_sent else "bad" if telegram_attempted else "warn")}
            {status_badge("ACTION LOGGED" if action_logged else "ACTION NOT LOGGED", "good" if action_logged else "warn")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_rows = [
        {"Item": "Channel", "Value": api_channel},
        {"Item": "Telegram Attempted", "Value": yes_no(telegram_attempted)},
        {"Item": "Telegram Sent", "Value": yes_no(telegram_sent)},
        {"Item": "Message Parts", "Value": telegram_parts},
        {"Item": "Action Logged", "Value": yes_no(action_logged)},
        {"Item": "API Success", "Value": yes_no(api_result.get("success", False))},
        {"Item": "Message", "Value": api_message},
        {"Item": "Error", "Value": api_error or "N/A"},
    ]
    st.dataframe(pd.DataFrame(api_rows), use_container_width=True)

    if telegram_sent:
        st.success(api_message or "Telegram alert sent successfully.")
    elif telegram_attempted:
        st.error(api_message or "Telegram was attempted but was not sent.")
    else:
        st.warning(api_message or "Telegram was not attempted. Check execute_api_action / force_api_action controls.")

    if api_error:
        st.error(f"Telegram/API error: {api_error}")

    json_debug_block("Technical Debug: API / Telegram Result", api_result, expanded=True)

    st.markdown("### Final Generated Report")
    st.text_area("Final Report", value=result.get("report", ""), height=260, key="enterprise_final_report_output")

    json_debug_block("Technical Debug: RAG Decision Result", rag_decision_result, expanded=False)
    json_debug_block("Technical Debug: ReAct Routing Result", react_routing_result, expanded=False)
    st.markdown("</div></div>", unsafe_allow_html=True)

# =========================================================
# ZONE 4 — SHAP RELIABILITY
# =========================================================

with tab4:
    st.markdown('<div class="soc-card"><div class="soc-card-inner">', unsafe_allow_html=True)
    zone_header(4, "SHAP Reliability", "SHAP importance, cosine similarity to cluster mean, reliability level, and the formal SHAP re-check loop.")

    z4a, z4b, z4c, z4d, z4e = st.columns(5)
    z4a.metric("SHAP Importance", fmt_score(shap_result.get("shap_score", 0.0)))
    z4b.metric("SHAP Cosine", fmt_score(shap_cosine_similarity))
    z4c.metric("Reliability Level", str(shap_reliability_level).upper())
    z4d.metric("Re-check Triggered", yes_no(shap_recheck_result.get("recheck_triggered", False)))
    z4e.metric("Final Cosine", fmt_score(shap_recheck_result.get("final_shap_cosine_similarity", shap_cosine_similarity)))

    st.caption("SHAP Reliability is calculated as cosine similarity between the live SHAP vector and the matched cluster mean SHAP-style vector.")

    rel_cols = st.columns(4)
    rel_cols[0].metric("Cluster Column", shap_reliability_result.get("cluster_column", "N/A"))
    rel_cols[1].metric("Cluster Value", shap_reliability_result.get("cluster_value", "N/A"))
    rel_cols[2].metric("Matched Rows", shap_reliability_result.get("matched_rows", "N/A"))
    rel_cols[3].metric("Fallback Used", str(shap_reliability_result.get("fallback_used", False)))

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### SHAP Top Features")
        if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
            st.dataframe(importance_df.head(20), use_container_width=True)
            fig = px.bar(importance_df.head(15), x="feature", y="shap_importance", title="Top SHAP Feature Importances")
            fig.update_layout(xaxis_tickangle=-45, height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SHAP feature importance table available.")

        st.markdown("### SHAP Re-check Loop")
        recheck_rows = [
            {"Item": "Triggered", "Value": yes_no(shap_recheck_result.get("recheck_triggered", False))},
            {"Item": "Original Cosine", "Value": shap_recheck_result.get("original_shap_cosine_similarity", shap_cosine_similarity)},
            {"Item": "Final Cosine", "Value": shap_recheck_result.get("final_shap_cosine_similarity", shap_cosine_similarity)},
            {"Item": "Final Reliability Level", "Value": shap_recheck_result.get("final_reliability_level", shap_reliability_level)},
            {"Item": "Route After Re-check", "Value": shap_recheck_result.get("route_after_recheck", "N/A")},
            {"Item": "Reason", "Value": shap_recheck_result.get("reason", "N/A")},
        ]
        st.dataframe(pd.DataFrame(recheck_rows), use_container_width=True)

    with right:
        st.markdown("### SHAP English Explanation Node")
        st.text_area(
            "English Explanation",
            value=shap_english_result.get("explanation", "No SHAP English explanation available."),
            height=460,
            key="enterprise_shap_english_explanation_output",
        )
        top_features = shap_english_result.get("top_features", [])
        if top_features:
            st.markdown("**Top Interpreted Features**")
            st.dataframe(pd.DataFrame(top_features), use_container_width=True)

    json_debug_block("Technical Debug: SHAP Reliability Result", shap_reliability_result, expanded=False)
    json_debug_block("Technical Debug: SHAP Re-check Result", shap_recheck_result, expanded=False)
    st.markdown("</div></div>", unsafe_allow_html=True)

# =========================================================
# ZONE 5 — OUTCOME TRACKING
# =========================================================

with tab5:
    st.markdown('<div class="soc-card"><div class="soc-card-inner">', unsafe_allow_html=True)
    zone_header(5, "Outcome Tracking", "Action outcome logging and between-cycle adaptation recommendations.")

    z5a, z5b, z5c, z5d = st.columns(4)
    z5a.metric("Outcome Logged", yes_no(outcome_tracking_result.get("logged", outcome_tracking_result.get("action_logged", False))))
    z5b.metric("Outcome Status", outcome_tracking_result.get("outcome_status", outcome_tracking_result.get("status", "N/A")))
    z5c.metric("Adaptation", outcome_tracking_result.get("adaptation_recommendation", outcome_tracking_result.get("recommendation", "N/A")))
    z5d.metric("Policy Bias", outcome_tracking_result.get("bias", outcome_tracking_result.get("policy_bias", "N/A")))

    st.markdown("### Current Case Outcome")
    outcome_rows = [
        {"Item": "Case ID", "Value": outcome_tracking_result.get("case_id", "N/A")},
        {"Item": "Prediction", "Value": prediction},
        {"Item": "Total Score", "Value": total_score},
        {"Item": "ReAct Route", "Value": react_routing_result.get("route_id", "N/A")},
        {"Item": "Adaptive Action", "Value": adaptive_action.get("action_type", "N/A")},
        {"Item": "API Success", "Value": yes_no(api_result.get("success", False))},
        {"Item": "Telegram Sent", "Value": yes_no(telegram_sent)},
        {"Item": "API Channel", "Value": api_channel},
        {"Item": "Outcome Reason", "Value": outcome_tracking_result.get("reason", "N/A")},
    ]
    st.dataframe(pd.DataFrame(outcome_rows), use_container_width=True)

    st.markdown("### Outcome Log")
    outcome_log_path = Path("artifacts/outcome_tracking_log.csv")
    if outcome_log_path.exists():
        try:
            outcome_log_df = pd.read_csv(outcome_log_path).tail(20)
            st.dataframe(outcome_log_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not read outcome tracking log: {e}")
    else:
        st.info("No outcome tracking log found yet. It will be created after the first live run with the updated pipeline.")

    json_debug_block("Technical Debug: Outcome Tracking Result", outcome_tracking_result, expanded=False)
    st.markdown("</div></div>", unsafe_allow_html=True)

# =========================================================
# ZONE 6 — SYSTEM ALERTS
# =========================================================

with tab6:
    st.markdown('<div class="soc-card"><div class="soc-card-inner">', unsafe_allow_html=True)
    zone_header(6, "System Alerts", "SARIMA drift, dual drift signals, outer ReAct rebuild loop, rollback/acceptance status, and system health.")

    z6a, z6b, z6c, z6d = st.columns(4)
    z6a.metric("Forecast Risk", fmt_score(sarima.get("forecast_risk_score", 0.0)))
    z6b.metric("Drift Score", fmt_score(drift.get("drift_score", 0.0)))
    z6c.metric("Dual Drift", yes_no(dual_drift_flag))
    z6d.metric("Outer Loop", "TRIGGERED" if outer_triggered else "STABLE")

    if drift.get("drift_detected") or dual_drift_flag or outer_triggered:
        st.error("System alert condition detected. Review drift and outer-loop details below.")
    else:
        st.success("No critical system alert detected in this run.")

    st.markdown("### Forecast / SARIMA")
    forecast_df = sarima.get("forecast_df", pd.DataFrame())
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Forecast Engine", sarima.get("engine", "unknown"))
    f2.metric("Forecast Risk Score", fmt_score(sarima.get("forecast_risk_score", 0.0)))
    f3.metric("Expected Change %", fmt_score(sarima.get("expected_change_pct", 0.0)))
    f4.metric("Drift Recommendation", short_text(drift.get("recommendation", "N/A"), 48))

    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        st.dataframe(forecast_df, use_container_width=True)
        if {"forecast_date", "forecast_value"}.issubset(forecast_df.columns):
            fig = px.line(forecast_df, x="forecast_date", y="forecast_value", title="SARIMA Forecast")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dual Drift Signals")
    dual_rows = [
        {"Signal": "Gap(SVM) Trend Decline", "Value": yes_no(dual_drift_result.get("gap_svm_decline_detected", False))},
        {"Signal": "Centroid Movement Drift", "Value": yes_no(dual_drift_result.get("centroid_movement_detected", False))},
        {"Signal": "Dual Drift Detected", "Value": yes_no(dual_drift_result.get("dual_drift_detected", dual_drift_result.get("outer_loop_required", False)))},
        {"Signal": "Outer Loop Required", "Value": yes_no(dual_drift_result.get("outer_loop_required", False))},
        {"Signal": "Decision", "Value": dual_drift_result.get("decision", "N/A")},
    ]
    st.dataframe(pd.DataFrame(dual_rows), use_container_width=True)

    st.markdown("### Outer ReAct Rebuild Loop")
    outer_rows = [
        {"Item": "Triggered", "Value": yes_no(outer_loop_result.get("triggered", False))},
        {"Item": "Decision", "Value": outer_loop_result.get("decision", "N/A")},
        {"Item": "Accepted Rebuild", "Value": yes_no(outer_loop_result.get("accepted", outer_loop_result.get("accepted_rebuild", False)))},
        {"Item": "Rollback", "Value": yes_no(outer_loop_result.get("rollback", outer_loop_result.get("rollback_triggered", False)))},
        {"Item": "Actions", "Value": outer_loop_result.get("actions", outer_loop_result.get("actions_performed", "N/A"))},
        {"Item": "Reason", "Value": outer_loop_result.get("reason", outer_loop_result.get("status", "N/A"))},
    ]
    st.dataframe(pd.DataFrame(outer_rows), use_container_width=True)

    outer_audit_path = Path("artifacts/outer_react_loop_audit.json")
    if outer_audit_path.exists():
        with st.expander("Latest Outer Loop Audit File", expanded=False):
            try:
                st.code(outer_audit_path.read_text(encoding="utf-8"), language="json")
            except Exception as e:
                st.warning(f"Could not read outer loop audit file: {e}")

    json_debug_block("Technical Debug: Drift Result", drift, expanded=False)
    json_debug_block("Technical Debug: Dual Drift Result", dual_drift_result, expanded=False)
    json_debug_block("Technical Debug: Outer Loop Result", outer_loop_result, expanded=False)
    st.markdown("</div></div>", unsafe_allow_html=True)

# =========================================================
# MISSION TIMELINE
# =========================================================

with tab7:
    st.markdown('<div class="soc-card"><div class="soc-card-inner">', unsafe_allow_html=True)
    st.subheader("⏱️ Mission Timeline")

    for i, item in enumerate(result.get("timeline", []), start=1):
        label = f"{i}. {item.get('step', 'Unknown Step')}"
        details = item.get("details", "")
        if item.get("status") == "success":
            st.success(label)
        elif item.get("status") == "warning":
            st.warning(label)
        elif item.get("status") == "error":
            st.error(label)
        else:
            st.info(label)
        if details:
            st.caption(details)

    st.markdown("</div></div>", unsafe_allow_html=True)

# =========================================================
# TECHNICAL AUDIT
# =========================================================

with tab8:
    st.markdown('<div class="soc-card"><div class="soc-card-inner">', unsafe_allow_html=True)
    st.subheader("💾 Technical Audit")

    meta = ensure_dict(result.get("artifacts_metadata", {}))
    artifact_state = ensure_dict(result.get("artifact_status", artifact_status))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Saved Artifacts", "READY")
    m2.metric("Training Rows", meta.get("training_rows", meta.get("training_rows_used_for_svm", "N/A")))
    m3.metric("SVM Accuracy", round(safe_float(meta.get("svm_accuracy", 0.0)), 4))
    m4.metric("Retrained This Run", "NO" if not result.get("svm_retrained_this_run", False) else "YES")

    st.markdown("### Artifact Summary")
    summary_df = pd.DataFrame(
        [
            {"Item": "Artifacts Ready", "Value": "YES"},
            {"Item": "Training Rows", "Value": meta.get("training_rows", meta.get("training_rows_used_for_svm", "N/A"))},
            {"Item": "SVM Accuracy", "Value": round(safe_float(meta.get("svm_accuracy", 0.0)), 4)},
            {"Item": "SVM Classes", "Value": ", ".join(meta.get("svm_classes", [])) if isinstance(meta.get("svm_classes", []), list) else meta.get("svm_classes", "N/A")},
            {"Item": "Original SVM Prediction", "Value": svm_original_prediction},
            {"Item": "Validated Final Prediction", "Value": prediction},
            {"Item": "Cipher Validation Override", "Value": yes_no(validation_override)},
            {"Item": "Decode Success", "Value": yes_no(decode_result.get("success", False))},
            {"Item": "Decoded Plain Text", "Value": short_text(decoded_plain_text, 260)},
            {"Item": "FCM FPC", "Value": round(safe_float(meta.get("fcm_fpc", 0.0)), 4)},
            {"Item": "Forecast Target", "Value": meta.get("forecast_target", "N/A")},
            {"Item": "SHAP Cosine Reliability", "Value": round(shap_cosine_similarity, 4)},
            {"Item": "SHAP Reliability Level", "Value": shap_reliability_level},
            {"Item": "Final ReAct Route", "Value": react_routing_result.get("route_id", "N/A")},
            {"Item": "RAG Case Type", "Value": rag_decision_result.get("case_type", "N/A")},
            {"Item": "Telegram Channel", "Value": api_channel},
            {"Item": "Telegram Attempted", "Value": yes_no(telegram_attempted)},
            {"Item": "Telegram Sent", "Value": yes_no(telegram_sent)},
            {"Item": "Telegram Message Parts", "Value": telegram_parts},
            {"Item": "API Action Message", "Value": short_text(api_message, 260)},
            {"Item": "API Action Error", "Value": short_text(api_error, 260) if api_error else "N/A"},
            {"Item": "Dual Drift Decision", "Value": dual_drift_result.get("decision", "N/A")},
        ]
    )
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("### Command Reference")
    st.markdown(
        """
        <div class="terminal-box">python train_local_pipeline.py

python -m streamlit run app.py</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Export")
    labeled_df = result.get("labeled_df", pd.DataFrame())
    if isinstance(labeled_df, pd.DataFrame) and not labeled_df.empty:
        st.download_button(
            label="Download Final Combined Labeled CSV",
            data=labeled_df.to_csv(index=False).encode("utf-8"),
            file_name="crypto_ai_final_output.csv",
            mime="text/csv",
        )
    else:
        st.info("No labeled dataframe available to export.")

    if Path(ACTION_LOG_PATH).exists():
        st.markdown("### Latest Telegram/API Action Logs")
        try:
            action_logs_df = pd.read_csv(ACTION_LOG_PATH).tail(10)
            st.dataframe(action_logs_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not read action log: {e}")
    else:
        st.info("No action log file found yet.")

    json_debug_block("Technical Debug: API / Telegram Result", api_result, expanded=True)
    json_debug_block("Technical Debug: Cipher Validation Result", cipher_validation_result, expanded=False)
    json_debug_block("Technical Debug: Decode Result", decode_result, expanded=False)
    json_debug_block("Technical Debug: Full Artifact Metadata", meta, expanded=False)
    json_debug_block("Technical Debug: Full Artifact Status", artifact_state, expanded=False)
    json_debug_block("Technical Debug: Full Result Keys", {"keys": sorted(list(result.keys()))}, expanded=False)
    st.markdown("</div></div>", unsafe_allow_html=True)
