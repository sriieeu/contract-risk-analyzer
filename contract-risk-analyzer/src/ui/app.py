"""
Contract Risk Analyzer — Streamlit UI

Run with:
    streamlit run src/ui/app.py
"""

import sys
import os
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System — Warm Cream / White Editorial Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,800;0,900;1,700&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Design Tokens ── */
:root {
    --bg:           #FAF7F3;
    --bg-soft:      #F0E4D3;
    --bg-card:      #F0E4D3;
    --bg-cream:     #F0E4D3;

    --ink:          #111111;
    --ink-mid:      #111111;
    --ink-soft:     #333333;

    --border:       #DCC5B2;
    --border-strong:#D9A299;

    --red:          #D9A299;
    --red-soft:     #DCC5B2;
    --red-mid:      #DCC5B2;

    --gold:         #DCC5B2;
    --gold-soft:    #F0E4D3;
    --gold-mid:     #DCC5B2;

    --critical:     #991b1b;
    --critical-bg:  #fef2f2;
    --critical-bdr: #fecaca;
    --high:         #c2410c;
    --high-bg:      #fff7ed;
    --high-bdr:     #fed7aa;
    --medium:       #92400e;
    --medium-bg:    #fffbeb;
    --medium-bdr:   #fde68a;
    --low:          #166534;
    --low-bg:       #f0fdf4;
    --low-bdr:      #bbf7d0;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--ink);
    background: var(--bg) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-cream) !important;
    border-right: 1px solid var(--border-strong) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: var(--ink-mid) !important; }

.sidebar-brand {
    padding: 32px 24px 22px;
    border-bottom: 1px solid var(--border-strong);
    margin-bottom: 4px;
}
.sidebar-brand-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
    line-height: 1.2;
    margin: 0 0 6px;
}
.sidebar-brand-sub {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9.5px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--ink-soft) !important;
    margin: 0;
}

/* Nav */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 2px !important;
    display: flex;
    flex-direction: column;
    padding: 8px 0;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 10px 24px !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    color: var(--ink-mid) !important;
    cursor: pointer;
    transition: all 0.15s;
    border-left: 3px solid transparent !important;
    letter-spacing: 0.02em;
    background: transparent !important;
}
/* Hide the default radio circle */
section[data-testid="stSidebar"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] {
    margin-left: 0 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label input {
    display: none !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: var(--bg-soft) !important;
    color: var(--ink) !important;
    border-left-color: var(--border-strong) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: var(--bg-soft) !important;
    color: var(--ink) !important;
    border-left-color: var(--red) !important;
    font-weight: 500 !important;
}

.sidebar-meta {
    padding: 20px 24px;
    border-top: 1px solid var(--border);
    margin-top: 8px;
}
.sidebar-meta-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--ink-soft) !important;
    margin-bottom: 10px;
    font-weight: 500;
}
.sidebar-meta-item {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: var(--ink-mid) !important;
    margin: 5px 0;
    line-height: 1.5;
}
.sidebar-meta-item code {
    background: var(--bg-soft);
    border: 1px solid var(--border);
    padding: 1px 7px;
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    color: var(--red) !important;
}

/* ── Main canvas ── */
.main .block-container {
    padding: 0 52px 60px;
    max-width: 1240px;
    background: var(--bg);
}

/* ── Page header ── */
.page-header {
    padding: 40px 0 28px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 36px;
}
.page-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 900;
    color: var(--ink);
    margin: 0 0 8px;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.page-header h1 .accent { color: var(--red); font-style: italic; }
.page-header p {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--ink-soft);
    margin: 0;
    line-height: 1.75;
    letter-spacing: 0.01em;
}

/* ── Options panel ── */
.options-panel {
    background: var(--bg-soft);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 22px 24px;
}
.options-panel h4 {
    font-family: 'Playfair Display', serif;
    font-size: 14px;
    font-weight: 800;
    color: var(--ink);
    margin: 0 0 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    letter-spacing: -0.01em;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: var(--border);
    margin: 28px 0;
}

/* ── Section label ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Score ring ── */
.score-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 16px 0;
}
.score-ring {
    width: 100px; height: 100px;
    border-radius: 50%;
    border-width: 5px;
    border-style: solid;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    background: var(--bg);
}
.score-ring .score-num {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 900;
    line-height: 1;
}
.score-ring .score-denom {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: var(--ink-soft);
    letter-spacing: 0.04em;
}
.score-level-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.score-caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--ink-soft);
}

/* ── Stat tiles ── */
.stat-tile {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 18px 20px;
}
.stat-tile .stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-tile .stat-desc {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px;
    color: var(--ink-soft);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}

/* ── Risk badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 9px;
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 1px solid;
}
.badge-critical { background: var(--critical-bg); color: var(--critical); border-color: var(--critical-bdr); }
.badge-high     { background: var(--high-bg);     color: var(--high);     border-color: var(--high-bdr); }
.badge-medium   { background: var(--medium-bg);   color: var(--medium);   border-color: var(--medium-bdr); }
.badge-low      { background: var(--low-bg);      color: var(--low);      border-color: var(--low-bdr); }

/* ── Summary bar ── */
.summary-bar {
    background: var(--bg-soft);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--ink-mid);
    line-height: 1.75;
}

/* ── Clause text ── */
.clause-text {
    background: var(--bg-soft);
    border-left: 3px solid var(--border-strong);
    padding: 14px 16px;
    border-radius: 0 4px 4px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    line-height: 1.8;
    color: var(--ink-mid);
    max-height: 180px;
    overflow-y: auto;
    white-space: pre-wrap;
    margin-bottom: 16px;
}

/* ── Flag items ── */
.flag {
    padding: 10px 14px;
    border-radius: 0 4px 4px 0;
    margin: 5px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    line-height: 1.6;
    border-left: 3px solid;
}
.flag-critical { background: var(--critical-bg); border-color: var(--critical); }
.flag-high     { background: var(--high-bg);     border-color: var(--high); }
.flag-medium   { background: var(--medium-bg);   border-color: var(--medium); }
.flag-low      { background: var(--low-bg);      border-color: var(--low); }
.flag strong {
    display: block;
    font-size: 10.5px;
    font-weight: 500;
    margin-bottom: 3px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Recommendations ── */
.rec {
    background: var(--gold-soft);
    border-left: 3px solid var(--gold);
    padding: 9px 14px;
    border-radius: 0 4px 4px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: var(--ink-mid);
    margin: 4px 0;
    line-height: 1.6;
}

/* ── Score breakdown ── */
.score-breakdown {
    background: var(--bg-soft);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px 16px;
}
.sb-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--ink-mid);
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
}
.sb-row:last-child {
    border-bottom: none;
    font-weight: 600;
    color: var(--ink);
    padding-top: 8px;
}
.sb-row code {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    background: var(--bg-cream);
    border: 1px solid var(--border);
    padding: 1px 7px;
    border-radius: 2px;
    color: var(--ink);
}

/* ── Plain english ── */
.plain-eng {
    background: var(--gold-soft);
    border: 1px solid var(--gold-mid);
    border-radius: 4px;
    padding: 13px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    line-height: 1.75;
    color: var(--ink-mid);
}

/* ── SHAP bars ── */
.shap-bar-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 5px 0;
}
.shap-token {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    min-width: 130px;
    color: var(--ink-mid);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.shap-track {
    flex: 1;
    height: 5px;
    background: var(--border);
    border-radius: 999px;
    overflow: hidden;
}
.shap-fill { height: 100%; border-radius: 999px; }
.shap-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--ink-soft);
    min-width: 32px;
    text-align: right;
}

/* ── Streamlit widget overrides ── */
.main { background: var(--bg) !important; }

.stButton > button {
    background: var(--ink) !important;
    color: var(--bg) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: var(--red) !important; }

div[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border-strong) !important;
    border-radius: 4px !important;
    background: var(--bg-soft) !important;
}
div[data-testid="stFileUploader"]:hover { border-color: var(--gold) !important; }
div[data-testid="stFileUploader"] * { color: var(--ink-mid) !important; }

.stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    background: var(--bg-soft) !important;
    color: var(--ink-mid) !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    background: var(--bg-soft) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}

.stToggle label, .stCheckbox label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    color: var(--ink-mid) !important;
}

.stSlider label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    color: var(--ink-mid) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--red) !important;
    border-color: var(--red) !important;
}

.stAlert {
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    border-left-width: 3px !important;
}
.stAlert[data-baseweb="notification"] { background: var(--gold-soft) !important; }

.stExpander {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    background: var(--bg-card) !important;
}
.stExpander summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--ink-mid) !important;
    letter-spacing: 0.02em !important;
    padding: 14px 16px !important;
}
.stExpander summary:hover { color: var(--ink) !important; }

div[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 26px !important;
    color: var(--ink) !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--ink-soft) !important;
}
div[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 12px 16px;
    border-radius: 4px;
}

div[data-testid="stStatusWidget"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    background: var(--bg-soft) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}

.stRadio label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-soft); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 999px; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "analysis_done":  False,
        "contract_risk":  None,
        "extracted_doc":  None,
        "segmentation":   None,
        "classifications": None,
        "explanations":   {},
        "original_text":  "",
        "diff_result":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from extraction.pdf_extractor import PDFExtractor
    from extraction.clause_segmenter import ClauseSegmenter
    from classification.cuad_model import CUADModel
    from risk.risk_scorer import RiskScorer
    from risk.redline_diff import RedlineDiff
    from explainability.shap_explainer import SHAPExplainer

    extractor  = PDFExtractor()
    segmenter  = ClauseSegmenter(use_spacy=True)
    classifier = CUADModel()
    scorer     = RiskScorer()
    differ     = RedlineDiff()
    explainer  = SHAPExplainer()
    classifier.load()
    return extractor, segmenter, classifier, scorer, differ, explainer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def score_color(score: int) -> str:
    if score <= 25:  return "#166534"
    if score <= 55:  return "#92400e"
    if score <= 80:  return "#c2410c"
    return "#991b1b"

def score_level(score: int) -> str:
    if score <= 25:  return "Low"
    if score <= 55:  return "Medium"
    if score <= 80:  return "High"
    return "Critical"

def render_gauge(score: int, label: str = "Overall Risk"):
    color = score_color(score)
    level = score_level(score).upper()
    st.markdown(f"""
    <div class="score-ring-wrap">
        <div class="score-ring" style="border-color:{color};">
            <span class="score-num" style="color:{color};">{score}</span>
            <span class="score-denom">/ 100</span>
        </div>
        <span class="score-level-label" style="color:{color};">{level}</span>
        <span class="score-caption">{label}</span>
    </div>
    """, unsafe_allow_html=True)

def expander_title(cl) -> str:
    icons = {"critical": "⬛", "high": "🟥", "medium": "🟧", "low": "🟩"}
    icon = icons.get(cl.risk_level, "")
    return f"{icon}  {cl.clause_label}  ·  {cl.risk_score}/100"

def section_label(text: str) -> str:
    return f'<div class="section-label">{text}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">⚖ Contract Risk<br>Analyzer</div>
        <div class="sidebar-brand-sub">Powered by BERT · CUAD</div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "",
        ["Upload & Analyze", "Clause Review", "SHAP Explainability", "Redline Diff", "Risk Dashboard"],
        format_func=lambda x: f"  {x}",
    )

    st.markdown("""
    <div class="sidebar-meta">
        <div class="sidebar-meta-label">Model Info</div>
        <div class="sidebar-meta-item">Base: <code>bert-base-uncased</code></div>
        <div class="sidebar-meta-item">Dataset: CUAD · 510 contracts</div>
        <div class="sidebar-meta-item">Categories: 32 clause types</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Upload & Analyze
# ─────────────────────────────────────────────────────────────────────────────
if nav == "Upload & Analyze":

    st.markdown("""
    <div class="page-header">
        <h1><span style="color:var(--ink);">Contract Risk</span> <span class="accent">Analyzer.</span></h1>
        <p>Upload a legal contract PDF — clauses are extracted, classified via BERT fine-tuned on CUAD,<br>
        and scored with plain-English risk explanations and SHAP token-level attribution.</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown(section_label("Contract Input"), unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop a PDF contract here",
            type=["pdf"],
            label_visibility="collapsed",
        )
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    with col_right:
        st.subheader("Analysis Options")
        use_bert  = st.toggle("BERT Classifier", value=True,
                              help="Disable to use keyword fallback only")
        show_shap = st.toggle("Compute SHAP", value=True,
                              help="Token-level explanations (~5s extra)")
        min_words = st.slider("Min clause length (words)", 5, 50, 10)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    ready   = uploaded_file is not None
    analyze = st.button(
        "Analyze Contract →",
        type="primary",
        disabled=not ready,
    )

    if analyze:
        with st.spinner("Initialising pipeline…"):
            extractor, segmenter, classifier, scorer, differ, explainer = load_pipeline()

        try:
            with st.status("Extracting text…", expanded=False):
                if uploaded_file:
                    doc  = extractor.extract_from_bytes(uploaded_file.read(), uploaded_file.name)
                    text = doc.cleaned_text
                    st.session_state.extracted_doc = doc
                    st.write(f"✓  {doc.total_pages} pages · {len(text):,} chars")
                else:
                    st.error("Please upload a PDF.")
                    st.stop()
                st.session_state.original_text = text

            with st.status("Segmenting clauses…", expanded=False):
                seg = segmenter.segment(text)
                st.session_state.segmentation = seg
                st.write(f"✓  {seg.total_clauses} clauses via '{seg.method_used}'")
                for w in seg.warnings:
                    st.warning(w)

            with st.status("Classifying with BERT/CUAD…", expanded=False):
                to_classify     = [c for c in seg.clauses if c.word_count >= min_words]
                classifications = classifier.batch_classify(to_classify, show_progress=False)
                st.session_state.classifications = classifications
                model_used = classifications[0].model_used if classifications else "—"
                st.write(f"✓  {len(classifications)} clauses · model: {model_used}")

            with st.status("Scoring risk…", expanded=False):
                cr = scorer.score_contract(classifications)
                st.session_state.contract_risk = cr
                st.write(f"✓  Score: {cr.overall_score}/100 ({cr.overall_level.upper()})")

            if show_shap:
                with st.status("Computing SHAP…", expanded=False):
                    exps = {}
                    for risk, cls in zip(cr.clause_risks, classifications):
                        exps[risk.clause_id] = explainer.explain(
                            risk.raw_text, risk.clause_id, risk.clause_type, cls
                        )
                    st.session_state.explanations = exps
                    st.write(f"✓  {len(exps)} explanations")

            st.session_state.analysis_done = True

            # ── Result summary tiles ──
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(section_label("Results"), unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1: render_gauge(cr.overall_score)
            with c2:
                st.markdown(f"""<div class="stat-tile">
                    <div class="stat-num" style="color:var(--critical);">{len(cr.critical_clauses)}</div>
                    <div class="stat-desc">Critical</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="stat-tile">
                    <div class="stat-num" style="color:var(--high);">{len(cr.high_risk_clauses)}</div>
                    <div class="stat-desc">High Risk</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="stat-tile">
                    <div class="stat-num" style="color:var(--ink);">{len(cr.clause_risks)}</div>
                    <div class="stat-desc">Total Clauses</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(
                f'<div class="summary-bar" style="margin-top:16px;">{cr.summary}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            st.info("✓  Analysis complete — navigate with the sidebar.")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Clause Review
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Clause Review":
    st.markdown("""
    <div class="page-header">
        <h1><span style="color:#e8e4df;">Clause</span> <span class="accent">Review.</span></h1>
        <p>Every detected clause, its risk classification, flags, and recommended negotiation actions.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("Run an analysis first from **Upload & Analyze**.")
        st.stop()

    cr = st.session_state.contract_risk

    # ── Header row ──
    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        render_gauge(cr.overall_score)
    with c2:
        st.markdown(f'<div class="summary-bar">{cr.summary}</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        ca, cb, cc = st.columns(3)
        ca.markdown(f"""<div class="stat-tile">
            <div class="stat-num" style="color:var(--critical);">{len(cr.critical_clauses)}</div>
            <div class="stat-desc">Critical</div></div>""", unsafe_allow_html=True)
        cb.markdown(f"""<div class="stat-tile">
            <div class="stat-num" style="color:var(--high);">{len(cr.high_risk_clauses)}</div>
            <div class="stat-desc">High Risk</div></div>""", unsafe_allow_html=True)
        cc.markdown(f"""<div class="stat-tile">
            <div class="stat-num" style="color:var(--ink);">{len(cr.clause_risks)}</div>
            <div class="stat-desc">Total</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Filters ──
    fc1, fc2 = st.columns([2, 2])
    with fc1:
        risk_filter = st.multiselect(
            "Filter by risk level",
            ["critical", "high", "medium", "low"],
            default=["critical", "high", "medium"],
        )
    with fc2:
        sort_by = st.selectbox("Sort", ["Score ↓", "Score ↑", "Clause Type A–Z"])

    clauses = cr.clause_risks
    if sort_by == "Score ↓":   clauses = sorted(clauses, key=lambda x: x.risk_score, reverse=True)
    elif sort_by == "Score ↑": clauses = sorted(clauses, key=lambda x: x.risk_score)
    else:                      clauses = sorted(clauses, key=lambda x: x.clause_label)
    clauses = [c for c in clauses if c.risk_level in risk_filter]

    st.markdown(section_label(f"{len(clauses)} clauses"), unsafe_allow_html=True)

    for cl in clauses:
        color = score_color(cl.risk_score)
        with st.expander(expander_title(cl), expanded=(cl.risk_level == "critical")):
            body_l, body_r = st.columns([3, 1], gap="large")

            with body_l:
                st.markdown(section_label("Clause Text"), unsafe_allow_html=True)
                preview = cl.raw_text[:700] + ("…" if len(cl.raw_text) > 700 else "")
                st.markdown(
                    f'<div class="clause-text" style="border-left-color:{color};">{preview}</div>',
                    unsafe_allow_html=True,
                )

                if cl.flags:
                    st.markdown(section_label("Risk Flags"), unsafe_allow_html=True)
                    for f in cl.flags:
                        matched = (f'<code style="font-size:10.5px;font-family:IBM Plex Mono,monospace;'
                                   f'background:rgba(0,0,0,0.04);padding:1px 6px;border-radius:2px;">'
                                   f'{f.matched_text}</code>') if f.matched_text else ""
                        st.markdown(f"""
                        <div class="flag flag-{f.severity}">
                            <strong>{f.title}</strong>
                            {f.explanation} {matched}
                        </div>""", unsafe_allow_html=True)

                if cl.recommendations:
                    st.markdown(section_label("Recommendations"), unsafe_allow_html=True)
                    for r in cl.recommendations:
                        st.markdown(f'<div class="rec">→ {r}</div>', unsafe_allow_html=True)

            with body_r:
                render_gauge(cl.risk_score, cl.clause_label)

                st.markdown(section_label("Score"), unsafe_allow_html=True)
                st.markdown(f"""
                <div class="score-breakdown">
                    <div class="sb-row"><span>Base</span><code>{cl.base_score}</code></div>
                    <div class="sb-row"><span>Modifiers</span><code>{'+'if cl.modifier_score>=0 else ''}{cl.modifier_score}</code></div>
                    <div class="sb-row"><span>Final</span><code>{cl.risk_score}</code></div>
                </div>""", unsafe_allow_html=True)

                st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
                st.markdown(section_label("Summary"), unsafe_allow_html=True)
                st.markdown(f'<div class="plain-eng">{cl.plain_english}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "SHAP Explainability":
    st.markdown("""
    <div class="page-header">
        <h1><span style="color:#e8e4df;">SHAP</span> <span class="accent">Explainability.</span></h1>
        <p>Token-level feature importance — which words and phrases drove each risk classification.<br>
        Red = increases risk &nbsp;·&nbsp; Green = decreases risk.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("Run an analysis first from **Upload & Analyze**.")
        st.stop()

    exps = st.session_state.explanations
    cr   = st.session_state.contract_risk

    if not exps:
        st.warning("No SHAP data. Re-run with **Compute SHAP** enabled.")
        st.stop()

    st.markdown(section_label("Select Clause"), unsafe_allow_html=True)
    options  = {
        f"{r.clause_label}  ·  {r.risk_score}/100": r.clause_id
        for r in sorted(cr.clause_risks, key=lambda x: x.risk_score, reverse=True)
    }
    selected = st.selectbox("Clause", list(options.keys()), label_visibility="collapsed")
    sel_id   = options[selected]
    exp      = exps.get(sel_id)
    risk     = next((r for r in cr.clause_risks if r.clause_id == sel_id), None)

    if exp and risk:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 2], gap="large")

        with col_l:
            st.markdown(section_label("Highlighted Clause"), unsafe_allow_html=True)
            st.caption(f"Method: {exp.method}")
            st.markdown(f"""
            <div style="background:var(--bg-soft);border:1px solid var(--border);
                 border-radius:4px;padding:20px 22px;font-size:13px;
                 line-height:1.9;font-family:'IBM Plex Mono',monospace;">
            {exp.explanation_html}
            </div>""", unsafe_allow_html=True)

            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            st.markdown(section_label("Summary"), unsafe_allow_html=True)
            st.markdown(f'<div class="plain-eng">{risk.plain_english}</div>', unsafe_allow_html=True)

        with col_r:
            render_gauge(risk.risk_score, risk.clause_label)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            if exp.top_positive_tokens:
                st.markdown(section_label("Risk-Increasing Terms"), unsafe_allow_html=True)
                for ti in exp.top_positive_tokens[:8]:
                    pct = int(ti.importance * 100)
                    st.markdown(f"""
                    <div class="shap-bar-wrap">
                        <span class="shap-token">{ti.token}</span>
                        <div class="shap-track">
                            <div class="shap-fill" style="width:{pct}%;background:var(--red);"></div>
                        </div>
                        <span class="shap-pct">{pct}%</span>
                    </div>""", unsafe_allow_html=True)

            if exp.top_negative_tokens:
                st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
                st.markdown(section_label("Risk-Reducing Terms"), unsafe_allow_html=True)
                for ti in exp.top_negative_tokens[:5]:
                    pct = int(abs(ti.importance) * 100)
                    st.markdown(f"""
                    <div class="shap-bar-wrap">
                        <span class="shap-token">{ti.token}</span>
                        <div class="shap-track">
                            <div class="shap-fill" style="width:{pct}%;background:var(--low);"></div>
                        </div>
                        <span class="shap-pct">{pct}%</span>
                    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Redline Diff
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Redline Diff":
    st.markdown("""
    <div class="page-header">
        <h1><span style="color:var(--ink-soft);">Redline</span> <span class="accent">Diff.</span></h1>
        <p>Compare two contract versions side-by-side. Deleted text in red · inserted text in green.</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown(section_label("Original"), unsafe_allow_html=True)
        orig_file = st.file_uploader("Original PDF", type=["pdf"], key="orig_pdf",
                                      label_visibility="collapsed")
    with col_r:
        st.markdown(section_label("Revised"), unsafe_allow_html=True)
        rev_file  = st.file_uploader("Revised PDF", type=["pdf"], key="rev_pdf",
                                      label_visibility="collapsed")

    if st.button("Generate Redlines →", type="primary"):
        extractor, *_, differ, _ = load_pipeline()
        orig = ""
        rev  = ""
        if orig_file:
            with st.spinner("Extracting original…"):
                orig = extractor.extract_from_bytes(orig_file.read(), orig_file.name).cleaned_text
        if rev_file:
            with st.spinner("Extracting revised…"):
                rev = extractor.extract_from_bytes(rev_file.read(), rev_file.name).cleaned_text
        if not orig.strip() or not rev.strip():
            st.error("Provide both original and revised PDF files.")
        else:
            with st.spinner("Diffing…"):
                st.session_state.diff_result = differ.diff(orig, rev)

    if st.session_state.diff_result:
        dr = st.session_state.diff_result
        s  = dr.change_summary

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Lines Added",   f"+{s['added_lines']}")
        m2.metric("Lines Removed", f"−{s['removed_lines']}")
        m3.metric("Similarity",    f"{int(s['similarity_ratio']*100)}%")
        m4.metric("% Changed",     f"{s['pct_changed']}%")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        mode = st.radio("View", ["Side by Side", "Inline"], horizontal=True)
        if mode == "Side by Side":
            st.markdown(dr.side_by_side_html, unsafe_allow_html=True)
        else:
            st.markdown(dr.inline_html, unsafe_allow_html=True)

        if dr.changed_sections:
            with st.expander(f"{len(dr.changed_sections)} changed sections"):
                for sec in dr.changed_sections:
                    st.markdown(f"**{sec['heading']}** · `{sec['tag']}`")
                    if sec["orig_text"]: st.caption(f"Before: {sec['orig_text'][:120]}…")
                    if sec["rev_text"]:  st.caption(f"After:  {sec['rev_text'][:120]}…")
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Risk Dashboard
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Risk Dashboard":
    st.markdown("""
    <div class="page-header">
        <h1><span style="color:#e8e4df;">Risk</span> <span class="accent">Dashboard.</span></h1>
        <p>Visual overview of contract risk distribution, clause scores, and top concerns.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("Run an analysis first from **Upload & Analyze**.")
        st.stop()

    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
    except ImportError:
        st.error("Install plotly and pandas: `pip install plotly pandas`")
        st.stop()

    cr = st.session_state.contract_risk

    # ── Top row ──
    col_gauge, col_dist = st.columns([1, 2], gap="large")

    with col_gauge:
        render_gauge(cr.overall_score, "Overall Contract Risk")
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="summary-bar" style="font-size:11.5px;">{cr.summary}</div>',
            unsafe_allow_html=True,
        )

    with col_dist:
        lc = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for r in cr.clause_risks: lc[r.risk_level] += 1

        fig = go.Figure(go.Pie(
            labels=["Critical", "High", "Medium", "Low"],
            values=[lc["critical"], lc["high"], lc["medium"], lc["low"]],
            hole=0.58,
            marker_colors=["#991b1b", "#c2410c", "#92400e", "#166534"],
            marker_line=dict(color="#e8e4df", width=2),
            textinfo="label+value",
            textfont=dict(family="IBM Plex Mono", size=11),
        ))
        fig.update_layout(
            title=dict(
                text="Clause Risk Distribution",
                font=dict(family="Playfair Display", size=15, color="#1c1916"),
            ),
            height=280,
            margin=dict(t=44, b=0, l=0, r=0),
            font=dict(family="IBM Plex Mono"),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Bar chart ──
    df = pd.DataFrame([{
        "Clause":  r.clause_label,
        "Score":   r.risk_score,
        "Level":   r.risk_level.title(),
        "Flags":   len(r.flags),
        "Preview": r.raw_text[:80] + "…",
    } for r in cr.clause_risks])

    fig2 = px.bar(
        df.sort_values("Score", ascending=True),
        x="Score", y="Clause", orientation="h",
        color="Level",
        color_discrete_map={
            "Critical": "#991b1b",
            "High":     "#c2410c",
            "Medium":   "#92400e",
            "Low":      "#166534",
        },
        hover_data=["Flags", "Preview"],
        title="Risk Score by Clause",
    )
    fig2.update_layout(
        height=max(320, len(df) * 30),
        font=dict(family="IBM Plex Mono", size=11),
        yaxis_title="", xaxis_range=[0, 100],
        xaxis_title="Risk Score",
        margin=dict(l=0, r=20, t=44, b=20),
        title_font=dict(family="Playfair Display", size=15, color="#1c1916"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f7f5f2",
        legend_title_text="",
        legend=dict(font=dict(family="IBM Plex Mono", size=11)),
    )
    fig2.update_xaxes(gridcolor="#e8e4df", gridwidth=1, tickfont=dict(family="IBM Plex Mono", size=10))
    fig2.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(family="IBM Plex Mono", size=10))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Top concerns ──
    if cr.top_concerns:
        st.markdown(section_label("Top Concerns"), unsafe_allow_html=True)
        for concern in cr.top_concerns:
            st.markdown(f'<div class="flag flag-high">{concern}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Demo contract
# ─────────────────────────────────────────────────────────────────────────────
DEMO_CONTRACT = """SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 1, 2025 ("Effective Date")
by and between Acme Corp, a Delaware corporation ("Service Provider"), and GlobalCo Inc.,
a California corporation ("Client").

1. SERVICES
Service Provider shall perform software development services as described in Exhibit A.

2. TERM AND RENEWAL
This Agreement shall commence on the Effective Date and shall automatically renew for successive
one-year terms unless either party provides written notice of non-renewal at least 15 days prior
to the end of the then-current term.

3. INTELLECTUAL PROPERTY
All work product, inventions, and deliverables created by Service Provider under this Agreement
shall be considered work for hire and Client shall own all right, title, and interest in and to
all such work product, including all intellectual property rights therein.

4. INDEMNIFICATION
Service Provider shall indemnify, defend, and hold harmless Client from and against any and all
claims, damages, losses, and expenses arising out of or relating to Service Provider's performance
under this Agreement. Such indemnification shall be irrevocable and shall survive termination of
this Agreement.

5. LIMITATION OF LIABILITY
IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL
DAMAGES. THE TOTAL AGGREGATE LIABILITY OF SERVICE PROVIDER SHALL NOT EXCEED THE FEES PAID IN THE
ONE MONTH PRECEDING THE CLAIM.

6. NON-COMPETE
During the term and for a period of three (3) years following termination, Service Provider shall
not, directly or indirectly, engage in any business that competes with Client on a worldwide basis
without Client's prior written consent.

7. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.

8. DISPUTE RESOLUTION
Any dispute arising under this Agreement shall be resolved by binding arbitration under the AAA
Commercial Arbitration Rules. The parties waive any right to a jury trial.

9. CONFIDENTIALITY
Service Provider shall maintain the confidentiality of all Client information in perpetuity and
shall not disclose such information to any third party.

10. ASSIGNMENT
Service Provider may not assign this Agreement or any rights hereunder without Client's prior
written consent in its sole discretion.
"""
