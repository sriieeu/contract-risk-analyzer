"""
Contract Risk Analyzer — Streamlit UI

Run with:
    streamlit run src/ui/app.py
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.risk-critical { background: #7f1d1d; color: white; border-radius: 6px; padding: 4px 10px; font-weight: 600; font-size: 12px; }
.risk-high     { background: #dc2626; color: white; border-radius: 6px; padding: 4px 10px; font-weight: 600; font-size: 12px; }
.risk-medium   { background: #d97706; color: white; border-radius: 6px; padding: 4px 10px; font-weight: 600; font-size: 12px; }
.risk-low      { background: #16a34a; color: white; border-radius: 6px; padding: 4px 10px; font-weight: 600; font-size: 12px; }

.score-ring {
    width: 120px; height: 120px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Serif Display', serif;
    font-size: 32px; font-weight: bold;
    margin: 0 auto;
}

.clause-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
.clause-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}

.flag-item {
    background: #fef9f0;
    border-left: 4px solid #f59e0b;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 0 6px 6px 0;
    font-size: 14px;
}
.flag-critical { border-left-color: #7f1d1d; background: #fff1f2; }
.flag-high     { border-left-color: #dc2626; background: #fff1f2; }
.flag-medium   { border-left-color: #d97706; background: #fef9f0; }
.flag-low      { border-left-color: #16a34a; background: #f0fdf4; }

.rec-item {
    background: #f0f9ff;
    border-left: 4px solid #0ea5e9;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 4px 4px 0;
    font-size: 13px;
}

.stProgress > div > div > div { border-radius: 999px; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0f172a;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Initialize session state
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "analysis_done": False,
        "contract_risk": None,
        "extracted_doc": None,
        "segmentation": None,
        "classifications": None,
        "explanations": {},
        "original_text": "",
        "revised_text": "",
        "diff_result": None,
        "active_tab": "upload",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load all pipeline components (cached across reruns)."""
    from extraction.pdf_extractor import PDFExtractor
    from extraction.clause_segmenter import ClauseSegmenter
    from classification.cuad_model import CUADModel
    from risk.risk_scorer import RiskScorer
    from risk.redline_diff import RedlineDiff
    from explainability.shap_explainer import SHAPExplainer

    extractor = PDFExtractor()
    segmenter = ClauseSegmenter(use_spacy=True)
    classifier = CUADModel()  # Will use keyword fallback if transformers not available
    scorer = RiskScorer()
    differ = RedlineDiff()
    explainer = SHAPExplainer()

    # Try to load BERT model
    classifier.load()

    return extractor, segmenter, classifier, scorer, differ, explainer


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Contract Risk Analyzer")
    st.markdown("*Powered by BERT fine-tuned on CUAD*")
    st.divider()

    nav = st.radio(
        "Navigation",
        ["📤 Upload & Analyze", "📋 Clause Review", "🔍 SHAP Explainability", "📝 Redline Diff", "📊 Risk Dashboard"],
        index=0,
    )

    st.divider()
    st.markdown("**Model Info**")
    st.caption("Base: `bert-base-uncased`")
    st.caption("Dataset: CUAD (510 contracts)")
    st.caption("Clause types: 32 categories")
    st.divider()
    st.markdown("**About CUAD**")
    st.caption(
        "The Contract Understanding Atticus Dataset (CUAD) contains 510 "
        "commercial contracts with 13,000+ expert annotations across 41 "
        "legal clause categories."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def risk_badge(level: str) -> str:
    icons = {"critical": "🚨", "high": "🔴", "medium": "⚠️", "low": "✅"}
    return f'<span class="risk-{level}">{icons.get(level, "")} {level.upper()}</span>'


def score_color(score: int) -> str:
    if score <= 25:
        return "#16a34a"
    elif score <= 55:
        return "#d97706"
    elif score <= 80:
        return "#dc2626"
    else:
        return "#7f1d1d"


def render_score_gauge(score: int, label: str = "Overall Risk"):
    color = score_color(score)
    level_labels = {
        range(0, 26): "LOW RISK",
        range(26, 56): "MEDIUM RISK",
        range(56, 81): "HIGH RISK",
        range(81, 101): "CRITICAL RISK",
    }
    level_text = next((v for k, v in level_labels.items() if score in k), "")

    st.markdown(f"""
    <div style="text-align: center; padding: 20px;">
        <div style="
            width: 140px; height: 140px;
            border-radius: 50%;
            border: 8px solid {color};
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            box-shadow: 0 0 30px {color}40;
        ">
            <div style="font-size: 38px; font-weight: 800; color: {color}; font-family: 'DM Serif Display', serif;">{score}</div>
            <div style="font-size: 11px; color: #64748b; font-weight: 500;">/ 100</div>
        </div>
        <div style="margin-top: 12px; font-size: 13px; font-weight: 700; color: {color}; letter-spacing: 0.08em;">{level_text}</div>
        <div style="font-size: 12px; color: #64748b; margin-top: 4px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Upload & Analyze
# ─────────────────────────────────────────────────────────────────────────────
if "Upload" in nav:
    st.title("⚖️ Contract Risk Analyzer")
    st.markdown(
        "Upload a legal contract PDF to automatically extract clauses, "
        "classify them using BERT fine-tuned on CUAD, and score risk with plain-English explanations."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Contract PDF",
            type=["pdf"],
            help="Supports standard PDF contracts. Image-based PDFs require OCR preprocessing.",
        )

        # Demo text option
        use_demo = st.checkbox(
            "Use demo contract text instead",
            help="Test the system without a PDF using a sample contract snippet",
        )

        if use_demo:
            demo_text = st.text_area(
                "Contract Text",
                value=DEMO_CONTRACT,
                height=300,
                help="Paste or edit contract text directly",
            )

    with col2:
        st.markdown("#### Analysis Options")
        use_bert = st.toggle("Use BERT Classifier", value=True,
                             help="When disabled, uses faster keyword-based classifier")
        show_shap = st.toggle("Compute SHAP Values", value=True,
                              help="Explains what drove each risk flag (slower)")
        min_clause_len = st.slider("Min clause length (words)", 5, 50, 10)

    st.divider()

    analyze_btn = st.button(
        "🔍 Analyze Contract",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None and not use_demo),
    )

    if analyze_btn:
        with st.spinner("Loading pipeline..."):
            extractor, segmenter, classifier, scorer, differ, explainer = load_pipeline()

        try:
            # Step 1: Extract text
            with st.status("📄 Extracting contract text...", expanded=True) as status:
                if uploaded_file is not None:
                    pdf_bytes = uploaded_file.read()
                    doc = extractor.extract_from_bytes(pdf_bytes, uploaded_file.name)
                    text = doc.cleaned_text
                    st.session_state.extracted_doc = doc
                    st.write(f"✅ Extracted {doc.total_pages} pages, {len(text):,} characters")
                else:
                    text = demo_text
                    st.session_state.extracted_doc = None
                    st.write("✅ Using demo contract text")

                st.session_state.original_text = text

            # Step 2: Segment clauses
            with st.status("✂️ Segmenting clauses...", expanded=True) as status:
                segmentation = segmenter.segment(text)
                st.session_state.segmentation = segmentation
                st.write(f"✅ Found {segmentation.total_clauses} clauses via '{segmentation.method_used}' method")
                for w in segmentation.warnings:
                    st.warning(w)

            # Step 3: Classify
            with st.status("🤖 Classifying clauses with BERT/CUAD...", expanded=True) as status:
                clauses_to_classify = [
                    c for c in segmentation.clauses
                    if c.word_count >= min_clause_len
                ]
                classifications = classifier.batch_classify(clauses_to_classify, show_progress=False)
                st.session_state.classifications = classifications
                st.write(f"✅ Classified {len(classifications)} clauses")
                model_used = classifications[0].model_used if classifications else "none"
                st.write(f"   Model: `{model_used}`")

            # Step 4: Score risk
            with st.status("📊 Scoring risk...", expanded=True) as status:
                contract_risk = scorer.score_contract(classifications)
                st.session_state.contract_risk = contract_risk
                st.write(f"✅ Risk score: **{contract_risk.overall_score}/100** ({contract_risk.overall_level.upper()})")
                if contract_risk.critical_clauses:
                    st.write(f"   🚨 {len(contract_risk.critical_clauses)} critical issues found")

            # Step 5: SHAP (optional)
            if show_shap:
                with st.status("🔍 Computing SHAP explanations...", expanded=True) as status:
                    explanations = {}
                    for cr, cls_result in zip(contract_risk.clause_risks, classifications):
                        exp = explainer.explain(
                            cr.raw_text, cr.clause_id, cr.clause_type, cls_result
                        )
                        explanations[cr.clause_id] = exp
                    st.session_state.explanations = explanations
                    st.write(f"✅ Generated {len(explanations)} explanations")

            st.session_state.analysis_done = True
            st.success(f"✅ Analysis complete! Navigate using the sidebar.")
            st.balloons()

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Clause Review
# ─────────────────────────────────────────────────────────────────────────────
elif "Clause Review" in nav:
    st.title("📋 Clause-by-Clause Review")

    if not st.session_state.analysis_done:
        st.info("Upload and analyze a contract first.")
        st.stop()

    cr: object = st.session_state.contract_risk

    # Summary bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_score_gauge(cr.overall_score, "Overall Risk")
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 32px; font-weight: 800; color: #7f1d1d;">{len(cr.critical_clauses)}</div>
            <div style="color: #64748b; font-size: 13px;">Critical Clauses</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 32px; font-weight: 800; color: #dc2626;">{len(cr.high_risk_clauses)}</div>
            <div style="color: #64748b; font-size: 13px;">High-Risk Clauses</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 32px; font-weight: 800; color: #1e293b;">{len(cr.clause_risks)}</div>
            <div style="color: #64748b; font-size: 13px;">Total Clauses</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"**Summary:** {cr.summary}")
    st.divider()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.multiselect(
            "Filter by risk level",
            ["critical", "high", "medium", "low"],
            default=["critical", "high", "medium"],
        )
    with col2:
        sort_by = st.selectbox("Sort by", ["Risk Score (High→Low)", "Risk Score (Low→High)", "Clause Type"])

    # Sort
    clauses = cr.clause_risks
    if sort_by == "Risk Score (High→Low)":
        clauses = sorted(clauses, key=lambda x: x.risk_score, reverse=True)
    elif sort_by == "Risk Score (Low→High)":
        clauses = sorted(clauses, key=lambda x: x.risk_score)
    else:
        clauses = sorted(clauses, key=lambda x: x.clause_label)

    # Filter
    clauses = [c for c in clauses if c.risk_level in risk_filter]

    st.markdown(f"*Showing {len(clauses)} clauses*")

    for clause_risk in clauses:
        color = clause_risk.risk_color

        with st.expander(
            f"{risk_badge(clause_risk.risk_level)} **{clause_risk.clause_label}** — Score: {clause_risk.risk_score}/100",
            expanded=clause_risk.risk_level in ("critical",),
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown("**Clause Text:**")
                st.markdown(f"""
                <div style="background: #f8fafc; border-left: 4px solid {color};
                     padding: 12px 16px; border-radius: 0 8px 8px 0;
                     font-size: 14px; line-height: 1.7; max-height: 200px;
                     overflow-y: auto; white-space: pre-wrap;">
                {clause_risk.raw_text[:800]}{'...' if len(clause_risk.raw_text) > 800 else ''}
                </div>
                """, unsafe_allow_html=True)

                if clause_risk.flags:
                    st.markdown("**Risk Flags:**")
                    for flag in clause_risk.flags:
                        st.markdown(f"""
                        <div class="flag-item flag-{flag.severity}">
                            <strong>{flag.title}</strong><br>
                            <span style="font-size: 13px; color: #475569;">{flag.explanation}</span>
                            {f'<br><code style="font-size: 11px;">{flag.matched_text}</code>' if flag.matched_text else ''}
                        </div>
                        """, unsafe_allow_html=True)

                if clause_risk.recommendations:
                    st.markdown("**Recommendations:**")
                    for rec in clause_risk.recommendations:
                        st.markdown(f'<div class="rec-item">💡 {rec}</div>', unsafe_allow_html=True)

            with col2:
                render_score_gauge(clause_risk.risk_score, clause_risk.clause_label)

                st.markdown("**Score Breakdown:**")
                st.markdown(f"Base score: `{clause_risk.base_score}`")
                st.markdown(f"Modifiers: `{'+' if clause_risk.modifier_score >= 0 else ''}{clause_risk.modifier_score}`")
                st.markdown(f"**Final: `{clause_risk.risk_score}`**")

                st.markdown("**Plain English:**")
                st.info(clause_risk.plain_english)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────
elif "SHAP" in nav:
    st.title("🔍 SHAP Explainability")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) shows which words and phrases "
        "in each clause drove the risk classification. "
        "**Red = increases risk, Green = decreases risk.**"
    )

    if not st.session_state.analysis_done:
        st.info("Upload and analyze a contract first.")
        st.stop()

    explanations = st.session_state.explanations
    cr = st.session_state.contract_risk

    if not explanations:
        st.warning("SHAP explanations were not computed. Re-analyze with 'Compute SHAP Values' enabled.")
        st.stop()

    # Select clause
    clause_options = {
        f"{r.clause_label} (Score: {r.risk_score})": r.clause_id
        for r in sorted(cr.clause_risks, key=lambda x: x.risk_score, reverse=True)
    }

    selected_label = st.selectbox("Select clause to explain:", list(clause_options.keys()))
    selected_id = clause_options[selected_label]

    exp = explanations.get(selected_id)
    clause_risk = next((r for r in cr.clause_risks if r.clause_id == selected_id), None)

    if exp and clause_risk:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Highlighted Clause Text")
            st.caption(f"Method: `{exp.method}` | Red = risk-increasing | Green = risk-reducing")
            st.markdown(exp.explanation_html, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Top Risk-Increasing Terms")
            if exp.top_positive_tokens:
                for ti in exp.top_positive_tokens[:8]:
                    importance_pct = int(ti.importance * 100)
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">
                        <code style="min-width: 120px; font-size: 13px;">{ti.token}</code>
                        <div style="flex: 1; background: #fee2e2; border-radius: 999px; height: 8px;">
                            <div style="width: {importance_pct}%; background: #dc2626; height: 8px; border-radius: 999px;"></div>
                        </div>
                        <span style="font-size: 12px; color: #64748b;">{importance_pct}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No strongly risk-increasing terms detected.")

            if exp.top_negative_tokens:
                st.markdown("#### Risk-Reducing Terms")
                for ti in exp.top_negative_tokens[:5]:
                    importance_pct = int(abs(ti.importance) * 100)
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">
                        <code style="min-width: 120px; font-size: 13px;">{ti.token}</code>
                        <div style="flex: 1; background: #dcfce7; border-radius: 999px; height: 8px;">
                            <div style="width: {importance_pct}%; background: #16a34a; height: 8px; border-radius: 999px;"></div>
                        </div>
                        <span style="font-size: 12px; color: #64748b;">-{importance_pct}%</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()
        st.markdown("**Risk explanation:**")
        st.markdown(clause_risk.plain_english)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Redline Diff
# ─────────────────────────────────────────────────────────────────────────────
elif "Redline" in nav:
    st.title("📝 Redline Diff Viewer")
    st.markdown("Compare two versions of a contract to see exactly what changed.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Original Contract")
        orig_file = st.file_uploader("Upload original PDF", type=["pdf"], key="orig_pdf")
        or_text = st.text_area(
            "Or paste original text",
            value=st.session_state.original_text,
            height=200,
            key="orig_text_area",
        )

    with col2:
        st.markdown("#### Revised Contract")
        rev_file = st.file_uploader("Upload revised PDF", type=["pdf"], key="rev_pdf")
        rev_text = st.text_area(
            "Or paste revised text",
            height=200,
            key="rev_text_area",
            placeholder="Paste the revised/amended contract text here...",
        )

    diff_btn = st.button("🔀 Generate Redlines", type="primary", use_container_width=True)

    if diff_btn:
        extractor, _, _, _, differ, _ = load_pipeline()

        orig = or_text
        rev = rev_text

        if orig_file:
            with st.spinner("Extracting original PDF..."):
                doc = extractor.extract_from_bytes(orig_file.read(), orig_file.name)
                orig = doc.cleaned_text

        if rev_file:
            with st.spinner("Extracting revised PDF..."):
                doc = extractor.extract_from_bytes(rev_file.read(), rev_file.name)
                rev = doc.cleaned_text

        if not orig.strip() or not rev.strip():
            st.error("Please provide both original and revised contract text.")
        else:
            with st.spinner("Generating redlines..."):
                diff_result = differ.diff(orig, rev)
                st.session_state.diff_result = diff_result

    if st.session_state.diff_result:
        dr = st.session_state.diff_result
        summary = dr.change_summary

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lines Added", f"+{summary['added_lines']}")
        col2.metric("Lines Removed", f"-{summary['removed_lines']}")
        col3.metric("Similarity", f"{int(summary['similarity_ratio']*100)}%")
        col4.metric("% Changed", f"{summary['pct_changed']}%")

        st.divider()

        view_mode = st.radio("View mode", ["Side by Side", "Inline"], horizontal=True)

        if view_mode == "Side by Side":
            st.markdown(dr.side_by_side_html, unsafe_allow_html=True)
        else:
            st.markdown(dr.inline_html, unsafe_allow_html=True)

        if dr.changed_sections:
            with st.expander(f"📍 {len(dr.changed_sections)} Changed Sections"):
                for sec in dr.changed_sections:
                    st.markdown(f"**{sec['heading']}** ({sec['tag']})")
                    if sec["orig_text"]:
                        st.markdown(f"*Before:* `{sec['orig_text'][:100]}...`")
                    if sec["rev_text"]:
                        st.markdown(f"*After:* `{sec['rev_text'][:100]}...`")
                    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Risk Dashboard
# ─────────────────────────────────────────────────────────────────────────────
elif "Dashboard" in nav:
    st.title("📊 Risk Dashboard")

    if not st.session_state.analysis_done:
        st.info("Upload and analyze a contract first.")
        st.stop()

    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
    except ImportError:
        st.error("Plotly required: pip install plotly pandas")
        st.stop()

    cr = st.session_state.contract_risk

    col1, col2 = st.columns([1, 2])

    with col1:
        render_score_gauge(cr.overall_score, "Overall Contract Risk")
        st.markdown(f"**{cr.summary}**")

        if cr.top_concerns:
            st.markdown("#### Top Concerns")
            for concern in cr.top_concerns:
                st.warning(concern)

    with col2:
        # Risk level distribution donut chart
        level_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for r in cr.clause_risks:
            level_counts[r.risk_level] += 1

        fig = go.Figure(go.Pie(
            labels=["Critical", "High", "Medium", "Low"],
            values=[level_counts["critical"], level_counts["high"], level_counts["medium"], level_counts["low"]],
            hole=0.5,
            marker_colors=["#7f1d1d", "#dc2626", "#d97706", "#16a34a"],
            textinfo="label+value",
        ))
        fig.update_layout(
            title="Clause Risk Distribution",
            height=300,
            margin=dict(t=40, b=0, l=0, r=0),
            font=dict(family="DM Sans"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Risk score scatter
    df = pd.DataFrame([
        {
            "Clause": r.clause_label,
            "Risk Score": r.risk_score,
            "Risk Level": r.risk_level.title(),
            "Flags": len(r.flags),
            "Preview": r.raw_text[:80] + "...",
        }
        for r in cr.clause_risks
    ])

    fig2 = px.bar(
        df.sort_values("Risk Score", ascending=True),
        x="Risk Score",
        y="Clause",
        color="Risk Level",
        color_discrete_map={
            "Critical": "#7f1d1d",
            "High": "#dc2626",
            "Medium": "#d97706",
            "Low": "#16a34a",
        },
        orientation="h",
        title="Risk Score by Clause",
        hover_data=["Flags", "Preview"],
    )
    fig2.update_layout(
        height=max(300, len(df) * 28),
        font=dict(family="DM Sans"),
        yaxis_title="",
        xaxis_range=[0, 100],
        margin=dict(l=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Clause type breakdown
    if cr.clause_type_distribution:
        st.markdown("#### Clause Type Frequency")
        type_df = pd.DataFrame(
            list(cr.clause_type_distribution.items()),
            columns=["Clause Type", "Count"],
        ).sort_values("Count", ascending=False)

        fig3 = px.bar(
            type_df,
            x="Clause Type",
            y="Count",
            color="Count",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
        )
        fig3.update_layout(
            height=300,
            font=dict(family="DM Sans"),
            margin=dict(t=10),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Demo contract text
# ─────────────────────────────────────────────────────────────────────────────
DEMO_CONTRACT = """
SERVICE AGREEMENT

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
