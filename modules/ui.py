
import streamlit as st
from pathlib import Path

# =====================================================
# EVENTECHO DESIGN SYSTEM
# =====================================================

COLORS = {
    "background": "#080D19",
    "sidebar": "#0B1220",
    "panel": "#111B2E",
    "border": "#263247",
    "primary": "#3B82F6",
    "positive": "#34D399",
    "negative": "#F87171",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "violet": "#A78BFA"
}


# =====================================================
# GLOBAL CSS
# =====================================================

def apply_global_styles():

    st.markdown("""
    <style>

    :root {
        --bg: #080D19;
        --sidebar: #0B1220;
        --panel: #111B2E;
        --border: #263247;
        --primary: #3B82F6;
        --positive: #34D399;
        --negative: #F87171;
        --text: #F8FAFC;
        --muted: #94A3B8;
        --violet: #A78BFA;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        background-color: var(--sidebar);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    /* Hide default navigation.
       We will use custom page links below. */

    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Branding */

    .ee-brand {
        color: #F8FAFC;
        font-size: 27px;
        font-weight: 750;
        letter-spacing: -1px;
        margin-bottom: 2px;
    }

    .ee-tagline {
        color: #94A3B8;
        font-size: 10px;
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }

    .ee-brand-line {
        height: 3px;
        width: 42px;
        background: #3B82F6;
        border-radius: 5px;
        margin: 13px 0 22px 0;
    }

    /* Sidebar navigation links */

    [data-testid="stSidebar"] [data-testid="stPageLink"] a {
        border-radius: 8px;
        padding: 9px 11px;
        color: #CBD5E1;
        transition: background-color 0.2s ease;
    }

    [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
        background-color: rgba(59, 130, 246, 0.15);
        color: #F8FAFC;
    }

    /* Section headings */

    h1, h2, h3 {
        color: #F8FAFC !important;
        letter-spacing: -0.4px;
    }

    .ee-caption {
        color: #94A3B8;
        font-size: 13px;
        margin-top: -8px;
        margin-bottom: 20px;
    }

    /* Metric cards */

    [data-testid="stMetric"] {
        background-color: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 17px 19px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 13px;
    }

    [data-testid="stMetricValue"] {
        color: var(--text);
        font-size: 27px;
        font-weight: 700;
    }

    /* Inputs */

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background-color: #111827;
        border-color: var(--border);
    }

    /* Multiselect tags */

    [data-baseweb="tag"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
    }

    /* Dividers */

    hr {
        border-color: var(--border);
        margin-top: 25px;
        margin-bottom: 25px;
    }

    /* Tables */

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Footer */

    .ee-footer {
        color: #94A3B8;
        font-size: 12px;
        border-top: 1px solid #263247;
        padding-top: 18px;
        margin-top: 35px;
        line-height: 1.8;
    }

    </style>
    """, unsafe_allow_html=True)


# =====================================================
# SIDEBAR BRANDING AND NAVIGATION
# =====================================================

def render_sidebar():

    with st.sidebar:

        st.markdown("""
        <div class="ee-brand">EventEcho</div>
        <div class="ee-tagline">Market Intelligence Platform</div>
        <div class="ee-brand-line"></div>
        """, unsafe_allow_html=True)

        st.caption("QUANTITATIVE RESEARCH")

        st.markdown("**Corporate Event Analysis**")

        st.divider()

        st.markdown("#### Navigation")

        project_root = Path(__file__).resolve().parent.parent

        pages = [
            ("Overview", "app.py"),
            ("Event Study", "pages/2_Event_Study.py"),
            ("Market Analysis", "pages/3_Market_Analysis.py"),
            ("Company Comparison", "pages/4_Company_Comparison.py"),
            ("NLP Intelligence", "pages/5_NLP_Intelligence.py"),
            ("Statistical Validation", "pages/6_Statistical_Validation.py"),
            ("Strategy & Backtesting", "pages/7_Strategy_Backtesting.py")
        ]

        for label, path in pages:

            if (project_root / path).exists():
                st.page_link(path, label=label)

        st.divider()

        st.caption(
            "Explore market reactions, abnormal returns, "
            "financial news and statistical evidence."
        )


# =====================================================
# SECTION HEADER
# =====================================================

def render_page_header(title, description):

    st.title(title)

    st.markdown(
        f'<div class="ee-caption">{description}</div>',
        unsafe_allow_html=True
    )


# =====================================================
# FOOTER
# =====================================================

def render_footer():

    st.markdown("""
    <div class="ee-footer">
        <b>EventEcho</b> | Quantitative Event Study & Market Reaction Analysis
        <br>
        Research and educational use only.
        Historical associations do not establish causality.
    </div>
    """, unsafe_allow_html=True)
