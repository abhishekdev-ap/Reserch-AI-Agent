"""
Multi-Agent Research Assistant — Ultra-Premium Streamlit UI v2.0
"""
import os
import sys
import time
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from agents.research_graph import run_research
from rag.rag_engine import rag_query, ingest_text, get_collection_stats, clear_collection

# ─── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Ultra-Premium CSS ─────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ═══ HIDE STREAMLIT DEFAULT CHROME ═══ */
[data-testid="stToolbar"]      { display: none !important; }
.stDeployButton                { display: none !important; }
#MainMenu                      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
footer                         { display: none !important; }

:root {
    --primary: #6366f1;
    --primary-light: #818cf8;
    --primary-dark: #4f46e5;
    --accent: #38bdf8;
    --accent-warm: #f472b6;
    --success: #22c55e;
    --warning: #eab308;
    --surface: rgba(255, 255, 255, 0.03);
    --surface-hover: rgba(255, 255, 255, 0.06);
    --border: rgba(99, 102, 241, 0.15);
    --border-hover: rgba(99, 102, 241, 0.4);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --glow: 0 0 30px rgba(99, 102, 241, 0.15);
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-xl: 28px;
}

* { font-family: 'Inter', sans-serif !important; }
code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

/* ═══ ANIMATED BACKGROUND ═══ */
.stApp {
    background: #05060f;
    min-height: 100vh;
    position: relative;
}
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: 
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(56, 189, 248, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(244, 114, 182, 0.04) 0%, transparent 60%);
    animation: bgPulse 12s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes bgPulse {
    0%   { opacity: 0.6; transform: scale(1); }
    50%  { opacity: 1;   transform: scale(1.03); }
    100% { opacity: 0.7; transform: scale(0.98); }
}

/* ═══ MAIN CONTAINER ═══ */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1200px;
    position: relative;
    z-index: 1;
}

/* ═══ HERO HEADER ═══ */
.hero-v2 {
    background: linear-gradient(145deg, rgba(15, 15, 35, 0.9) 0%, rgba(20, 25, 60, 0.85) 50%, rgba(10, 20, 50, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: var(--radius-xl);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    box-shadow: 
        0 25px 80px rgba(0, 0, 0, 0.4),
        0 0 40px rgba(99, 102, 241, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.hero-v2::before {
    content: '';
    position: absolute;
    top: -100px; right: -60px;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, transparent 70%);
    border-radius: 50%;
    animation: heroOrb1 8s ease-in-out infinite alternate;
}
.hero-v2::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.08) 0%, transparent 70%);
    border-radius: 50%;
    animation: heroOrb2 10s ease-in-out infinite alternate;
}
@keyframes heroOrb1 {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(-30px, 20px) scale(1.15); }
}
@keyframes heroOrb2 {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(20px, -15px) scale(1.1); }
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--primary-light);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
    position: relative;
}
.hero-eyebrow::before {
    content: '';
    width: 24px; height: 2px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    border-radius: 2px;
}
.hero-title-v2 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 0.75rem 0;
    position: relative;
}
.hero-title-v2 .gradient-text {
    background: linear-gradient(135deg, #c4b5fd 0%, #818cf8 30%, #6366f1 60%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% 200%;
    animation: shimmer 4s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}
.hero-desc {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.6;
    max-width: 600px;
    margin-bottom: 1.2rem;
    position: relative;
}
.hero-stack {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    position: relative;
}
.stack-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 14px;
    border-radius: 50px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.04);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}
.stack-chip:hover {
    background: rgba(99, 102, 241, 0.15);
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-1px);
}
.chip-purple { color: #c4b5fd; }
.chip-blue   { color: #7dd3fc; }
.chip-pink   { color: #f9a8d4; }
.chip-green  { color: #86efac; }
.chip-amber  { color: #fde68a; }

/* ═══ METRIC CARDS ═══ */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 1.5rem;
}
.metric-card-v2 {
    background: linear-gradient(145deg, rgba(15, 15, 30, 0.7), rgba(20, 20, 40, 0.5));
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem 1rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.metric-card-v2::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(145deg, transparent 40%, rgba(99, 102, 241, 0.05) 100%);
    opacity: 0;
    transition: opacity 0.35s;
}
.metric-card-v2:hover {
    border-color: var(--border-hover);
    transform: translateY(-3px);
    box-shadow: var(--glow);
}
.metric-card-v2:hover::before { opacity: 1; }
.metric-icon { font-size: 1.5rem; margin-bottom: 6px; display: block; }
.metric-val {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
    display: block;
}
.metric-lbl {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 6px;
    font-weight: 600;
    display: block;
}

/* ═══ GLASS CARD ═══ */
.glass-card {
    background: linear-gradient(145deg, rgba(15, 15, 30, 0.6), rgba(10, 10, 25, 0.4));
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.75rem;
    backdrop-filter: blur(16px);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}
.glass-card:hover {
    border-color: var(--border-hover);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3), var(--glow);
}

/* ═══ SIDEBAR ═══ */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #080a14 0%, #0a0d1a 50%, #060812 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.1) !important;
}
.sidebar-brand {
    text-align: center;
    padding: 1.5rem 0 1rem;
}
.sidebar-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    border-radius: 14px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 10px;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
}
.sidebar-name {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
}
.sidebar-ver {
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 2px;
    letter-spacing: 0.5px;
}
.sidebar-section-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.25rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sidebar-section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.2), transparent);
}

/* ═══ AGENT PIPELINE CARDS ═══ */
.agent-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.agent-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 0 3px 3px 0;
    transition: background 0.3s;
}
.agent-card:hover {
    background: rgba(99, 102, 241, 0.06);
    border-color: rgba(99, 102, 241, 0.2);
    transform: translateX(4px);
}
.agent-card:hover::before { background: var(--primary); }
.agent-avatar {
    width: 34px; height: 34px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.av-search  { background: rgba(56, 189, 248, 0.12); }
.av-critique { background: rgba(244, 114, 182, 0.12); }
.av-synth   { background: rgba(34, 197, 94, 0.12); }
.av-rag     { background: rgba(168, 85, 247, 0.12); }
.agent-meta { flex: 1; min-width: 0; }
.agent-name-v2 {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.2;
}
.agent-role {
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 1px;
}
.agent-badge {
    font-size: 0.6rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 50px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-ready {
    background: rgba(34, 197, 94, 0.1);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.25);
}

/* ═══ STATS WIDGET ═══ */
.stats-widget {
    background: linear-gradient(145deg, rgba(99, 102, 241, 0.08), rgba(56, 189, 248, 0.04));
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: var(--radius-md);
    padding: 1rem;
    text-align: center;
    margin: 1rem 0;
}
.stats-number {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    display: block;
}
.stats-label {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 4px;
    font-weight: 600;
    display: block;
}

/* ═══ POWERED BY ═══ */
.powered-by {
    text-align: center;
    padding: 1rem 0 0.5rem;
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.5px;
}
.powered-by a {
    color: var(--primary-light);
    text-decoration: none;
    font-weight: 600;
}

/* ═══ TABS ═══ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 5px;
    gap: 5px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.3s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
    background: rgba(99, 102, 241, 0.06) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
}

/* ═══ BUTTONS ═══ */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1.6rem !important;
    letter-spacing: 0.4px !important;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
    filter: brightness(1.1) !important;
}
.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* ═══ INPUTS ═══ */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-size: 0.92rem !important;
    transition: all 0.3s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
    background: rgba(255, 255, 255, 0.04) !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ═══ SELECT BOX ═══ */
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 12px !important;
}

/* ═══ SLIDER ═══ */
.stSlider > div > div > div       { background: rgba(99, 102, 241, 0.2) !important; }
.stSlider > div > div > div > div { background: linear-gradient(90deg, var(--primary), var(--accent)) !important; }
[data-testid="stThumbValue"] { color: var(--primary-light) !important; font-weight: 700 !important; }

/* ═══ SECTION HEADERS ═══ */
.section-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.5rem;
}
.section-icon {
    width: 42px; height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
}
.si-search  { background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(99,102,241,0.1)); }
.si-rag     { background: linear-gradient(135deg, rgba(168,85,247,0.15), rgba(244,114,182,0.1)); }
.si-ingest  { background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(56,189,248,0.1)); }
.section-title-text {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
}
.section-subtitle {
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 1.25rem;
}

/* ═══ RESEARCH REPORT CARD ═══ */
.report-wrap {
    background: linear-gradient(145deg, rgba(12, 12, 25, 0.9), rgba(8, 8, 18, 0.7));
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: var(--radius-lg);
    padding: 2rem 2.5rem;
    line-height: 1.85;
    color: var(--text-secondary);
    position: relative;
    overflow: hidden;
}
.report-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--accent), var(--accent-warm));
    border-radius: 3px 3px 0 0;
}
.report-wrap h1, .report-wrap h2, .report-wrap h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
    margin-top: 1.5rem;
}
.report-wrap h2 {
    font-size: 1.3rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(99, 102, 241, 0.12);
}
.report-wrap strong { color: #c4b5fd; }
.report-wrap a { color: var(--accent); }
.report-wrap li { margin-bottom: 0.3rem; }
.report-wrap code {
    background: rgba(99, 102, 241, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
    color: #c4b5fd;
}

/* ═══ CITATION CARD ═══ */
.cite-card {
    background: rgba(10, 10, 25, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-left: 3px solid;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    transition: all 0.3s;
}
.cite-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    background: rgba(15, 15, 35, 0.8);
    transform: translateX(4px);
}
.cite-card.c1 { border-left-color: #818cf8; }
.cite-card.c2 { border-left-color: #38bdf8; }
.cite-card.c3 { border-left-color: #f472b6; }
.cite-card.c4 { border-left-color: #22c55e; }
.cite-card.c5 { border-left-color: #eab308; }
.cite-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.cite-idx {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--primary-light);
}
.cite-score {
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 50px;
    background: rgba(34, 197, 94, 0.1);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.2);
}
.cite-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 3px;
}
.cite-url {
    font-size: 0.72rem;
    color: var(--accent);
    word-break: break-all;
    opacity: 0.8;
}

/* ═══ EMPTY STATE ═══ */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
}
.empty-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    display: block;
    opacity: 0.7;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-10px); }
}
.empty-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
}
.empty-desc {
    font-size: 0.85rem;
    color: var(--text-muted);
    max-width: 400px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ═══ PROGRESS ANIMATION ═══ */
.research-progress {
    background: rgba(10, 10, 25, 0.8);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    margin: 1rem 0;
}
.progress-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    font-size: 0.88rem;
    color: var(--text-muted);
    transition: color 0.3s;
}
.progress-step.active {
    color: var(--text-primary);
}
.progress-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--text-muted);
    flex-shrink: 0;
    transition: all 0.3s;
}
.progress-step.active .progress-dot {
    background: var(--primary);
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
    animation: dotPulse 1s infinite;
}
.progress-step.done .progress-dot {
    background: var(--success);
}
@keyframes dotPulse {
    0%, 100% { transform: scale(1); }
    50%      { transform: scale(1.4); }
}

/* ═══ ALERT OVERRIDES ═══ */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
}

/* ═══ EXPANDER ═══ */
.stExpander {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: rgba(255, 255, 255, 0.01) !important;
    margin-bottom: 10px !important;
}

/* ═══ SCROLLBAR ═══ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { 
    background: rgba(99, 102, 241, 0.3); 
    border-radius: 3px; 
}
::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.6); }

/* ═══ DIVIDER ═══ */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.15) 50%, transparent 100%) !important;
    margin: 1.5rem 0 !important;
}

/* ═══ TIPS BOX ═══ */
.tips-box {
    background: rgba(99, 102, 241, 0.04);
    border: 1px solid rgba(99, 102, 241, 0.1);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
}
.tips-title {
    font-weight: 700;
    color: var(--primary-light);
    font-size: 0.88rem;
    margin-bottom: 0.5rem;
}
.tips-box li {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-bottom: 0.3rem;
    line-height: 1.5;
}

/* ═══ REPORT STATS BAR ═══ */
.report-stats {
    display: flex;
    gap: 16px;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.r-stat {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border);
    border-radius: 50px;
    font-size: 0.78rem;
    color: var(--text-secondary);
}
.r-stat-val {
    font-weight: 700;
    color: var(--primary-light);
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Hide default metric labels */
[data-testid="stMetric"] { display: none !important; }

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ═══ LIGHT MODE OVERRIDES ════════════════════════════════════════════════ */
/* ═══════════════════════════════════════════════════════════════════════════ */

.light-mode {
    --primary: #4f46e5;
    --primary-light: #6366f1;
    --primary-dark: #3730a3;
    --accent: #0284c7;
    --accent-warm: #db2777;
    --surface: rgba(0, 0, 0, 0.02);
    --surface-hover: rgba(0, 0, 0, 0.04);
    --border: rgba(99, 102, 241, 0.18);
    --border-hover: rgba(99, 102, 241, 0.45);
    --text-primary: #1e293b;
    --text-secondary: #475569;
    --text-muted: #94a3b8;
    --glow: 0 4px 20px rgba(99, 102, 241, 0.1);
}

/* Background */
.stApp.light-mode {
    background: #f8fafc !important;
}
.stApp.light-mode::before {
    background:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(99, 102, 241, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(56, 189, 248, 0.04) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(244, 114, 182, 0.03) 0%, transparent 60%) !important;
}

/* Sidebar */
.light-mode section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #f1f5f9 0%, #e8ecf2 50%, #f1f5f9 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.12) !important;
}
.light-mode .sidebar-name { color: #1e293b !important; }
.light-mode .sidebar-ver  { color: #64748b !important; }
.light-mode .sidebar-section-title { color: #64748b !important; }
.light-mode .sidebar-section-title::after {
    background: linear-gradient(90deg, rgba(99,102,241,0.15), transparent) !important;
}

/* Agent cards in sidebar */
.light-mode .agent-card {
    background: rgba(255, 255, 255, 0.7) !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}
.light-mode .agent-card:hover {
    background: rgba(99, 102, 241, 0.06) !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
}
.light-mode .agent-name-v2 { color: #1e293b !important; }
.light-mode .agent-role    { color: #64748b !important; }

/* Stats widget */
.light-mode .stats-widget {
    background: linear-gradient(145deg, rgba(99, 102, 241, 0.06), rgba(56, 189, 248, 0.04)) !important;
    border-color: rgba(99, 102, 241, 0.12) !important;
}

/* Hero header */
.light-mode .hero-v2 {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.9) 0%, rgba(238, 242, 255, 0.85) 50%, rgba(224, 231, 255, 0.9) 100%) !important;
    border-color: rgba(99, 102, 241, 0.15) !important;
    box-shadow: 0 10px 40px rgba(99, 102, 241, 0.08), 0 1px 3px rgba(0, 0, 0, 0.06) !important;
}
.light-mode .hero-v2::before {
    background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 70%) !important;
}
.light-mode .hero-v2::after {
    background: radial-gradient(circle, rgba(56, 189, 248, 0.06) 0%, transparent 70%) !important;
}
.light-mode .hero-eyebrow { color: #4f46e5 !important; }
.light-mode .hero-title-v2 .gradient-text {
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 30%, #6366f1 60%, #0284c7 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}
.light-mode .hero-desc { color: #475569 !important; }
.light-mode .stack-chip {
    background: rgba(255, 255, 255, 0.6) !important;
    border-color: rgba(99, 102, 241, 0.12) !important;
}
.light-mode .chip-purple { color: #4f46e5 !important; }
.light-mode .chip-blue   { color: #0369a1 !important; }
.light-mode .chip-pink   { color: #be185d !important; }
.light-mode .chip-green  { color: #15803d !important; }
.light-mode .chip-amber  { color: #a16207 !important; }

/* Metric cards */
.light-mode .metric-card-v2 {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.9), rgba(248, 250, 252, 0.8)) !important;
    border-color: rgba(99, 102, 241, 0.1) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
}
.light-mode .metric-card-v2:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.1) !important;
}
.light-mode .metric-val { color: #1e293b !important; }
.light-mode .metric-lbl { color: #64748b !important; }

/* Tabs */
.light-mode .stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.6) !important;
    border-color: rgba(99, 102, 241, 0.1) !important;
}
.light-mode .stTabs [data-baseweb="tab"] {
    color: #64748b !important;
}
.light-mode .stTabs [data-baseweb="tab"]:hover {
    color: #4f46e5 !important;
    background: rgba(99, 102, 241, 0.05) !important;
}
.light-mode .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #4338ca) !important;
    color: white !important;
}

/* Section headers */
.light-mode .section-title-text { color: #1e293b !important; }
.light-mode .section-subtitle   { color: #475569 !important; }
.light-mode .si-search  { background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(99,102,241,0.08)) !important; }
.light-mode .si-rag     { background: linear-gradient(135deg, rgba(168,85,247,0.1), rgba(244,114,182,0.08)) !important; }
.light-mode .si-ingest  { background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(56,189,248,0.08)) !important; }

/* Inputs */
.light-mode .stTextInput > div > div > input,
.light-mode .stTextArea > div > div > textarea {
    background: rgba(255, 255, 255, 0.8) !important;
    border-color: rgba(99, 102, 241, 0.2) !important;
    color: #1e293b !important;
}
.light-mode .stTextInput > div > div > input:focus,
.light-mode .stTextArea > div > div > textarea:focus {
    background: white !important;
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
}
.light-mode .stTextInput > div > div > input::placeholder,
.light-mode .stTextArea > div > div > textarea::placeholder {
    color: #94a3b8 !important;
}

/* Buttons */
.light-mode .stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.25) !important;
}
.light-mode .stButton > button:hover {
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.35) !important;
}

/* Report wrap */
.light-mode .report-wrap {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.9)) !important;
    border-color: rgba(99, 102, 241, 0.12) !important;
    color: #334155 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
}
.light-mode .report-wrap h1,
.light-mode .report-wrap h2,
.light-mode .report-wrap h3 {
    color: #1e293b !important;
}
.light-mode .report-wrap h2 {
    border-bottom-color: rgba(99, 102, 241, 0.1) !important;
}
.light-mode .report-wrap strong { color: #4f46e5 !important; }
.light-mode .report-wrap code {
    background: rgba(99, 102, 241, 0.08) !important;
    color: #4338ca !important;
}

/* Citation cards */
.light-mode .cite-card {
    background: rgba(255, 255, 255, 0.8) !important;
    border-color: rgba(99, 102, 241, 0.1) !important;
}
.light-mode .cite-card:hover {
    background: rgba(248, 250, 252, 1) !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
}
.light-mode .cite-title { color: #1e293b !important; }
.light-mode .cite-idx   { color: #4f46e5 !important; }
.light-mode .cite-url   { color: #0284c7 !important; }

/* Empty state */
.light-mode .empty-title { color: #475569 !important; }
.light-mode .empty-desc  { color: #94a3b8 !important; }

/* Research progress */
.light-mode .research-progress {
    background: rgba(255, 255, 255, 0.8) !important;
    border-color: rgba(99, 102, 241, 0.12) !important;
}
.light-mode .progress-step       { color: #94a3b8 !important; }
.light-mode .progress-step.active { color: #1e293b !important; }

/* Report stats */
.light-mode .r-stat {
    background: rgba(255, 255, 255, 0.6) !important;
    border-color: rgba(99, 102, 241, 0.1) !important;
    color: #475569 !important;
}
.light-mode .r-stat-val { color: #4f46e5 !important; }

/* Tips box */
.light-mode .tips-box {
    background: rgba(99, 102, 241, 0.04) !important;
    border-color: rgba(99, 102, 241, 0.1) !important;
}
.light-mode .tips-title { color: #4f46e5 !important; }
.light-mode .tips-box li { color: #475569 !important; }

/* Expander */
.light-mode .stExpander {
    border-color: rgba(99, 102, 241, 0.1) !important;
    background: rgba(255, 255, 255, 0.5) !important;
}

/* Powered by */
.light-mode .powered-by { color: #94a3b8 !important; }

/* Selectbox */
.light-mode .stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.8) !important;
    border-color: rgba(99, 102, 241, 0.15) !important;
}

/* Slider */
.light-mode .stSlider > div > div > div { background: rgba(99, 102, 241, 0.15) !important; }
.light-mode [data-testid="stThumbValue"] { color: #4f46e5 !important; }

/* Divider */
.light-mode hr {
    background: linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.12) 50%, transparent 100%) !important;
}

/* Scrollbar */
.light-mode ::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.2) !important;
}

/* Glass card */
.light-mode .glass-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.8), rgba(248, 250, 252, 0.7)) !important;
    border-color: rgba(99, 102, 241, 0.1) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04) !important;
}

/* ═══ THEME TOGGLE BUTTON ═══ */
.theme-toggle-wrap {
    display: flex;
    justify-content: center;
    margin: 0.5rem 0;
}
.theme-toggle-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text-secondary);
    letter-spacing: 0.3px;
}
.theme-toggle-btn:hover {
    border-color: var(--border-hover);
    background: var(--surface-hover);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ─────────────────────────────────────────────────────────

if "research_history" not in st.session_state:
    st.session_state.research_history = []
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ─── Apply Theme Class via JS ──────────────────────────────────────────────────

if st.session_state.dark_mode:
    _theme_js = '<script>parent.document.querySelector(".stApp").classList.remove("light-mode");</script>'
else:
    _theme_js = '<script>parent.document.querySelector(".stApp").classList.add("light-mode");</script>'
components.html(_theme_js, height=0, width=0)


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🔬</div>
        <div class="sidebar-name">Research Assistant</div>
        <div class="sidebar-ver">v2.0 · Multi-Agent · RAG · LangGraph</div>
    </div>
    """, unsafe_allow_html=True)

    # Theme Toggle
    theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
    theme_label = "Dark Mode" if st.session_state.dark_mode else "Light Mode"
    if st.button(f"{theme_icon}  {theme_label}", key="theme_toggle", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("---")

    # Agent Config
    st.markdown('<div class="sidebar-section-title">⚙️ Configuration</div>', unsafe_allow_html=True)
    max_iterations = st.slider(
        "Research Depth",
        min_value=1, max_value=3, value=2,
        help="Higher = deeper research with more iterations, but slower"
    )

    st.markdown("---")

    # Agent Pipeline
    st.markdown('<div class="sidebar-section-title">🤖 Agent Pipeline</div>', unsafe_allow_html=True)
    
    agents_info = [
        ("🔍", "Search Agent", "Web discovery via Tavily", "av-search"),
        ("🧠", "Critique Agent", "Gap & bias analysis", "av-critique"),
        ("📝", "Synthesis Agent", "Report generation", "av-synth"),
        ("💾", "RAG Engine", "Knowledge retrieval", "av-rag"),
    ]
    for icon, name, role, av_cls in agents_info:
        st.markdown(f"""
        <div class="agent-card">
            <div class="agent-avatar {av_cls}">{icon}</div>
            <div class="agent-meta">
                <div class="agent-name-v2">{name}</div>
                <div class="agent-role">{role}</div>
            </div>
            <span class="agent-badge badge-ready">Ready</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # KB Stats
    st.markdown('<div class="sidebar-section-title">📊 Knowledge Base</div>', unsafe_allow_html=True)
    try:
        stats = get_collection_stats()
        kb_chunks = stats.get("total_chunks", 0)
    except:
        kb_chunks = 0

    st.markdown(f"""
    <div class="stats-widget">
        <span class="stats-number">{kb_chunks}</span>
        <span class="stats-label">Indexed Chunks</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        try:
            clear_collection()
            st.success("✅ Knowledge base cleared!")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")

    st.markdown("""
    <div class="powered-by">
        Built with<br/>
        <span style="color:#818cf8;">Gemini 2.5 Flash</span> · 
        <span style="color:#7dd3fc;">Tavily</span> · 
        <span style="color:#c4b5fd;">LangGraph</span> · 
        <span style="color:#86efac;">ChromaDB</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Header ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-v2">
    <div class="hero-eyebrow">Multi-Agent AI System</div>
    <div class="hero-title-v2">
        <span class="gradient-text">Research Assistant</span>
    </div>
    <div class="hero-desc">
        An autonomous research system that orchestrates specialized AI agents to search, 
        analyze, critique, and synthesize comprehensive reports — with full citations.
    </div>
    <div class="hero-stack">
        <span class="stack-chip chip-purple">🤖 LangGraph Orchestration</span>
        <span class="stack-chip chip-blue">✨ Gemini 2.5 Flash</span>
        <span class="stack-chip chip-green">🌐 Tavily Deep Search</span>
        <span class="stack-chip chip-pink">🗄️ ChromaDB RAG</span>
        <span class="stack-chip chip-amber">📚 Citation Aware</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Metrics Row ─────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card-v2">
        <span class="metric-icon">🔬</span>
        <span class="metric-val">{len(st.session_state.research_history)}</span>
        <span class="metric-lbl">Researches</span>
    </div>
    <div class="metric-card-v2">
        <span class="metric-icon">💬</span>
        <span class="metric-val">{len(st.session_state.rag_history)}</span>
        <span class="metric-lbl">RAG Queries</span>
    </div>
    <div class="metric-card-v2">
        <span class="metric-icon">📦</span>
        <span class="metric-val">{kb_chunks}</span>
        <span class="metric-lbl">KB Chunks</span>
    </div>
    <div class="metric-card-v2">
        <span class="metric-icon">⚡</span>
        <span class="metric-val">4</span>
        <span class="metric-lbl">Active Agents</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍  Research", "💬  RAG Q&A", "📄  Ingest Documents"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("""
    <div class="section-head">
        <div class="section-icon si-search">🔍</div>
        <div class="section-title-text">Start a Research Task</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter any topic — the multi-agent pipeline will autonomously search, analyze, and synthesize a comprehensive report with citations.</div>', unsafe_allow_html=True)

    query = st.text_area(
        "Research Query",
        placeholder="e.g. 'What are the latest breakthroughs in quantum computing?' or 'Analyze the impact of AI on healthcare in 2025'",
        height=110,
        label_visibility="collapsed",
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        run_btn = st.button("🚀 Run Research", use_container_width=True, type="primary")
    with col_btn2:
        clear_hist = st.button("🗑️ Clear History", use_container_width=True)

    if clear_hist:
        st.session_state.research_history = []
        st.rerun()

    if run_btn:
        if not query.strip():
            st.warning("⚠️ Please enter a research query.")
        else:
            with st.spinner(""):
                st.markdown("""
                <div class="research-progress">
                    <div class="progress-step active">
                        <div class="progress-dot"></div>
                        <span>🔍 Search Agent — Gathering data from multiple web sources via Tavily...</span>
                    </div>
                    <div class="progress-step">
                        <div class="progress-dot"></div>
                        <span>🧠 Critique Agent — Analyzing for gaps, biases and missing perspectives...</span>
                    </div>
                    <div class="progress-step">
                        <div class="progress-dot"></div>
                        <span>📝 Synthesis Agent — Compiling final structured report...</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                try:
                    start_time = time.time()
                    result = run_research(query, max_iterations=max_iterations)
                    elapsed = time.time() - start_time

                    def _safe_msg_text(msg):
                        c = getattr(msg, 'content', None)
                        if isinstance(c, list):
                            return '\n'.join(
                                b.get('text', '') if isinstance(b, dict) else str(b) for b in c
                            )
                        return str(c) if c else ''

                    messages = result.get("messages", [])
                    final_content = result.get("final_report", "")
                    if not final_content:
                        for msg in reversed(messages):
                            text = _safe_msg_text(msg)
                            if text and len(text) > 100:
                                final_content = text
                                break

                    st.session_state.research_history.insert(0, {
                        "query": query,
                        "report": final_content,
                        "iterations": result.get("iteration", 0),
                        "elapsed": round(elapsed, 1),
                        "messages": len(messages),
                    })

                    # Auto-ingest into RAG
                    if final_content:
                        try:
                            ingest_text(final_content, source=f"research:{query[:50]}")
                        except:
                            pass

                    st.success(f"✅ Research completed in **{elapsed:.1f}s** — {len(messages)} agent messages processed!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Research failed: {str(e)}")
                    st.info("💡 Check API keys and internet connection, or try again in a few seconds if rate-limited.")

    # Display research history
    if st.session_state.research_history:
        st.markdown("---")
        st.markdown("""
        <div class="section-head">
            <div class="section-icon si-search">📋</div>
            <div class="section-title-text">Research Reports</div>
        </div>
        """, unsafe_allow_html=True)

        for i, item in enumerate(st.session_state.research_history):
            label = f"{'🔬' if i == 0 else '📄'} {item['query'][:80]}{'...' if len(item['query']) > 80 else ''}"
            with st.expander(label, expanded=(i == 0)):
                # Stats bar
                st.markdown(f"""
                <div class="report-stats">
                    <div class="r-stat">🔄 Iterations: <span class="r-stat-val">{item['iterations']}</span></div>
                    <div class="r-stat">💬 Messages: <span class="r-stat-val">{item['messages']}</span></div>
                    <div class="r-stat">⏱️ Time: <span class="r-stat-val">{item['elapsed']}s</span></div>
                </div>
                """, unsafe_allow_html=True)

                if item["report"]:
                    st.markdown(f'<div class="report-wrap">{item["report"]}</div>', unsafe_allow_html=True)
                    st.markdown("")
                    st.markdown(item["report"])
                else:
                    st.info("No report content was generated. Try a more specific query.")
    else:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">🧪</span>
            <div class="empty-title">Ready to research anything</div>
            <div class="empty-desc">Enter a query above and click Run Research. The multi-agent system will autonomously search, critique, and compile a report.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: RAG Q&A
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div class="section-head">
        <div class="section-icon si-rag">💬</div>
        <div class="section-title-text">Query Your Knowledge Base</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Ask follow-up questions about previously researched topics. Answers are generated with source citations from the knowledge base.</div>', unsafe_allow_html=True)

    rag_q = st.text_input(
        "Question",
        placeholder="e.g. 'What are the key benefits of quantum computing?' or 'Summarize the findings about AI governance'",
        label_visibility="collapsed",
    )

    col_r1, col_r2, _ = st.columns([1, 1, 4])
    with col_r1:
        ask_btn = st.button("💬 Ask Question", use_container_width=True, type="primary")
    with col_r2:
        rag_k = st.selectbox("Top-K", [3, 5, 8, 10], index=1, label_visibility="collapsed")

    if ask_btn:
        if not rag_q.strip():
            st.warning("⚠️ Please enter a question.")
        elif kb_chunks == 0:
            st.warning("📭 Knowledge base is empty. Run a research query first to populate it.")
        else:
            with st.spinner("🔍 Searching knowledge base..."):
                try:
                    result = rag_query(rag_q, k=rag_k)
                    answer_text = result.get("answer", "")
                    if isinstance(answer_text, list):
                        answer_text = '\n'.join(
                            b.get('text', '') if isinstance(b, dict) else str(b) for b in answer_text
                        )
                    st.session_state.rag_history.insert(0, {
                        "question": rag_q,
                        "answer": str(answer_text),
                        "sources": result.get("sources", []),
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ RAG query failed: {str(e)}")

    if st.session_state.rag_history:
        st.markdown("---")
        for i, item in enumerate(st.session_state.rag_history):
            label = f"{'❓' if i == 0 else '💭'} {item['question'][:70]}{'...' if len(item['question']) > 70 else ''}"
            with st.expander(label, expanded=(i == 0)):
                tab_a, tab_s = st.tabs(["💡 Answer", "📎 Sources"])

                with tab_a:
                    st.markdown(item["answer"])

                with tab_s:
                    if item["sources"]:
                        for j, src in enumerate(item["sources"]):
                            color_cls = f"c{(j % 5) + 1}"
                            st.markdown(f"""
                            <div class="cite-card {color_cls}">
                                <div class="cite-head">
                                    <span class="cite-idx">[{src['index']}]</span>
                                    <span class="cite-score">Score: {src['score']}</span>
                                </div>
                                <div class="cite-title">{src.get('title') or 'Untitled Source'}</div>
                                <div class="cite-url">🔗 {src['source']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No sources available for this answer.")
    else:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">💬</span>
            <div class="empty-title">Your knowledge base awaits</div>
            <div class="empty-desc">Run a research query first to populate the knowledge base, then ask follow-up questions here with source citations.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: INGEST DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("""
    <div class="section-head">
        <div class="section-icon si-ingest">📄</div>
        <div class="section-title-text">Ingest Custom Documents</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Add your own text to the knowledge base. Ingested content will be chunked, embedded, and made available for RAG queries.</div>', unsafe_allow_html=True)

    col_i1, col_i2 = st.columns([3, 1])
    with col_i1:
        ingest_source = st.text_input(
            "Source Label",
            placeholder="e.g. 'research_paper', 'meeting_notes', 'article_summary'",
            value="manual"
        )
    with col_i2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    ingest_content = st.text_area(
        "Text Content",
        placeholder="Paste any text here — articles, research papers, notes, documentation, transcripts...",
        height=260,
    )

    if st.button("📥 Ingest into Knowledge Base", type="primary"):
        if not ingest_content.strip():
            st.warning("⚠️ Please provide some text to ingest.")
        else:
            with st.spinner("📥 Chunking, embedding, and indexing..."):
                try:
                    n_chunks = ingest_text(ingest_content, source=ingest_source)
                    st.success(f"✅ Successfully ingested **{n_chunks} chunks** from '{ingest_source}' into the knowledge base!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {str(e)}")

    st.markdown("""
    <div class="tips-box">
        <div class="tips-title">💡 Tips for best results</div>
        <ul>
            <li>Ingest multiple related documents to get richer, cross-referenced answers</li>
            <li>Each research run automatically ingests its report into the knowledge base</li>
            <li>The RAG engine searches all ingested content when you query</li>
            <li>Use descriptive source labels to track where information came from</li>
            <li>Larger texts are automatically split into 1000-character chunks with 200-char overlap</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
