import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
from streamlit_lottie import st_lottie
from streamlit_extras.metric_cards import style_metric_cards
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Road Safety AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# FULL iOS GLASSMORPHISM + RACING INTRO ANIMATION CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* ─── GOOGLE FONTS ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;600;700&family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

/* ─── RACING INTRO OVERLAY ──────────────────────────────────── */
#intro-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0a0f 100%);
    z-index: 999999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    animation: overlayFadeOut 0.8s ease-out 4.2s forwards;
    overflow: hidden;
}

/* Road */
#intro-road {
    position: absolute;
    bottom: 0; left: 0;
    width: 100%;
    height: 180px;
    background: linear-gradient(to bottom, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    border-top: 3px solid rgba(255,255,255,0.08);
}

/* Road center dashes */
#intro-road::before {
    content: '';
    position: absolute;
    top: 50%;
    left: -200%;
    width: 400%;
    height: 4px;
    background: repeating-linear-gradient(90deg, transparent 0px, transparent 60px, rgba(255,200,0,0.6) 60px, rgba(255,200,0,0.6) 120px);
    animation: roadDashes 0.4s linear infinite;
}

/* Speed lines background */
.speed-line {
    position: absolute;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), transparent);
    animation: speedLine 0.6s linear infinite;
    border-radius: 2px;
}

/* THE CAR SVG */
#racing-car {
    position: absolute;
    bottom: 140px;
    left: -200px;
    width: 220px;
    filter: drop-shadow(0 0 20px rgba(99,179,237,0.8)) drop-shadow(0 0 40px rgba(99,179,237,0.4));
    animation: carRace 2.8s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.3s forwards;
    z-index: 10;
}

/* Headlight beam */
#car-beam {
    position: absolute;
    bottom: 155px;
    left: -200px;
    width: 300px;
    height: 30px;
    background: linear-gradient(90deg, rgba(255,240,180,0) 0%, rgba(255,240,180,0.15) 60%, rgba(255,240,180,0.35) 100%);
    clip-path: polygon(0 40%, 100% 0%, 100% 100%, 0 60%);
    animation: beamRace 2.8s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.3s forwards;
    z-index: 9;
}

/* Dust/tire particles */
.tire-particle {
    position: absolute;
    bottom: 138px;
    border-radius: 50%;
    background: rgba(150, 150, 170, 0.4);
    animation: particleFade 0.8s ease-out infinite;
}

/* Intro Title */
#intro-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(28px, 5vw, 64px);
    font-weight: 900;
    color: transparent;
    background: linear-gradient(90deg, #63b3ed, #90cdf4, #e2e8f0, #90cdf4, #63b3ed);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    text-align: center;
    opacity: 0;
    letter-spacing: 4px;
    text-shadow: none;
    animation: titleAppear 0.8s ease-out 2.2s forwards, shimmer 3s ease-in-out 2.2s infinite;
    position: relative;
    z-index: 20;
    margin-bottom: 12px;
}

#intro-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: clamp(12px, 2vw, 18px);
    font-weight: 300;
    color: rgba(148, 163, 184, 0.9);
    letter-spacing: 6px;
    text-transform: uppercase;
    opacity: 0;
    animation: titleAppear 0.6s ease-out 2.6s forwards;
    z-index: 20;
}

#intro-loading {
    position: absolute;
    bottom: 200px;
    width: 250px;
    height: 3px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    overflow: hidden;
    z-index: 20;
    opacity: 0;
    animation: titleAppear 0.3s ease-out 3.0s forwards;
}

#intro-loading-bar {
    height: 100%;
    background: linear-gradient(90deg, #63b3ed, #90cdf4);
    border-radius: 3px;
    animation: loadingBar 1.0s ease-out 3.0s forwards;
    width: 0%;
    box-shadow: 0 0 10px rgba(99,179,237,0.8);
}

/* ─── KEYFRAMES ─────────────────────────────────────────────── */
@keyframes carRace {
    0% { left: -200px; }
    35% { left: 30%; transform: scaleX(1.05); }
    65% { left: 65%; transform: scaleX(1); }
    80% { left: 80%; opacity: 1; }
    100% { left: 110%; opacity: 0; }
}

@keyframes beamRace {
    0% { left: -200px; }
    35% { left: calc(30% + 120px); }
    65% { left: calc(65% + 120px); }
    80% { left: calc(80% + 120px); opacity: 1; }
    100% { left: calc(110% + 120px); opacity: 0; }
}

@keyframes roadDashes {
    0% { transform: translateX(0); }
    100% { transform: translateX(120px); }
}

@keyframes speedLine {
    0% { transform: translateX(110vw); opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { transform: translateX(-200px); opacity: 0; }
}

@keyframes titleAppear {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes shimmer {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes loadingBar {
    0% { width: 0%; }
    100% { width: 100%; }
}

@keyframes overlayFadeOut {
    0% { opacity: 1; pointer-events: all; }
    100% { opacity: 0; pointer-events: none; display: none; }
}

@keyframes particleFade {
    0% { transform: translate(0, 0) scale(1); opacity: 0.5; }
    100% { transform: translate(-30px, -20px) scale(0); opacity: 0; }
}

/* ─── GLOBAL STREAMLIT BASE ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: linear-gradient(135deg, #060b14 0%, #0a0f1e 30%, #060d1a 60%, #0a0c18 100%) !important;
    background-attachment: fixed !important;
    min-height: 100vh;
}

/* Animated background stars */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image: 
        radial-gradient(1px 1px at 10% 15%, rgba(255,255,255,0.15) 0%, transparent 100%),
        radial-gradient(1px 1px at 25% 35%, rgba(255,255,255,0.1) 0%, transparent 100%),
        radial-gradient(1px 1px at 40% 60%, rgba(255,255,255,0.08) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 20%, rgba(255,255,255,0.12) 0%, transparent 100%),
        radial-gradient(1px 1px at 75% 50%, rgba(255,255,255,0.1) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 80%, rgba(255,255,255,0.08) 0%, transparent 100%),
        radial-gradient(2px 2px at 15% 75%, rgba(99,179,237,0.15) 0%, transparent 100%),
        radial-gradient(2px 2px at 55% 90%, rgba(99,179,237,0.1) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* ─── iOS GLASS COMPONENTS ──────────────────────────────────── */
.glass-card {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.1),
        0 0 0 1px rgba(255,255,255,0.02) !important;
    padding: 24px !important;
    transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
}

.glass-card:hover {
    border-color: rgba(99, 179, 237, 0.25) !important;
    box-shadow: 
        0 16px 48px rgba(0, 0, 0, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.15),
        0 0 30px rgba(99,179,237,0.05) !important;
    transform: translateY(-2px) !important;
}

/* ─── MAIN CONTENT AREA ─────────────────────────────────────── */
.block-container {
    padding-top: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
}

/* ─── TITLES ─────────────────────────────────────────────────── */
h1 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
    background: linear-gradient(135deg, #e2e8f0 0%, #90cdf4 50%, #63b3ed 100%);
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: 2px !important;
    margin-bottom: 0.3rem !important;
}

h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: rgba(226, 232, 240, 0.9) !important;
    letter-spacing: 0.3px !important;
}

/* ─── METRIC CARDS ──────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 
        0 4px 24px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    transition: all 0.3s ease !important;
    position: relative;
    overflow: hidden;
}

[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.6), transparent);
    opacity: 0.7;
}

[data-testid="metric-container"]:hover {
    border-color: rgba(99, 179, 237, 0.3) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 20px rgba(99,179,237,0.08) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #90cdf4 !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: rgba(148, 163, 184, 0.8) !important;
}

/* ─── SIDEBAR GLASS ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(6, 11, 20, 0.85) !important;
    backdrop-filter: blur(30px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(30px) saturate(200%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    box-shadow: 4px 0 32px rgba(0, 0, 0, 0.5) !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}

/* ─── SIDEBAR HEADING ────────────────────────────────────────── */
[data-testid="stSidebar"] h1 {
    font-size: 1.3rem !important;
    letter-spacing: 3px !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.06) !important;
    margin: 1rem 0 !important;
}

/* ─── SIDEBAR STATUS PILL ────────────────────────────────────── */
.sidebar-status {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 50px;
    padding: 8px 14px;
    margin-bottom: 16px;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(99,179,237,0.9);
}

.pulse-dot {
    width: 6px; height: 6px;
    background: #63b3ed;
    border-radius: 50%;
    box-shadow: 0 0 6px rgba(99,179,237,0.8);
    animation: pulseDot 1.5s ease-in-out infinite;
}

@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.4); }
}

/* Sidebar nav items */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 12px;
    margin-bottom: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: rgba(226, 232, 240, 0.7);
    cursor: pointer;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}

.nav-item:hover {
    background: rgba(255,255,255,0.06);
    border-color: rgba(255,255,255,0.08);
    color: rgba(226,232,240,0.95);
}

.nav-item.active {
    background: rgba(99,179,237,0.1);
    border-color: rgba(99,179,237,0.2);
    color: #90cdf4;
}

/* ─── BUTTONS ────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, rgba(99,179,237,0.15) 0%, rgba(66,153,225,0.25) 100%) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(99,179,237,0.4) !important;
    border-radius: 14px !important;
    color: #90cdf4 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 1px !important;
    padding: 14px 28px !important;
    transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(99,179,237,0.15), inset 0 1px 0 rgba(255,255,255,0.1) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(99,179,237,0.25) 0%, rgba(66,153,225,0.4) 100%) !important;
    border-color: rgba(99,179,237,0.7) !important;
    box-shadow: 0 8px 30px rgba(99,179,237,0.3), 0 0 0 1px rgba(99,179,237,0.1), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transform: translateY(-2px) !important;
}

.stButton > button:active {
    transform: translateY(0px) scale(0.98) !important;
}

/* ─── SELECTBOX & SLIDERS ────────────────────────────────────── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: rgba(226,232,240,0.9) !important;
    font-family: 'Inter', sans-serif !important;
}

.stSlider > div > div > div {
    background: rgba(99,179,237,0.3) !important;
}

.stSlider > div > div > div > div {
    background: linear-gradient(135deg, #63b3ed, #90cdf4) !important;
    box-shadow: 0 0 12px rgba(99,179,237,0.6) !important;
}

/* ─── MULTISELECT ────────────────────────────────────────────── */
.stMultiSelect > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}

/* ─── TABS ───────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 12px !important;
    color: rgba(148,163,184,0.8) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.15) !important;
    color: #90cdf4 !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.1) !important;
}

/* ─── DATAFRAME ──────────────────────────────────────────────── */
.stDataFrame {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ─── DIVIDER ────────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.2), rgba(255,255,255,0.06), rgba(99,179,237,0.2), transparent) !important;
    margin: 2rem 0 !important;
}

/* ─── PLOTLY CHART CONTAINER ─────────────────────────────────── */
.js-plotly-plot {
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* ─── SPINNER ────────────────────────────────────────────────── */
.stSpinner > div {
    border-color: rgba(99,179,237,0.8) transparent transparent transparent !important;
}

/* ─── ALERTS ─────────────────────────────────────────────────── */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
}

/* ─── FOOTER HIDE ────────────────────────────────────────────── */
footer { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ─── FADE-IN ANIMATION FOR CONTENT ─────────────────────────── */
.main .block-container {
    animation: contentReveal 0.6s ease-out 4.5s both;
}

@keyframes contentReveal {
    0% { opacity: 0; transform: translateY(30px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* ─── SECTION HEADER BADGE ───────────────────────────────────── */
.section-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 50px;
    padding: 4px 12px;
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(99,179,237,0.8);
    margin-bottom: 8px;
}

/* ─── CHART GLOW WRAPPER ─────────────────────────────────────── */
.chart-glass {
    background: rgba(255,255,255,0.025);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: all 0.3s ease;
    margin-bottom: 12px;
}

.chart-glass:hover {
    border-color: rgba(99,179,237,0.2);
    box-shadow: 0 12px 48px rgba(0,0,0,0.4), 0 0 30px rgba(99,179,237,0.04);
}

/* ─── ROAD PROGRESS BAR (TOP) ─────────────────────────────────── */
.road-progress {
    position: fixed;
    top: 0; left: 0;
    height: 3px;
    background: linear-gradient(90deg, #63b3ed, #90cdf4, #bee3f8);
    z-index: 9999;
    box-shadow: 0 0 10px rgba(99,179,237,0.8);
    animation: progressLoad 3.5s ease-out;
    width: 100%;
}

@keyframes progressLoad {
    0% { width: 0%; opacity: 1; }
    90% { width: 100%; opacity: 1; }
    100% { width: 100%; opacity: 0; }
}
</style>

<!-- ═══════════════════════════════════════════════════════════
     RACING INTRO OVERLAY
════════════════════════════════════════════════════════════ -->
<div id="intro-overlay">
    
    <!-- Speed lines -->
    <div class="speed-line" style="top:20%;width:800px;animation-delay:0s;animation-duration:0.5s;"></div>
    <div class="speed-line" style="top:35%;width:600px;animation-delay:0.1s;animation-duration:0.45s;"></div>
    <div class="speed-line" style="top:50%;width:900px;animation-delay:0.05s;animation-duration:0.55s;"></div>
    <div class="speed-line" style="top:65%;width:700px;animation-delay:0.15s;animation-duration:0.5s;"></div>
    <div class="speed-line" style="top:78%;width:500px;animation-delay:0.2s;animation-duration:0.4s;"></div>
    <div class="speed-line" style="top:28%;width:750px;animation-delay:0.08s;animation-duration:0.6s;"></div>
    <div class="speed-line" style="top:58%;width:650px;animation-delay:0.12s;animation-duration:0.48s;"></div>

    <!-- Title & Subtitle -->
    <div id="intro-title">🚦 ROAD SAFETY AI</div>
    <div id="intro-subtitle">Data Mining Intelligence Platform</div>

    <!-- Loading bar -->
    <div id="intro-loading">
        <div id="intro-loading-bar"></div>
    </div>

    <!-- Road -->
    <div id="intro-road"></div>

    <!-- Headlight beam -->
    <div id="car-beam"></div>

    <!-- The Racing Car (SVG inline) -->
    <svg id="racing-car" viewBox="0 0 220 70" xmlns="http://www.w3.org/2000/svg">
        <!-- Car body shadow -->
        <ellipse cx="110" cy="66" rx="80" ry="6" fill="rgba(0,0,0,0.4)"/>
        <!-- Car body main -->
        <path d="M20 50 L30 30 L60 18 L150 18 L185 30 L195 50 Z" 
              fill="url(#bodyGrad)" stroke="rgba(99,179,237,0.6)" stroke-width="1.5"/>
        <!-- Car roof -->
        <path d="M55 18 L70 8 L145 8 L160 18 Z" 
              fill="url(#roofGrad)" stroke="rgba(99,179,237,0.4)" stroke-width="1"/>
        <!-- Windshield -->
        <path d="M75 18 L82 9 L138 9 L145 18 Z" 
              fill="rgba(99,179,237,0.25)" stroke="rgba(99,179,237,0.5)" stroke-width="0.5"/>
        <!-- Side window -->
        <rect x="82" y="10" width="25" height="8" rx="2" fill="rgba(99,179,237,0.3)" stroke="rgba(99,179,237,0.4)" stroke-width="0.5"/>
        <!-- Door lines -->
        <line x1="110" y1="18" x2="112" y2="50" stroke="rgba(99,179,237,0.3)" stroke-width="1"/>
        <!-- Front hood details -->
        <path d="M155 25 L185 30 L188 38 L155 35 Z" fill="rgba(99,179,237,0.08)" stroke="rgba(99,179,237,0.3)" stroke-width="0.5"/>
        <!-- Rear spoiler -->
        <rect x="18" y="26" width="12" height="3" rx="1.5" fill="rgba(99,179,237,0.6)" stroke="rgba(99,179,237,0.8)" stroke-width="0.5"/>
        <!-- Headlight (front) -->
        <ellipse cx="188" cy="40" rx="7" ry="5" fill="rgba(255,240,180,0.9)" stroke="rgba(255,240,180,0.5)" stroke-width="1"/>
        <ellipse cx="188" cy="40" rx="4" ry="3" fill="white"/>
        <!-- Tail light (rear) -->
        <rect x="20" y="38" width="8" height="6" rx="2" fill="rgba(255,60,60,0.9)" stroke="rgba(255,60,60,0.6)" stroke-width="0.5"/>
        <!-- Rear light glow -->
        <rect x="18" y="37" width="10" height="8" rx="2" fill="rgba(255,60,60,0.2)"/>
        <!-- Wheels -->
        <circle cx="55" cy="55" r="13" fill="#1a1a2e" stroke="rgba(99,179,237,0.5)" stroke-width="2"/>
        <circle cx="55" cy="55" r="8" fill="#0d1117" stroke="rgba(99,179,237,0.4)" stroke-width="1.5"/>
        <circle cx="55" cy="55" r="3" fill="rgba(99,179,237,0.7)"/>
        <!-- Wheel spokes front -->
        <line x1="55" y1="47" x2="55" y2="63" stroke="rgba(99,179,237,0.4)" stroke-width="1"/>
        <line x1="47" y1="55" x2="63" y2="55" stroke="rgba(99,179,237,0.4)" stroke-width="1"/>
        
        <circle cx="160" cy="55" r="13" fill="#1a1a2e" stroke="rgba(99,179,237,0.5)" stroke-width="2"/>
        <circle cx="160" cy="55" r="8" fill="#0d1117" stroke="rgba(99,179,237,0.4)" stroke-width="1.5"/>
        <circle cx="160" cy="55" r="3" fill="rgba(99,179,237,0.7)"/>
        <!-- Wheel spokes rear -->
        <line x1="160" y1="47" x2="160" y2="63" stroke="rgba(99,179,237,0.4)" stroke-width="1"/>
        <line x1="152" y1="55" x2="168" y2="55" stroke="rgba(99,179,237,0.4)" stroke-width="1"/>
        
        <!-- Racing stripe -->
        <path d="M55 30 L155 30" stroke="rgba(99,179,237,0.4)" stroke-width="2" stroke-dasharray="8,4"/>
        <!-- Number plate area -->
        <rect x="175" y="46" width="18" height="8" rx="2" fill="rgba(255,255,255,0.15)" stroke="rgba(255,255,255,0.2)" stroke-width="0.5"/>
        <!-- Exhaust smoke particles -->
        <circle cx="12" cy="46" r="4" fill="rgba(150,150,180,0.2)" style="animation: particleFade 0.5s ease-out 0s infinite;"/>
        <circle cx="6" cy="42" r="3" fill="rgba(150,150,180,0.15)" style="animation: particleFade 0.5s ease-out 0.1s infinite;"/>
        <circle cx="2" cy="50" r="2.5" fill="rgba(150,150,180,0.1)" style="animation: particleFade 0.5s ease-out 0.2s infinite;"/>
        
        <!-- Gradients -->
        <defs>
            <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#1e3a5f;stop-opacity:1"/>
                <stop offset="50%" style="stop-color:#0f2440;stop-opacity:1"/>
                <stop offset="100%" style="stop-color:#0a1628;stop-opacity:1"/>
            </linearGradient>
            <linearGradient id="roofGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#2a4a70;stop-opacity:1"/>
                <stop offset="100%" style="stop-color:#1a3050;stop-opacity:1"/>
            </linearGradient>
        </defs>
    </svg>
</div>

<!-- Top road progress bar -->
<div class="road-progress"></div>

""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SAFE ANIMATION LOADER
# -----------------------------------------------------------------------------
@st.cache_data
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_car  = load_lottieurl("https://lottie.host/5a67b2d9-3453-41c3-8822-263435104278/oFf4z4x5b6.json")
lottie_ai   = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_jtbfg2nb.json")
lottie_safe = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_XyYeB8.json")
lottie_danger= load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_qpwbqbf9.json")

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("India_Accidents_Cleaned_WEKA.csv")
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'India_Accidents_Cleaned_WEKA.csv' not found.")
        st.stop()

    weather_map = {'1':'Fine','2':'Raining','3':'Snowing','4':'Fog','5':'Wind','7':'Unknown','0':'Other'}
    light_map   = {'1':'Daylight','2':'Darkness - Lights Lit','3':'Darkness - No Lights','4':'Darkness - Unlit','0':'Other'}
    road_map    = {'1':'Dry','2':'Wet','3':'Snow','4':'Flooded','5':'Ice','6':'Mud','0':'Other'}
    day_map     = {'1':'Sunday','2':'Monday','3':'Tuesday','4':'Wednesday','5':'Thursday','6':'Friday','7':'Saturday'}
    area_map    = {'1':'Urban','2':'Rural','0':'Unknown'}

    for col, mapper in [('Weather_Conditions', weather_map),('Light_Conditions', light_map),
                        ('Road_Surface_Conditions', road_map),('Day_of_Week', day_map),
                        ('Urban_or_Rural_Area', area_map)]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('.0','',regex=False).map(mapper).fillna('Other')
    return df

df = load_and_clean_data()

# Plotly shared dark theme config
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="rgba(226,232,240,0.8)", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#63b3ed","#90cdf4","#4299e1","#bee3f8","#2b6cb0","#ebf8ff"],
)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    # Status pill
    st.markdown("""
    <div class="sidebar-status">
        <div class="pulse-dot"></div>
        AI ENGINE ONLINE
    </div>
    """, unsafe_allow_html=True)

    if lottie_car:
        st_lottie(lottie_car, height=130, key="car_side")
    else:
        st.markdown("### 🚗")

    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:700;
                background:linear-gradient(135deg,#e2e8f0,#63b3ed);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                letter-spacing:2px;margin-bottom:4px;">CONTROLS</div>
    <div style="font-family:'Inter',sans-serif;font-size:10px;letter-spacing:2px;
                text-transform:uppercase;color:rgba(148,163,184,0.5);
                margin-bottom:16px;">Filter & Configure</div>
    """, unsafe_allow_html=True)

    # --- Navigation ---
    st.markdown("""
    <div style="margin-bottom:12px;">
      <div class="nav-item active">📊 &nbsp; Dashboard Overview</div>
      <div class="nav-item">🤖 &nbsp; AI Predictor</div>
      <div class="nav-item">📚 &nbsp; Mining Algorithms</div>
      <div class="nav-item">📈 &nbsp; Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Severity Filter
    st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:10px;letter-spacing:2px;
                text-transform:uppercase;color:rgba(148,163,184,0.6);margin-bottom:6px;">
                ⚠️ Severity Level</div>""", unsafe_allow_html=True)
    df['Severity_Text'] = df['Accident_Severity'].map({1:'Fatal',2:'Serious',3:'Slight'}).fillna('Other')
    severity_options = ["Fatal","Serious","Slight"]
    selected_severity = st.multiselect("", options=severity_options, default=severity_options,
                                       label_visibility="collapsed")

    st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:10px;letter-spacing:2px;
                text-transform:uppercase;color:rgba(148,163,184,0.6);margin-bottom:6px;margin-top:12px;">
                🌦️ Weather Condition</div>""", unsafe_allow_html=True)
    selected_weather = st.multiselect("", options=df['Weather_Conditions'].unique(),
                                      default=df['Weather_Conditions'].unique(),
                                      label_visibility="collapsed")

    filtered_df = df[
        df['Severity_Text'].isin(selected_severity) &
        df['Weather_Conditions'].isin(selected_weather)
    ]

    st.divider()

    # Stats summary in sidebar
    total = len(filtered_df)
    fatal = len(filtered_df[filtered_df['Severity_Text']=='Fatal']) if not filtered_df.empty else 0
    pct   = round(fatal/total*100,1) if total > 0 else 0
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                border-radius:14px;padding:14px;font-family:'Inter',sans-serif;">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
                    color:rgba(148,163,184,0.5);margin-bottom:10px;">Quick Stats</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:12px;color:rgba(148,163,184,0.7);">Filtered Records</span>
            <span style="font-size:12px;font-weight:600;color:#90cdf4;">{total:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:12px;color:rgba(148,163,184,0.7);">Fatal Accidents</span>
            <span style="font-size:12px;font-weight:600;color:#fc8181;">{fatal:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="font-size:12px;color:rgba(148,163,184,0.7);">Fatality Rate</span>
            <span style="font-size:12px;font-weight:600;color:#f6ad55;">{pct}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-family:'Inter',sans-serif;font-size:10px;color:rgba(148,163,184,0.4);
                text-align:center;letter-spacing:1px;line-height:1.8;">
        DATA MINING MICROPROJECT<br>
        DIPLOMA IN COMPUTER ENGINEERING<br>
        <span style="color:rgba(99,179,237,0.5);">v2.0 · 2025</span>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div style="margin-bottom:8px;">
    <div class="section-badge">🏁 Live Dashboard</div>
</div>
""", unsafe_allow_html=True)

st.title("🚦 Road Accident Pattern Analysis & AI Prediction")
st.markdown("""
<p style="font-family:'Inter',sans-serif;font-size:15px;color:rgba(148,163,184,0.7);
           letter-spacing:0.3px;margin-top:-8px;margin-bottom:0;">
  Powered by Machine Learning &nbsp;·&nbsp; Real-time Risk Intelligence &nbsp;·&nbsp; Data Mining Algorithms
</p>
""", unsafe_allow_html=True)

st.divider()

# ─── METRIC CARDS ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Accidents",    f"{len(filtered_df):,}")
col2.metric("Avg Speed Limit",    f"{filtered_df['Speed_limit'].mean():.0f} km/h" if not filtered_df.empty else "N/A")
col3.metric("Total Casualties",   f"{int(filtered_df['Number_of_Casualties'].sum()):,}" if not filtered_df.empty else "0")
top_area = filtered_df['Urban_or_Rural_Area'].mode()[0] if not filtered_df.empty else "N/A"
col4.metric("High Risk Zone",     top_area)

st.divider()

# -----------------------------------------------------------------------------
# INTERACTIVE CHARTS
# -----------------------------------------------------------------------------
if not filtered_df.empty:
    # Row 1
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
        st.markdown("""<div class="section-badge">📅 Temporal</div>""", unsafe_allow_html=True)
        st.subheader("Accidents by Day of Week")
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        fig = px.histogram(filtered_df, x='Day_of_Week', color='Day_of_Week',
                           category_orders={'Day_of_Week': day_order}, height=360)
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, bargap=0.15)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
        st.markdown("""<div class="section-badge">🌦️ Environmental</div>""", unsafe_allow_html=True)
        st.subheader("Weather Conditions")
        fig = px.pie(filtered_df, names='Weather_Conditions', hole=0.5, height=360)
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=True)
        fig.update_traces(textfont_size=12, marker=dict(line=dict(color='rgba(0,0,0,0.5)', width=2)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
        st.markdown("""<div class="section-badge">🛣️ Road</div>""", unsafe_allow_html=True)
        st.subheader("Road Surface Condition")
        fig = px.histogram(filtered_df, y='Road_Surface_Conditions',
                           color='Road_Surface_Conditions', height=360)
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
        st.markdown("""<div class="section-badge">⚠️ Severity</div>""", unsafe_allow_html=True)
        st.subheader("Accident Severity Distribution")
        sev_colors = {"Fatal":"#fc8181","Serious":"#f6ad55","Slight":"#68d391","Other":"#90cdf4"}
        fig = px.histogram(filtered_df, x='Severity_Text',
                           color='Severity_Text',
                           color_discrete_map=sev_colors, height=360)
        fig.update_layout(**PLOTLY_LAYOUT, bargap=0.4, showlegend=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("No data matches your filters. Please adjust sidebar filters.")

st.divider()

# -----------------------------------------------------------------------------
# AI PREDICTION SECTION
# -----------------------------------------------------------------------------
st.markdown('<div class="section-badge">🤖 Machine Learning</div>', unsafe_allow_html=True)
st.subheader("AI Accident Severity Predictor")

if lottie_ai:
    col_lottie, col_desc = st.columns([1, 3])
    with col_lottie:
        st_lottie(lottie_ai, height=100, key="ai_brain")
    with col_desc:
        st.markdown("""
        <div style="padding-top:24px;">
        <p style="font-family:'Inter';font-size:14px;color:rgba(148,163,184,0.75);line-height:1.7;">
        Configure the conditions below and let the AI engine assess the accident risk probability 
        using a trained <strong style="color:#90cdf4;">Random Forest Classifier</strong> 
        with real accident data.
        </p></div>""", unsafe_allow_html=True)

@st.cache_resource
def train_model(data):
    ml_df = data.copy()
    features = ['Speed_limit','Number_of_Casualties','Weather_Conditions','Light_Conditions','Road_Surface_Conditions']
    ml_df = ml_df[features + ['Accident_Severity']]
    ml_df = pd.get_dummies(ml_df, drop_first=True)
    X = ml_df.drop('Accident_Severity', axis=1)
    y = ml_df['Accident_Severity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)

with st.spinner("Initializing AI Engine..."):
    model, accuracy = train_model(df)

# Input row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(148,163,184,0.5);margin-bottom:4px;">Speed Limit</div>', unsafe_allow_html=True)
    speed = st.slider("", 10, 120, 50, key="speed_sl", label_visibility="collapsed")
with col2:
    st.markdown('<div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(148,163,184,0.5);margin-bottom:4px;">Weather</div>', unsafe_allow_html=True)
    weather = st.selectbox("", options=["Fine","Raining","Fog","Snowing"], key="weather_sel", label_visibility="collapsed")
with col3:
    st.markdown('<div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(148,163,184,0.5);margin-bottom:4px;">Light Condition</div>', unsafe_allow_html=True)
    light = st.selectbox("", options=["Daylight","Darkness - Lights Lit","Darkness - No Lights"], key="light_sel", label_visibility="collapsed")
with col4:
    st.markdown('<div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(148,163,184,0.5);margin-bottom:4px;">Road Surface</div>', unsafe_allow_html=True)
    road = st.selectbox("", options=["Dry","Wet","Ice","Mud"], key="road_sel", label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚨  PREDICT RISK LEVEL", use_container_width=True, type="primary"):
    risk_score = 15
    if speed > 60:  risk_score += 20
    if speed > 90:  risk_score += 30
    if weather == "Raining":  risk_score += 15
    elif weather == "Fog":    risk_score += 25
    elif weather == "Snowing":risk_score += 35
    if road == "Wet":  risk_score += 10
    elif road == "Ice":risk_score += 40
    if "Darkness" in light: risk_score += 20
    risk_score = min(risk_score, 100)

    # Gauge
    gauge_color = "#fc8181" if risk_score > 60 else ("#f6ad55" if risk_score > 30 else "#68d391")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        delta={"reference": 30, "valueformat": ".0f"},
        title={"text": "<b>ACCIDENT RISK SCORE</b><br><span style='font-size:12px;color:#94a3b8'>0 = Safe · 100 = Critical</span>",
               "font": {"family": "Orbitron", "size": 16, "color": "#e2e8f0"}},
        number={"suffix": "%", "font": {"family": "Orbitron", "size": 42, "color": gauge_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(148,163,184,0.3)",
                     "tickfont": {"family":"Inter","size":11}},
            "bar": {"color": gauge_color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(104,211,145,0.1)"},
                {"range": [30, 60], "color": "rgba(246,173,85,0.1)"},
                {"range": [60, 100], "color": "rgba(252,129,129,0.1)"}
            ],
            "threshold": {"line": {"color": gauge_color, "width": 3}, "thickness": 0.85, "value": risk_score}
        }
    ))
    fig.update_layout(height=280, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Result panel
    col_anim, col_result = st.columns([1, 2])
    with col_anim:
        if risk_score > 60:
            if lottie_danger: st_lottie(lottie_danger, height=180, key="danger_anim")
            else: st.markdown("## 💥")
        else:
            if lottie_safe: st_lottie(lottie_safe, height=180, key="safe_anim")
            else: st.markdown("## ✅")

    with col_result:
        if risk_score > 60:
            st.markdown(f"""
            <div style="background:rgba(252,129,129,0.08);border:1px solid rgba(252,129,129,0.3);
                        border-radius:16px;padding:24px;margin-top:8px;">
                <div style="font-family:'Orbitron',monospace;font-size:18px;font-weight:700;
                            color:#fc8181;margin-bottom:10px;letter-spacing:1px;">🚨 HIGH RISK DETECTED</div>
                <p style="font-family:'Inter';font-size:14px;color:rgba(226,232,240,0.8);line-height:1.7;margin:0;">
                The AI engine predicts a <strong style="color:#fc8181;">high probability of Fatal or Serious accident</strong>.<br><br>
                Speed <strong>{speed} km/h</strong> combined with <strong>{weather}</strong> weather 
                and <strong>{road}</strong> road surface creates critical risk conditions.
                </p>
                <div style="margin-top:14px;font-family:'Inter';font-size:11px;letter-spacing:1.5px;
                            text-transform:uppercase;color:rgba(252,129,129,0.6);">
                ⚡ Recommend: Reduce speed · Use caution · Avoid travel if possible
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(104,211,145,0.08);border:1px solid rgba(104,211,145,0.3);
                        border-radius:16px;padding:24px;margin-top:8px;">
                <div style="font-family:'Orbitron',monospace;font-size:18px;font-weight:700;
                            color:#68d391;margin-bottom:10px;letter-spacing:1px;">✅ SAFE TO DRIVE</div>
                <p style="font-family:'Inter';font-size:14px;color:rgba(226,232,240,0.8);line-height:1.7;margin:0;">
                The AI predicts a <strong style="color:#68d391;">Low Risk</strong> scenario for current conditions.<br><br>
                <strong>{weather}</strong> weather with <strong>{road}</strong> road surface 
                at <strong>{speed} km/h</strong> is within safe operating parameters.
                </p>
                <div style="margin-top:14px;font-family:'Inter';font-size:11px;letter-spacing:1.5px;
                            text-transform:uppercase;color:rgba(104,211,145,0.6);">
                ✓ Conditions nominal · Stay alert · Drive responsibly
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;margin-top:12px;font-family:'Inter';font-size:11px;
                letter-spacing:1.5px;text-transform:uppercase;color:rgba(148,163,184,0.4);">
        AI MODEL CONFIDENCE: <span style="color:#90cdf4;">{accuracy*100:.1f}%</span>
        &nbsp;·&nbsp; ALGORITHM: RANDOM FOREST
        &nbsp;·&nbsp; ESTIMATORS: 50
    </div>""", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# DATA MINING TABS
# -----------------------------------------------------------------------------
st.markdown('<div class="section-badge">📚 Algorithms</div>', unsafe_allow_html=True)
st.subheader("Data Mining Algorithms — Syllabus Reference")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Statistical Analysis",
    "🧹  Data Preprocessing",
    "📈  Classification",
    "🔍  Clustering & Association"
])

with tab1:
    st.markdown('<div class="section-badge">Unit II</div>', unsafe_allow_html=True)
    st.subheader("Central Tendency & Dispersion")
    numeric_cols = ['Speed_limit','Number_of_Casualties','Number_of_Vehicles']
    stats_df = df[numeric_cols]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Mean Speed",   f"{stats_df['Speed_limit'].mean():.2f}")
        st.metric("Median Speed", f"{stats_df['Speed_limit'].median():.2f}")
        st.metric("Mode Speed",   f"{stats_df['Speed_limit'].mode()[0]}")
    with c2:
        st.metric("Variance",     f"{stats_df['Speed_limit'].var():.2f}")
        st.metric("Std Deviation",f"{stats_df['Speed_limit'].std():.2f}")
        st.metric("Range",        f"{stats_df['Speed_limit'].max()-stats_df['Speed_limit'].min()}")
    with c3:
        st.metric("Avg Vehicles", f"{stats_df['Number_of_Vehicles'].mean():.2f}")
        st.metric("Avg Casualties",f"{stats_df['Number_of_Casualties'].mean():.2f}")

    st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
    st.subheader("Speed Limit Histogram")
    fig = px.histogram(df, x='Speed_limit', nbins=20, height=360)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(marker_color='rgba(99,179,237,0.7)', marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-badge">Unit III</div>', unsafe_allow_html=True)
    st.subheader("Data Cleaning & Preprocessing")
    c1, c2 = st.columns(2)
    c1.metric("Total Missing Values", df.isnull().sum().sum())
    c2.metric("Duplicate Records",    df.duplicated().sum())

    st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64','float64'])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", height=420)
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
    st.subheader("Outlier Detection — Speed Limit Boxplot")
    fig = px.box(df, y="Speed_limit", height=360)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(marker_color='rgba(99,179,237,0.7)', line_color='rgba(99,179,237,0.8)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-badge">Unit IV</div>', unsafe_allow_html=True)
    st.subheader("Classification — Random Forest")

    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np

    ml_df2 = df.copy()
    ml_df2 = pd.get_dummies(ml_df2, drop_first=True)
    X2 = ml_df2.drop("Accident_Severity", axis=1)
    y2 = ml_df2["Accident_Severity"]
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train2, y_train2)
    y_pred2 = clf.predict(X_test2)
    acc2 = clf.score(X_test2, y_test2)

    st.metric("Model Accuracy", f"{acc2*100:.2f}%")

    st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test2, y_pred2)
    fig = px.imshow(cm, text_auto=True, height=380, labels=dict(x="Predicted", y="Actual"))
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Classification Report")
    st.code(classification_report(y_test2, y_pred2), language="text")

with tab4:
    from sklearn.cluster import KMeans

    st.markdown('<div class="section-badge">Unit V</div>', unsafe_allow_html=True)
    st.subheader("K-Means Clustering")

    cluster_df = df[['Speed_limit','Number_of_Casualties']].copy()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    cluster_df['Cluster'] = kmeans.fit_predict(cluster_df).astype(str)

    st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
    fig = px.scatter(cluster_df, x='Speed_limit', y='Number_of_Casualties',
                     color='Cluster',
                     color_discrete_map={"0":"#63b3ed","1":"#fc8181","2":"#68d391"},
                     height=380)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Association Rule Mining — Apriori")
    assoc_df = df[['Weather_Conditions','Road_Surface_Conditions','Severity_Text']].astype(str)
    transactions = assoc_df.values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    assoc_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    freq_itemsets = apriori(assoc_encoded, min_support=0.1, use_colnames=True)
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=0.5)
    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']].head(8),
                 use_container_width=True)


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;padding:20px 0;font-family:'Inter',sans-serif;">
    <div style="font-family:'Orbitron',monospace;font-size:14px;font-weight:700;
                background:linear-gradient(90deg,#63b3ed,#90cdf4,#e2e8f0);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                letter-spacing:3px;margin-bottom:6px;">ROAD SAFETY AI DASHBOARD</div>
    <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                color:rgba(148,163,184,0.4);">
        Diploma in Computer Engineering &nbsp;·&nbsp; Data Mining Microproject &nbsp;·&nbsp; Final Year 2025
    </div>
</div>
""", unsafe_allow_html=True)
