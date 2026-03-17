import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
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
    initial_sidebar_state="expanded"          # ← Force starts expanded
)

# -----------------------------------------------------------------------------
# FIXED + COOLER CSS (Sidebar always visible + arrow + no overlap)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* ── GOOGLE FONTS ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

/* ── HIDE STREAMLIT CHROME ── */
.stDeployButton { display:none !important; }
footer { visibility:hidden !important; }
#MainMenu { visibility:hidden !important; }

/* ── COOL SIDEBAR - FIXED WIDTH + ALWAYS VISIBLE + COLLAPSE ARROW ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 2px solid #3b82f6 !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.6) !important;
    width: 340px !important;
    min-width: 340px !important;
    max-width: 340px !important;
    z-index: 1000 !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* SHOW COLLAPSE ARROW (cool glowing arrow) */
button[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    background: rgba(59,130,246,0.25) !important;
    border: 1px solid #3b82f6 !important;
    color: #90cdf4 !important;
    border-radius: 50% !important;
    width: 42px !important;
    height: 42px !important;
    box-shadow: 0 0 15px rgba(59,130,246,0.5) !important;
    transition: all 0.3s ease !important;
}
button[data-testid="stSidebarCollapseButton"]:hover {
    transform: scale(1.1) !important;
    box-shadow: 0 0 25px rgba(59,130,246,0.8) !important;
}

/* PREVENT ANY OVERLAP AT TOP + SHIFT MAIN CONTENT */
[data-testid="stMain"] {
    margin-left: 340px !important;
    padding-top: 1.5rem !important;
    transition: margin-left 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* ── APP BACKGROUND (slightly cooler gradient) ── */
.stApp { 
    background: linear-gradient(135deg,#060b14 0%,#0a0f1e 35%,#060d1a 70%,#0a0c18 100%) !important;
    background-attachment: fixed !important; 
}

/* ── MAIN CONTAINER (no overlap) ── */
.block-container { 
    padding-top:1.5rem !important; 
    padding-left: 2rem !important; 
    padding-right:2rem !important; 
    max-width:1400px !important; 
}

/* ── TITLE (extra glow for cooler look) ── */
h1 { 
    font-family:'Orbitron',monospace !important; 
    font-weight:900 !important; 
    font-size:2.5rem !important; 
    background: linear-gradient(270deg, #63b3ed, #90cdf4, #e2e8f0, #90cdf4, #63b3ed);
    background-size: 300% 300%; 
    -webkit-background-clip:text !important; 
    background-clip:text !important; 
    -webkit-text-fill-color:transparent !important; 
    letter-spacing:3px !important; 
    animation: gradientShift 5s ease infinite, fadeInUp 1s ease-out, titleGlow 2.5s ease-in-out infinite !important; 
    margin-bottom: 0.3rem !important; 
}

@keyframes titleGlow { 0%,100% { text-shadow: 0 0 25px rgba(99,179,237,0.4); } 50% { text-shadow: 0 0 45px rgba(99,179,237,0.8), 0 0 70px rgba(99,179,237,0.4); } }
@keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes fadeInUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }

.subtitle { 
    font-family:'Inter',sans-serif !important; 
    color:rgba(148,163,184,0.8) !important; 
    font-size:14px !important; 
    letter-spacing:0.5px !important; 
    animation: fadeInUp 1s ease-out 0.5s both; 
}

/* Rest of your original styles (unchanged - only sidebar + main fixed) */
[data-testid="metric-container"] { 
    background:rgba(255,255,255,0.03) !important; 
    backdrop-filter:blur(20px) !important; 
    border:1px solid rgba(255,255,255,0.08) !important; 
    border-radius:16px !important; 
    padding:20px !important; 
    box-shadow:0 4px 24px rgba(0,0,0,0.3) !important; 
    transition:all 0.3s ease !important; 
}
[data-testid="metric-container"]:hover { 
    border-color:rgba(99,179,237,0.4) !important; 
    transform:translateY(-4px) !important; 
    box-shadow:0 12px 40px rgba(0,0,0,0.5) !important; 
}
[data-testid="stMetricValue"] { 
    font-family:'Orbitron',monospace !important; 
    font-size:1.9rem !important; 
    font-weight:700 !important; 
    color:#90cdf4 !important; 
}
[data-testid="stMetricLabel"] { 
    font-family:'Inter',sans-serif !important; 
    font-size:0.75rem !important; 
    letter-spacing:1.5px !important; 
    text-transform:uppercase !important; 
    color:rgba(148,163,184,0.7) !important; 
}

/* Sidebar internal (unchanged) */
[data-testid="stSidebar"] .block-container { 
    padding:1.5rem 1.2rem !important; 
}
.sb-status { 
    display:flex; align-items:center; gap:8px; 
    background:rgba(59, 130, 246, 0.2); 
    border:1px solid #3b82f6; 
    border-radius:50px; 
    padding:8px 14px; 
    margin-bottom:16px; 
    font-family:'Inter',sans-serif; 
    font-size:10px; 
    letter-spacing:1.5px; 
    text-transform:uppercase; 
    color:#90cdf4; 
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.3); 
}
.pdot { 
    width:7px; height:7px; 
    background:#3b82f6; 
    border-radius:50%; 
    box-shadow:0 0 8px #3b82f6; 
    animation:pdot 1.5s ease-in-out infinite; 
}
@keyframes pdot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(1.3)} }

.sidebar-title { 
    font-family:'Orbitron',monospace !important; 
    font-size:1.1rem !important; 
    font-weight:700 !important; 
    color:#90cdf4 !important; 
    letter-spacing:2px !important; 
    margin-bottom:4px !important; 
    margin-top:10px !important; 
}
.sidebar-subtitle { 
    font-family:'Inter',sans-serif !important; 
    font-size:9px !important; 
    letter-spacing:2px !important; 
    text-transform:uppercase !important; 
    color:rgba(148,163,184,0.6) !important; 
    margin-bottom:16px !important; 
}

.nav-item { 
    display:flex; align-items:center; gap:10px; 
    padding:10px 14px; 
    border-radius:12px; 
    margin-bottom:6px; 
    font-family:'Inter',sans-serif; 
    font-size:13px; 
    font-weight:500; 
    color:rgba(226,232,240,0.7); 
    cursor:pointer; 
    transition:all 0.2s ease; 
    border:1px solid transparent; 
}
.nav-item:hover { 
    background:rgba(255,255,255,0.08); 
    border-color:rgba(255,255,255,0.15); 
    color:#e2e8f0; 
    transform: translateX(3px); 
}
.nav-item.active { 
    background:rgba(59, 130, 246, 0.2); 
    border-color:#3b82f6; 
    color:#90cdf4; 
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.2); 
}

.filter-label { 
    font-family:'Inter',sans-serif !important; 
    font-size:9px !important; 
    letter-spacing:2px !important; 
    text-transform:uppercase !important; 
    color:rgba(148,163,184,0.7) !important; 
    margin-bottom:6px !important; 
    margin-top:12px !important; 
}

.quick-stats { 
    background:rgba(255,255,255,0.05); 
    border:1px solid rgba(255,255,255,0.1); 
    border-radius:14px; 
    padding:15px; 
    font-family:'Inter',sans-serif; 
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05); 
    margin-top:15px; 
}

.sidebar-footer { 
    font-family:'Inter',sans-serif !important; 
    font-size:9px !important; 
    color:rgba(148,163,184,0.5) !important; 
    text-align:center !important; 
    letter-spacing:1px !important; 
    line-height:1.8 !important; 
    margin-top:20px !important; 
}

.stButton>button { 
    background:linear-gradient(135deg,rgba(59, 130, 246, 0.3),rgba(37, 99, 235, 0.4)) !important; 
    border:1px solid #3b82f6 !important; 
    border-radius:14px !important; 
    color:#e0f2fe !important; 
    font-family:'Inter',sans-serif !important; 
    font-weight:600 !important; 
    font-size:14px !important; 
    letter-spacing:1px !important; 
    text-transform:uppercase !important; 
    padding:14px 28px !important; 
    transition:all 0.3s ease !important; 
    box-shadow:0 4px 20px rgba(59, 130, 246, 0.3) !important; 
}
.stButton>button:hover { 
    background:linear-gradient(135deg,rgba(59, 130, 246, 0.4),rgba(37, 99, 235, 0.5)) !important; 
    border-color:#60a5fa !important; 
    box-shadow:0 8px 30px rgba(59, 130, 246, 0.5) !important; 
    transform:translateY(-2px) !important; 
}

.stSelectbox>div>div, .stMultiSelect>div { 
    background:rgba(255,255,255,0.08) !important; 
    border:1px solid rgba(255,255,255,0.15) !important; 
    border-radius:12px !important; 
    color:rgba(226,232,240,0.9) !important; 
}

.stTabs [data-baseweb="tab-list"] { 
    background:rgba(255,255,255,0.03) !important; 
    border-radius:15px !important; 
    padding:6px !important; 
    gap:4px !important; 
    border:1px solid rgba(255,255,255,0.08) !important; 
}
.stTabs [data-baseweb="tab"] { 
    background:transparent !important; 
    border-radius:11px !important; 
    color:rgba(148,163,184,0.7) !important; 
    font-family:'Inter',sans-serif !important; 
    font-weight:500 !important; 
    font-size:13px !important; 
    padding:9px 18px !important; 
    transition:all 0.2s ease !important; 
    border:none !important; 
}
.stTabs [aria-selected="true"] { 
    background:rgba(59, 130, 246, 0.2) !important; 
    color:#90cdf4 !important; 
    font-weight:600 !important; 
}

.cglass { 
    background:rgba(255,255,255,0.02); 
    backdrop-filter:blur(20px); 
    border:1px solid rgba(255,255,255,0.06); 
    border-radius:20px; 
    padding:22px; 
    box-shadow:0 8px 32px rgba(0,0,0,0.3); 
    transition:all 0.3s ease; 
    margin-bottom:15px; 
}
.cglass:hover { 
    border-color:rgba(59, 130, 246, 0.3); 
    box-shadow:0 12px 48px rgba(0,0,0,0.4); 
}

.sbadge { 
    display:inline-flex; align-items:center; gap:6px; 
    background:rgba(59, 130, 246, 0.15); 
    border:1px solid #3b82f6; 
    border-radius:50px; 
    padding:4px 12px; 
    font-family:'Inter',sans-serif; 
    font-size:9px; 
    letter-spacing:2px; 
    text-transform:uppercase; 
    color:#90cdf4; 
    margin-bottom:8px; 
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.2); 
}

hr { 
    border:none !important; 
    height:1px !important; 
    background:linear-gradient(90deg,transparent,rgba(59, 130, 246, 0.4),rgba(255,255,255,0.05),rgba(59, 130, 246, 0.4),transparent) !important; 
    margin:2rem 0 !important; 
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPERS (unchanged)
# -----------------------------------------------------------------------------
@st.cache_data
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_car    = load_lottieurl("https://lottie.host/5a67b2d9-3453-41c3-8822-263435104278/oFf4z4x5b6.json")
lottie_ai     = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_jtbfg2nb.json")
lottie_safe   = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_XyYeB8.json")
lottie_danger = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_qpwbqbf9.json")

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="rgba(226,232,240,0.8)", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#63b3ed","#90cdf4","#4299e1","#bee3f8","#2b6cb0","#ebf8ff"],
)

# -----------------------------------------------------------------------------
# DATA (unchanged)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("India_Accidents_Cleaned_WEKA.csv")
    except FileNotFoundError:
        st.error("'India_Accidents_Cleaned_WEKA.csv' not found. Place it in the same folder.")
        st.stop()

    weather_map = {'1':'Fine','2':'Raining','3':'Snowing','4':'Fog','5':'Wind','7':'Unknown','0':'Other'}
    light_map   = {'1':'Daylight','2':'Darkness - Lights Lit','3':'Darkness - No Lights','4':'Darkness - Unlit','0':'Other'}
    road_map    = {'1':'Dry','2':'Wet','3':'Snow','4':'Flooded','5':'Ice','6':'Mud','0':'Other'}
    day_map     = {'1':'Sunday','2':'Monday','3':'Tuesday','4':'Wednesday','5':'Thursday','6':'Friday','7':'Saturday'}
    area_map    = {'1':'Urban','2':'Rural','0':'Unknown'}

    for col, mp in [('Weather_Conditions',weather_map),('Light_Conditions',light_map),
                    ('Road_Surface_Conditions',road_map),('Day_of_Week',day_map),
                    ('Urban_or_Rural_Area',area_map)]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('.0','',regex=False).map(mp).fillna('Other')
    return df

df = load_data()

# -----------------------------------------------------------------------------
# SIDEBAR (NOW VISIBLE + ARROW + NO OVERLAP)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sb-status"><div class="pdot"></div>AI ENGINE ONLINE</div>', unsafe_allow_html=True)

    if lottie_car:
        st_lottie(lottie_car, height=100, key="sb_car")

    st.markdown('<div class="sidebar-title">CONTROLS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Filter & Configure</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:15px;">
      <div class="nav-item active">📊 &nbsp; Dashboard Overview</div>
      <div class="nav-item">🤖 &nbsp; AI Predictor</div>
      <div class="nav-item">📚 &nbsp; Mining Algorithms</div>
      <div class="nav-item">📈 &nbsp; Analytics</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="filter-label">⚠️ Severity Level</div>', unsafe_allow_html=True)
    df['Severity_Text'] = df['Accident_Severity'].map({1:'Fatal',2:'Serious',3:'Slight'}).fillna('Other')
    selected_severity = st.multiselect("", ["Fatal","Serious","Slight"], default=["Fatal","Serious","Slight"], label_visibility="collapsed")

    st.markdown('<div class="filter-label">🌦️ Weather Condition</div>', unsafe_allow_html=True)
    selected_weather = st.multiselect("", options=df['Weather_Conditions'].unique(),
                                      default=df['Weather_Conditions'].unique(), label_visibility="collapsed")

    filtered_df = df[df['Severity_Text'].isin(selected_severity) & df['Weather_Conditions'].isin(selected_weather)]

    st.divider()
    
    total = len(filtered_df)
    fatal = len(filtered_df[filtered_df['Severity_Text']=='Fatal']) if not filtered_df.empty else 0
    pct   = round(fatal/total*100,1) if total > 0 else 0

    st.markdown(f"""
    <div class="quick-stats">
        <div style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(148,163,184,0.6);margin-bottom:10px;">Quick Stats</div>
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
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sidebar-footer">DATA MINING MICROPROJECT<br>DIPLOMA IN COMPUTER ENGINEERING<br><span style="color:rgba(59, 130, 246, 0.6);">v2.0 · 2025</span></div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN DASHBOARD (rest unchanged - only sidebar fix applied)
# -----------------------------------------------------------------------------
st.markdown('<div class="sbadge">🏁 Live Dashboard</div>', unsafe_allow_html=True)
st.title("🚦 Road Accident Pattern Analysis & AI Prediction")
st.markdown('<p class="subtitle">Powered by Machine Learning &nbsp;·&nbsp; Real-time Risk Intelligence &nbsp;·&nbsp; Data Mining Algorithms</p>', unsafe_allow_html=True)

st.divider()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Accidents",  f"{len(filtered_df):,}")
c2.metric("Avg Speed Limit",  f"{filtered_df['Speed_limit'].mean():.0f} km/h" if not filtered_df.empty else "N/A")
c3.metric("Total Casualties", f"{int(filtered_df['Number_of_Casualties'].sum()):,}" if not filtered_df.empty else "0")
top_area = filtered_df['Urban_or_Rural_Area'].mode()[0] if not filtered_df.empty else "N/A"
c4.metric("High Risk Zone",   top_area)

st.divider()

if not filtered_df.empty:
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="cglass">', unsafe_allow_html=True)
        st.markdown('<div class="sbadge">📅 Temporal</div>', unsafe_allow_html=True)
        st.subheader("Accidents by Day of Week")
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        fig = px.histogram(filtered_df, x='Day_of_Week', color='Day_of_Week', category_orders={'Day_of_Week':day_order}, height=340)
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, bargap=0.15)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cb:
        st.markdown('<div class="cglass">', unsafe_allow_html=True)
        st.markdown('<div class="sbadge">🌦️ Environmental</div>', unsafe_allow_html=True)
        st.subheader("Weather Conditions")
        fig = px.pie(filtered_df, names='Weather_Conditions', hole=0.5, height=340)
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(textfont_size=12, marker=dict(line=dict(color='rgba(0,0,0,0.5)',width=2)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    cc, cd = st.columns(2)
    with cc:
        st.markdown('<div class="cglass">', unsafe_allow_html=True)
        st.markdown('<div class="sbadge">🛣️ Road</div>', unsafe_allow_html=True)
        st.subheader("Road Surface Condition")
        fig = px.histogram(filtered_df, y='Road_Surface_Conditions', color='Road_Surface_Conditions', height=340)
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cd:
        st.markdown('<div class="cglass">', unsafe_allow_html=True)
        st.markdown('<div class="sbadge">⚠️ Severity</div>', unsafe_allow_html=True)
        st.subheader("Accident Severity Distribution")
        sev_colors = {"Fatal":"#fc8181","Serious":"#f6ad55","Slight":"#68d391","Other":"#90cdf4"}
        fig = px.histogram(filtered_df, x='Severity_Text', color='Severity_Text', color_discrete_map=sev_colors, height=340)
        fig.update_layout(**PLOTLY_LAYOUT, bargap=0.4, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("No data matches your filters. Adjust the sidebar filters.")

st.divider()

# AI Predictor + rest of the app (completely unchanged)
st.markdown('<div class="sbadge">🤖 Machine Learning</div>', unsafe_allow_html=True)
st.subheader("AI Accident Severity Predictor")

if lottie_ai:
    lc, ld = st.columns([1,3])
    with lc: st_lottie(lottie_ai, height=90, key="ai_brain")
    with ld:
        st.markdown("""<div style="padding-top:20px;">
        <p style="font-family:'Inter';font-size:14px;color:rgba(148,163,184,0.75);line-height:1.7;">
        Configure conditions below — the AI engine assesses accident risk probability using a trained
        <strong style="color:#90cdf4;">Random Forest Classifier</strong> on real accident data.</p></div>""",
        unsafe_allow_html=True)

@st.cache_resource
def train_model(data):
    ml = data.copy()
    feats = ['Speed_limit','Number_of_Casualties','Weather_Conditions','Light_Conditions','Road_Surface_Conditions']
    ml = ml[feats + ['Accident_Severity']]
    ml = pd.get_dummies(ml, drop_first=True)
    X = ml.drop('Accident_Severity', axis=1)
    y = ml['Accident_Severity']
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(n_estimators=50)
    m.fit(Xtr, ytr)
    return m, m.score(Xte, yte)

with st.spinner("Initializing AI Engine..."):
    model, accuracy = train_model(df)

p1,p2,p3,p4 = st.columns(4)
with p1:
    st.markdown('<div class="filter-label">Speed Limit (km/h)</div>', unsafe_allow_html=True)
    speed = st.slider("", 10, 120, 50, key="spd", label_visibility="collapsed")
with p2:
    st.markdown('<div class="filter-label">Weather</div>', unsafe_allow_html=True)
    weather = st.selectbox("", ["Fine","Raining","Fog","Snowing"], key="wthr", label_visibility="collapsed")
with p3:
    st.markdown('<div class="filter-label">Light Condition</div>', unsafe_allow_html=True)
    light = st.selectbox("", ["Daylight","Darkness - Lights Lit","Darkness - No Lights"], key="lght", label_visibility="collapsed")
with p4:
    st.markdown('<div class="filter-label">Road Surface</div>', unsafe_allow_html=True)
    road = st.selectbox("", ["Dry","Wet","Ice","Mud"], key="rd", label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚨  PREDICT RISK LEVEL", use_container_width=True, type="primary"):
    risk = 15
    if speed > 60:  risk += 20
    if speed > 90:  risk += 30
    if weather == "Raining":  risk += 15
    elif weather == "Fog":    risk += 25
    elif weather == "Snowing":risk += 35
    if road == "Wet":  risk += 10
    elif road == "Ice":risk += 40
    if "Darkness" in light: risk += 20
    risk = min(risk, 100)

    gc = "#fc8181" if risk > 60 else ("#f6ad55" if risk > 30 else "#68d391")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk,
        delta={"reference":30,"valueformat":".0f"},
        title={"text":"<b>ACCIDENT RISK SCORE</b><br><span style='font-size:11px;color:#94a3b8'>0 = Safe · 100 = Critical</span>",
               "font":{"family":"Orbitron","size":15,"color":"#e2e8f0"}},
        number={"suffix":"%","font":{"family":"Orbitron","size":40,"color":gc}},
        gauge={"axis":{"range":[0,100],"tickwidth":1,"tickcolor":"rgba(148,163,184,0.3)"},
               "bar":{"color":gc,"thickness":0.25},
               "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
               "steps":[{"range":[0,30],"color":"rgba(104,211,145,0.1)"},
                         {"range":[30,60],"color":"rgba(246,173,85,0.1)"},
                         {"range":[60,100],"color":"rgba(252,129,129,0.1)"}],
               "threshold":{"line":{"color":gc,"width":3},"thickness":0.85,"value":risk}}
    ))
    fig.update_layout(height=270, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    ra, rb = st.columns([1,2])
    with ra:
        if risk > 60:
            if lottie_danger: st_lottie(lottie_danger, height=170, key="dng")
            else: st.markdown("## 💥")
        else:
            if lottie_safe: st_lottie(lottie_safe, height=170, key="sfe")
            else: st.markdown("## ✅")
    with rb:
        if risk > 60:
            st.markdown(f"""
            <div style="background:rgba(252,129,129,0.08);border:1px solid rgba(252,129,129,0.3);
                        border-radius:16px;padding:22px;margin-top:6px;
                        box-shadow: 0 0 30px rgba(252,129,129,0.1);">
                <div style="font-family:'Orbitron',monospace;font-size:17px;font-weight:700;
                            color:#fc8181;margin-bottom:9px;letter-spacing:1px;">🚨 HIGH RISK DETECTED</div>
                <p style="font-family:'Inter';font-size:13px;color:rgba(226,232,240,0.8);line-height:1.7;margin:0;">
                High probability of <strong style="color:#fc8181;">Fatal or Serious accident</strong>.
                Speed <strong>{speed} km/h</strong> + <strong>{weather}</strong> weather
                + <strong>{road}</strong> surface = critical conditions.</p>
                <div style="margin-top:12px;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(252,129,129,0.6);">
                ⚡ Reduce speed · Use caution · Avoid travel if possible</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(104,211,145,0.08);border:1px solid rgba(104,211,145,0.3);
                        border-radius:16px;padding:22px;margin-top:6px;
                        box-shadow: 0 0 30px rgba(104,211,145,0.1);">
                <div style="font-family:'Orbitron',monospace;font-size:17px;font-weight:700;
                            color:#68d391;margin-bottom:9px;letter-spacing:1px;">✅ SAFE TO DRIVE</div>
                <p style="font-family:'Inter';font-size:13px;color:rgba(226,232,240,0.8);line-height:1.7;margin:0;">
                <strong style="color:#68d391;">Low Risk</strong> scenario detected.
                <strong>{weather}</strong> weather + <strong>{road}</strong> surface
                at <strong>{speed} km/h</strong> — within safe parameters.</p>
                <div style="margin-top:12px;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(104,211,145,0.6);">
                ✓ Conditions nominal · Stay alert · Drive responsibly</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="text-align:center;margin-top:10px;font-family:'Inter';font-size:10px;
    letter-spacing:1.5px;text-transform:uppercase;color:rgba(148,163,184,0.4);">
    AI CONFIDENCE: <span style="color:#90cdf4;">{accuracy*100:.1f}%</span>
    &nbsp;·&nbsp; ALGORITHM: RANDOM FOREST &nbsp;·&nbsp; ESTIMATORS: 50</div>""",
    unsafe_allow_html=True)

st.divider()

# Tabs section (unchanged)
st.markdown('<div class="sbadge">📚 Algorithms</div>', unsafe_allow_html=True)
st.subheader("Data Mining Algorithms — Syllabus Reference")

tab1,tab2,tab3,tab4 = st.tabs(["📊  Statistical Analysis","🧹  Data Preprocessing","📈  Classification","🔍  Clustering & Association"])

with tab1:
    st.markdown('<div class="sbadge">Unit II</div>', unsafe_allow_html=True)
    st.subheader("Central Tendency & Dispersion")
    s = df[['Speed_limit','Number_of_Casualties','Number_of_Vehicles']]
    x1,x2,x3 = st.columns(3)
    with x1:
        st.metric("Mean Speed",   f"{s['Speed_limit'].mean():.2f}")
        st.metric("Median Speed", f"{s['Speed_limit'].median():.2f}")
        st.metric("Mode Speed",   f"{s['Speed_limit'].mode()[0]}")
    with x2:
        st.metric("Variance",      f"{s['Speed_limit'].var():.2f}")
        st.metric("Std Deviation", f"{s['Speed_limit'].std():.2f}")
        st.metric("Range",         f"{s['Speed_limit'].max()-s['Speed_limit'].min()}")
    with x3:
        st.metric("Avg Vehicles",  f"{s['Number_of_Vehicles'].mean():.2f}")
        st.metric("Avg Casualties",f"{s['Number_of_Casualties'].mean():.2f}")
    st.markdown('<div class="cglass">', unsafe_allow_html=True)
    st.subheader("Speed Limit Histogram")
    fig = px.histogram(df, x='Speed_limit', nbins=20, height=340)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(marker_color='rgba(99,179,237,0.7)', marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="sbadge">Unit III</div>', unsafe_allow_html=True)
    st.subheader("Data Cleaning & Preprocessing")
    d1,d2 = st.columns(2)
    d1.metric("Total Missing Values", df.isnull().sum().sum())
    d2.metric("Duplicate Records",    df.duplicated().sum())
    st.markdown('<div class="cglass">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=['int64','float64']).corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", height=400)
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="cglass">', unsafe_allow_html=True)
    st.subheader("Outlier Detection — Boxplot")
    fig = px.box(df, y="Speed_limit", height=340)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(marker_color='rgba(99,179,237,0.7)', line_color='rgba(99,179,237,0.8)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    from sklearn.metrics import confusion_matrix, classification_report
    st.markdown('<div class="sbadge">Unit IV</div>', unsafe_allow_html=True)
    st.subheader("Classification — Random Forest")
    ml2 = pd.get_dummies(df.copy(), drop_first=True)
    X2,y2 = ml2.drop("Accident_Severity",axis=1), ml2["Accident_Severity"]
    Xtr2,Xte2,ytr2,yte2 = train_test_split(X2,y2,test_size=0.2,random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(Xtr2,ytr2)
    yp2 = clf.predict(Xte2)
    st.metric("Model Accuracy", f"{clf.score(Xte2,yte2)*100:.2f}%")
    st.markdown('<div class="cglass">', unsafe_allow_html=True)
    st.subheader("Confusion Matrix")
    fig = px.imshow(confusion_matrix(yte2,yp2), text_auto=True, height=370,
                    labels=dict(x="Predicted",y="Actual"))
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.subheader("Classification Report")
    st.code(classification_report(yte2,yp2), language="text")

with tab4:
    from sklearn.cluster import KMeans
    st.markdown('<div class="sbadge">Unit V</div>', unsafe_allow_html=True)
    st.subheader("K-Means Clustering")
    cd2 = df[['Speed_limit','Number_of_Casualties']].copy()
    cd2['Cluster'] = KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(cd2).astype(str)
    st.markdown('<div class="cglass">', unsafe_allow_html=True)
    fig = px.scatter(cd2, x='Speed_limit', y='Number_of_Casualties', color='Cluster',
                     color_discrete_map={"0":"#63b3ed","1":"#fc8181","2":"#68d391"}, height=360)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Association Rule Mining — Apriori")
    at = df[['Weather_Conditions','Road_Surface_Conditions','Severity_Text']].astype(str)
    te = TransactionEncoder()
    ae = pd.DataFrame(te.fit(at.values.tolist()).transform(at.values.tolist()), columns=te.columns_)
    freq = apriori(ae, min_support=0.1, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.5)
    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']].head(8),
                 use_container_width=True)

st.divider()
st.markdown("""
<div style="text-align:center;padding:20px 0;font-family:'Inter',sans-serif;">
    <div style="font-family:'Orbitron',monospace;font-size:13px;font-weight:700;
                background:linear-gradient(90deg,#63b3ed,#90cdf4,#e2e8f0);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                letter-spacing:3px;margin-bottom:6px;">ROAD SAFETY AI DASHBOARD</div>
    <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                color:rgba(148,163,184,0.4);">
        Diploma in Computer Engineering &nbsp;·&nbsp; Data Mining Microproject &nbsp;·&nbsp; 2025
    </div>
</div>""", unsafe_allow_html=True)
