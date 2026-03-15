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
    page_title="Road Safety AI Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# PROFESSIONAL UI STYLING (FIXED)
# -----------------------------------------------------------------------------
hide_st_style = """
            <style>
            /* We REMOVED '#MainMenu {visibility: hidden;}' so you can see the 3 dots now */
            
            /* Hide footer (Made with Streamlit) */
            footer {visibility: hidden;}
            
            /* Hide the deploy button */
            .stDeployButton {display:none;}
            
            /* Ensure sidebar toggle is always visible */
            [data-testid="collapsedControl"] {
                visibility: visible !important;
                z-index: 999999 !important;
            }
            
            .block-container {padding-top: 1rem;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

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

# Animations
lottie_car = load_lottieurl("https://lottie.host/5a67b2d9-3453-41c3-8822-263435104278/oFf4z4x5b6.json") # Blue Car
if lottie_car is None:
    lottie_car = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")

lottie_ai = load_lottieurl("https://lottie.host/0209705d-6c2e-4027-bd92-805166415714/2db45672-8703-4638-99d7-832145327293.json") # AI Brain

# NEW: Danger/Safe Animations
lottie_danger = load_lottieurl("https://lottie.host/956f1084-5f4c-4700-b63c-352055666708/p4v123456.json") # Placeholder for danger
if lottie_danger is None:
    lottie_danger = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_qpwbqbf9.json") # Crash/Alert

lottie_safe_drive = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_XyYeB8.json") # Smooth driving

# -----------------------------------------------------------------------------
# LOAD AND CLEAN DATA
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("India_Accidents_Cleaned_WEKA.csv")
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'India_Accidents_Cleaned_WEKA.csv' not found. Please check file name.")
        st.stop()

    # MAPPINGS (Convert codes to text)
    weather_map = {'1':'Fine', '2':'Raining', '3':'Snowing', '4':'Fog', '5':'Wind', '7':'Unknown', '0':'Other'}
    light_map = {'1':'Daylight', '2':'Darkness - Lights Lit', '3':'Darkness - No Lights', '4':'Darkness - Unlit', '0':'Other'}
    road_map = {'1':'Dry', '2':'Wet', '3':'Snow', '4':'Flooded', '5':'Ice', '6':'Mud', '0':'Other'}
    day_map = {'1':'Sunday', '2':'Monday', '3':'Tuesday', '4':'Wednesday', '5':'Thursday', '6':'Friday', '7':'Saturday'}
    area_map = {'1':'Urban', '2':'Rural', '0':'Unknown'}

    # Clean columns safely
    for col, mapper in [('Weather_Conditions', weather_map), ('Light_Conditions', light_map), 
                        ('Road_Surface_Conditions', road_map), ('Day_of_Week', day_map), 
                        ('Urban_or_Rural_Area', area_map)]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('.0','', regex=False).map(mapper).fillna('Other')

    return df

df = load_and_clean_data()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    if lottie_car:
        st_lottie(lottie_car, height=150, key="car")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/3097/3097180.png", width=100)
        
    st.title("🔧 Controls")

    st.subheader("Filter Dataset")
    
    # Severity Filter
    severity_options = ["Fatal", "Serious", "Slight"]
    df['Severity_Text'] = df['Accident_Severity'].map({1:'Fatal', 2:'Serious', 3:'Slight'}).fillna('Other')
    
    selected_severity = st.multiselect(
        "Accident Severity:",
        options=severity_options,
        default=severity_options
    )

    selected_weather = st.multiselect(
        "Weather Condition:",
        options=df['Weather_Conditions'].unique(),
        default=df['Weather_Conditions'].unique()
    )

    # Apply Filters
    filtered_df = df[
        df['Severity_Text'].isin(selected_severity) &
        df['Weather_Conditions'].isin(selected_weather)
    ]

    st.divider()
    st.caption("Data Mining Microproject")
    st.caption("Diploma in Computer Engineering")

# -----------------------------------------------------------------------------
# MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.title("🚦 Road Accident Pattern Analysis & AI Prediction")
st.subheader("Data Mining Microproject")

st.divider()

# ANIMATED METRIC CARDS
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Total Accidents", value=f"{len(filtered_df):,}")
col2.metric(label="Average Speed", value=f"{filtered_df['Speed_limit'].mean():.0f} km/h")
col3.metric(label="Total Casualties", value=f"{int(filtered_df['Number_of_Casualties'].sum()):,}")
top_area = filtered_df['Urban_or_Rural_Area'].mode()[0] if not filtered_df.empty else "N/A"
col4.metric(label="High Risk Location", value=top_area)
style_metric_cards(background_color="#1E1E1E", border_left_color="#FF4B4B")

st.divider()

# -----------------------------------------------------------------------------
# INTERACTIVE CHARTS
# -----------------------------------------------------------------------------
if not filtered_df.empty:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 Accidents by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig = px.histogram(filtered_df, x='Day_of_Week', color='Day_of_Week',
                        category_orders={'Day_of_Week': day_order},
                        template="plotly_dark", height=400)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("🌦️ Weather Conditions")
        fig = px.pie(filtered_df, names='Weather_Conditions', hole=0.4,
                    template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("🛣️ Road Surface Condition")
        fig = px.histogram(filtered_df, y='Road_Surface_Conditions', color='Road_Surface_Conditions',
                        template="plotly_dark", height=400)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("⚠️ Severity Distribution")
        fig = px.histogram(filtered_df, x='Severity_Text', color='Severity_Text',
                        template="plotly_dark", height=400)
        fig.update_layout(bargap=0.5, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data matches your filters. Please adjust sidebar filters.")

st.divider()

# -----------------------------------------------------------------------------
# AI PREDICTION SECTION
# -----------------------------------------------------------------------------
st.subheader("🤖 AI Accident Severity Predictor")

if lottie_ai:
    st_lottie(lottie_ai, height=120, key="ai")

# Train ML Model (Cached to run only once)
@st.cache_resource
def train_model(data):
    ml_df = data.copy()
    features = ['Speed_limit', 'Number_of_Casualties', 'Weather_Conditions', 'Light_Conditions', 'Road_Surface_Conditions']
    ml_df = ml_df[features + ['Accident_Severity']]
    ml_df = pd.get_dummies(ml_df, drop_first=True)
    X = ml_df.drop('Accident_Severity', axis=1)
    y = ml_df['Accident_Severity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc

with st.spinner("Initializing AI Engine..."):
    model, accuracy = train_model(df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    speed = st.slider("Speed Limit (km/h)", 10, 120, 50)
with col2:
    weather = st.selectbox("Weather", options=["Fine", "Raining", "Fog", "Snowing"])
with col3:
    light = st.selectbox("Light Condition", options=["Daylight", "Darkness - Lights Lit", "Darkness - No Lights"])
with col4:
    road = st.selectbox("Road Surface", options=["Dry", "Wet", "Ice", "Mud"])

if st.button("🚨 Predict Risk Level", use_container_width=True, type="primary"):

    # SIMULATED LOGIC FOR DEMO
    risk_score = 15
    if speed > 60: risk_score += 20
    if speed > 90: risk_score += 30
    if weather == "Raining": risk_score += 15
    elif weather == "Fog": risk_score += 25
    elif weather == "Snowing": risk_score += 35
    if road == "Wet": risk_score += 10
    elif road == "Ice": risk_score += 40
    if "Darkness" in light: risk_score += 20
    risk_score = min(risk_score, 100)

    # Animated Gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Accident Probability Risk"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if risk_score > 60 else "orange"},
            'steps' : [{'range': [0, 30], 'color': "green"}, {'range': [30, 60], 'color': "yellow"}, {'range': [60, 100], 'color': "red"}]}
    ))
    fig.update_layout(height=250, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # NEW: ANIMATION BASED ON RISK (Safe vs Danger)
    # ---------------------------------------------------------
    st.markdown("### 🎬 Live Scenario Simulation")
    
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        if risk_score > 60:
            if lottie_danger: 
                st_lottie(lottie_danger, height=200, key="danger")
            else: 
                st.error("💥 CRASH DETECTED!")
        else:
            if lottie_safe_drive:
                st_lottie(lottie_safe_drive, height=200, key="safe")
            else:
                st.success("🚗 SAFE DRIVING")

    with col_sim2:
        if risk_score > 60:
            st.error(f"🚨 **HIGH RISK DETECTED!**\n\nThe AI predicts a high probability of a **Fatal or Serious Accident**. Conditions like High Speed ({speed} km/h) combined with {weather} weather are dangerous.")
        else:
            st.success(f"✅ **SAFE TO DRIVE**\n\nThe AI predicts a **Low Risk** scenario. Conditions appear safe for travel.")
            
    st.caption(f"AI Confidence Score: {accuracy*100:.1f}%")

st.divider()

# -----------------------------------------------------------------------------
# 📚 DATA MINING ALGORITHMS SECTION (SYLLABUS BASED)
# -----------------------------------------------------------------------------

st.header("📚 Data Mining Algorithms (As Per Syllabus)")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Statistical Analysis",
    "🧹 Data Preprocessing",
    "📈 Classification",
    "🔍 Clustering & Association"
])

# =============================================================================
# 📊 TAB 1 – STATISTICAL ANALYSIS (UNIT II)
# =============================================================================
with tab1:
    st.subheader("Central Tendency & Dispersion Measures")

    numeric_cols = ['Speed_limit', 'Number_of_Casualties', 'Number_of_Vehicles']
    stats_df = df[numeric_cols]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean Speed", f"{stats_df['Speed_limit'].mean():.2f}")
        st.metric("Median Speed", f"{stats_df['Speed_limit'].median():.2f}")
        st.metric("Mode Speed", f"{stats_df['Speed_limit'].mode()[0]}")

    with col2:
        st.metric("Variance (Speed)", f"{stats_df['Speed_limit'].var():.2f}")
        st.metric("Std Deviation", f"{stats_df['Speed_limit'].std():.2f}")
        st.metric("Range", f"{stats_df['Speed_limit'].max() - stats_df['Speed_limit'].min()}")

    with col3:
        st.metric("Total Vehicles (Mean)", f"{stats_df['Number_of_Vehicles'].mean():.2f}")
        st.metric("Total Casualties (Mean)", f"{stats_df['Number_of_Casualties'].mean():.2f}")

    st.subheader("Histogram (Parametric Data Reduction)")
    fig = px.histogram(df, x='Speed_limit', nbins=20, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 🧹 TAB 2 – DATA PREPROCESSING (UNIT III)
# =============================================================================
with tab2:
    st.subheader("Data Cleaning & Preprocessing")

    col1, col2 = st.columns(2)
    col1.metric("Total Missing Values", df.isnull().sum().sum())
    col2.metric("Duplicate Records", df.duplicated().sum())

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()

    fig = px.imshow(corr, text_auto=True, aspect="auto", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Outlier Detection (Boxplot)")
    fig = px.box(df, y="Speed_limit", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 📈 TAB 3 – CLASSIFICATION (UNIT IV)
# =============================================================================
with tab3:
    st.subheader("Classification using Random Forest")

    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np

    # Prepare data
    ml_df = df.copy()
    ml_df = pd.get_dummies(ml_df, drop_first=True)

    X = ml_df.drop("Accident_Severity", axis=1)
    y = ml_df["Accident_Severity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = clf.score(X_test, y_test)

    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig = px.imshow(cm, text_auto=True, template="plotly_dark",
                    labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))


# =============================================================================
# 🔍 TAB 4 – CLUSTERING & ASSOCIATION
# =============================================================================
with tab4:
    st.subheader("K-Means Clustering")

    from sklearn.cluster import KMeans

    cluster_df = df[['Speed_limit', 'Number_of_Casualties']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_df['Cluster'] = kmeans.fit_predict(cluster_df)

    fig = px.scatter(cluster_df, x='Speed_limit', y='Number_of_Casualties',
                     color='Cluster', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Association Rule Mining (Apriori)")

    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    # Convert categorical columns
    assoc_df = df[['Weather_Conditions', 'Road_Surface_Conditions', 'Severity_Text']].astype(str)

    transactions = assoc_df.values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    assoc_df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(assoc_df_encoded, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    st.dataframe(rules.head(5))

st.markdown("---")
st.markdown("<center>Developed for Diploma in Computer Engineering | Final Year Microproject</center>", unsafe_allow_html=True)