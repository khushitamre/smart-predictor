import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Predictive AI Dashboard", page_icon="📈", layout="wide")

# --- 2. MODERN MINIMALIST LASER THEME ---
modern_laser_css = """
<style>
.main {
    background-color: #0b0c10 !important;
    color: #c5c6c7 !important;
    font-family: 'Segoe UI', sans-serif;
}
.modern-title {
    font-size: 38px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    text-align: center;
    letter-spacing: 2px;
    margin-top: 10px;
}
.laser-line {
    height: 3px;
    width: 250px;
    background: linear-gradient(90deg, transparent, #00f3ff, transparent);
    margin: 0 auto 40px auto;
    box-shadow: 0 0 12px #00f3ff;
}
.laser-card {
    background: #1f2833 !important;
    border: 1px solid rgba(0, 243, 255, 0.4) !important;
    box-shadow: 0 0 15px rgba(0, 243, 255, 0.15) !important;
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 25px;
}
.stNumberInput input, .stSelectbox div, .stSlider div {
    background-color: #0b0c10 !important;
    color: #ffffff !important;
}
div.stButton > button:first-child {
    background: #0b0c10 !important;
    color: #00f3ff !important;
    border: 2px solid #00f3ff !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    width: 100%;
    height: 55px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 243, 255, 0.2) !important;
    transition: all 0.3s ease-in-out;
}
div.stButton > button:first-child:hover {
    background: #00f3ff !important;
    color: #0b0c10 !important;
    box-shadow: 0 0 25px #00f3ff !important;
}
</style>
"""
st.markdown(modern_laser_css, unsafe_allow_html=True)

# --- 3. LOADING ---
@st.cache_resource
def load_resources():
    model = joblib.load("logistic.pkl")
    df = pd.read_csv("customer_churn_prediction_dataset.csv")
    return model, df

try:
    model, df = load_resources()
    trained_features = list(model.feature_names_in_)
except Exception as e:
    st.error(f"File Load Error: {e}")
    st.stop()

st.markdown('<div class="modern-title">CHURN PREDICTION SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="laser-line"></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="laser-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#00f3ff; margin-top:0;'>📊 Customer Metrics</h3>", unsafe_allow_html=True)
    age = st.slider("Customer Age", min_value=18, max_value=100, value=48)
    salary = st.number_input("Annual Salary ($)", min_value=1000.0, value=42000.0, step=5000.0)
    gender_input = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Contract Tenure (Years)", 0.0, 10.0, 1.5, step=0.5)
    gender_value = 1 if gender_input == "Male" else 0
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="laser-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#00f3ff; margin-top:0;'>🔮 AI Diagnostic Radar</h3>", unsafe_allow_html=True)
    
    if st.button("RUN PREDICTIVE ANALYSIS"):
        try:
            input_dict = {}
            for feature in trained_features:
                feature_lower = feature.lower()
                if feature_lower == 'age': input_dict[feature] = float(age)
                elif feature_lower == 'gender': input_dict[feature] = float(gender_value)
                elif feature_lower == 'tenure': input_dict[feature] = float(tenure)
                elif feature_lower == 'salary': input_dict[feature] = float(salary)
                elif feature_lower == 'monthlycharges': input_dict[feature] = float(salary / 12)
                elif feature_lower == 'totalcharges': input_dict[feature] = float((salary / 12) * tenure)
                else:
                    if feature in df.columns:
                        if df[feature].dtype == 'object': input_dict[feature] = 0.0
                        else: input_dict[feature] = float(df[feature].median())
                    else: input_dict[feature] = 1.0

            input_df = pd.DataFrame([input_dict])[trained_features]
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] * 100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                title={'text': "RISK PROBABILITY FACTOR", 'font': {'color': '#c5c6c7', 'size': 15}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#ffffff"},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'bgcolor': "#0b0c10",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(0, 243, 255, 0.1)'},
                        {'range': [40, 70], 'color': 'rgba(255, 170, 0, 0.1)'},
                        {'range': [70, 100], 'color': 'rgba(255, 0, 85, 0.1)'}
                    ],
                    'threshold': {'line': {'color': "#ff007f", 'width': 5}, 'value': probability}
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#fff"}, height=250, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            if prediction == 1 or probability > 50:
                st.markdown(f"<div style='border: 1px solid #ff0055; padding: 15px;'><h4 style='color:#ff0055; text-align:center; margin:0;'>⚠️ SYSTEM ALERT: HIGH CHURN RISK DETECTED</h4></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='border: 1px solid #00ffcc; padding: 15px;'><h4 style='color:#00ffcc; text-align:center; margin:0;'>✅ STATUS STABLE: CUSTOMER RETENTION PROBABLE</h4></div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Analysis Interrupted: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
