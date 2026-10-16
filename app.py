
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "predictive_maintenance.joblib"

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="NEXUS AI | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# Futuristic glassmorphism styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #07111f;
        --panel: rgba(16, 31, 51, 0.78);
        --border: rgba(120, 190, 255, 0.16);
        --cyan: #4de2e8;
        --blue: #648cff;
        --muted: #93a9c4;
        --white: #edf5ff;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(40, 110, 190, 0.22), transparent 35%),
            radial-gradient(ellipse at 95% 15%, rgba(33, 206, 214, 0.12), transparent 30%),
            linear-gradient(145deg, #07111f 0%, #091626 50%, #07101d 100%);
        color: var(--white);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, 0.65);
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1450px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p, label {
        color: var(--white);
    }

    h1 {
        letter-spacing: -1.5px;
        font-weight: 800 !important;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
        padding: 12px 0 22px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 26px;
    }

    .brand {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 2px;
        color: #f4f8ff;
    }

    .brand span {
        color: var(--cyan);
    }

    .live-pill {
        display: inline-block;
        padding: 8px 13px;
        border: 1px solid rgba(77, 226, 232, 0.25);
        border-radius: 30px;
        background: rgba(77, 226, 232, 0.07);
        color: #7cf0e9;
        font-size: 11px;
        letter-spacing: 1px;
        font-weight: 700;
    }

    .eyebrow {
        color: var(--cyan);
        font-size: 11px;
        letter-spacing: 3px;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.8;
        max-width: 700px;
    }

    .glass-card {
        background: linear-gradient(
            135deg,
            rgba(24, 45, 69, 0.84),
            rgba(12, 25, 43, 0.82)
        );
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 14px 45px rgba(0, 0, 0, 0.16);
        backdrop-filter: blur(16px);
        margin-bottom: 14px;
    }

    .card-label {
        color: #9bb1cb;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .metric-value {
        font-size: clamp(24px, 3vw, 34px);
        font-weight: 800;
        letter-spacing: -1px;
        color: #f1f7ff;
    }

    .metric-note {
        color: #91a8c2;
        font-size: 12px;
        margin-top: 8px;
        line-height: 1.5;
    }

    .section-title {
        font-size: 19px;
        font-weight: 700;
        color: #f0f6ff;
        margin: 10px 0 4px;
    }

    .section-caption {
        color: #8ea6c2;
        font-size: 12px;
        margin-bottom: 18px;
    }

    .status-normal {
        color: #69f0c1;
        font-size: 25px;
        font-weight: 800;
    }

    .status-failure {
        color: #ff839b;
        font-size: 25px;
        font-weight: 800;
    }

    .risk-number {
        color: var(--cyan);
        font-size: 44px;
        font-weight: 800;
        letter-spacing: -2px;
    }

    .small-muted {
        color: #91a8c2;
        font-size: 12px;
        line-height: 1.7;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        width: 100%;
        min-height: 48px;
        border: 1px solid rgba(77, 226, 232, 0.4);
        border-radius: 11px;
        background: linear-gradient(100deg, #176c94, #2864bd);
        color: white;
        font-weight: 700;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        border-color: var(--cyan);
        background: linear-gradient(100deg, #1986a8, #3977d6);
        color: white;
        box-shadow: 0 0 22px rgba(77, 226, 232, 0.15);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: rgba(6, 17, 31, 0.8);
        border-color: rgba(120, 190, 255, 0.2);
        border-radius: 9px;
    }

    [data-testid="stNumberInput"] input {
        color: #f0f6ff;
    }

    [data-testid="stMetric"] {
        background: rgba(14, 30, 49, 0.72);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #9bb1cb;
    }

    [data-testid="stMetricValue"] {
        color: #f0f6ff;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4de2e8, #648cff);
    }

    hr {
        border-color: var(--border);
    }

    .footer {
        color: #7088a5;
        font-size: 11px;
        text-align: center;
        padding: 22px 0 5px;
        line-height: 1.8;
    }

    @media (max-width: 768px) {
        [data-testid="stAppViewContainer"] .main .block-container {
            padding: 1rem 1rem 2rem;
        }

        .glass-card {
            padding: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
st.markdown(
    """
    <div class="topbar">
        <div class="brand">NEXUS<span>AI</span>
            <div style="font-size:10px;color:#8ea6c2;
                letter-spacing:2px;font-weight:500;margin-top:5px;">
                INTELLIGENT MACHINE MONITORING
            </div>
        </div>
        <div class="live-pill">● &nbsp; PREDICTION SYSTEM READY</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">INDUSTRIAL INTELLIGENCE / 01</div>',
            unsafe_allow_html=True)
st.title("Predictive Maintenance")
st.markdown(
    """
    <div class="hero-subtitle">
        Analyze machine operating conditions and estimate failure risk
        using a trained machine-learning model. Enter sensor values below
        to generate a prediction.
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run `python -m src.train` "
        "from the project directory first."
    )
    st.stop()

try:
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
except Exception as error:
    st.error(f"Could not load the saved model: {error}")
    st.stop()

# --------------------------------------------------
# Overview cards
# --------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

overview1, overview2, overview3 = st.columns(3)

with overview1:
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-label">AI ENGINE</div>
            <div class="metric-value">Random Forest</div>
            <div class="metric-note">Trained classification model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview2:
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-label">INPUT CHANNELS</div>
            <div class="metric-value">06</div>
            <div class="metric-note">Machine operating features</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview3:
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-label">INFERENCE MODE</div>
            <div class="metric-value">On demand</div>
            <div class="metric-note">Prediction from entered values</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Inputs and results
# --------------------------------------------------
input_col, result_col = st.columns([1.05, 0.95], gap="large")

with input_col:
    st.markdown('<div class="section-title">Machine Parameters</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Configure operating conditions</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        with st.form("machine_parameters"):
            machine_type = st.selectbox(
                "Product / machine type",
                options=["L", "M", "H"],
                index=1,
                help="Type categories used by the dataset: L, M, and H.",
            )

            st.markdown("**Thermal conditions**")
            air_temperature = st.number_input(
                "Air temperature (K)",
                min_value=250.0,
                max_value=350.0,
                value=300.0,
                step=0.1,
            )
            process_temperature = st.number_input(
                "Process temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=310.0,
                step=0.1,
            )

            st.markdown("**Mechanical conditions**")
            rotational_speed = st.number_input(
                "Rotational speed (rpm)",
                min_value=0,
                max_value=10000,
                value=1500,
                step=10,
            )
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=200.0,
                value=40.0,
                step=0.5,
            )
            tool_wear = st.number_input(
                "Tool wear (minutes)",
                min_value=0.0,
                max_value=500.0,
                value=100.0,
                step=1.0,
            )

            submitted = st.form_submit_button(
                "Run AI Prediction  →",
                type="primary",
                use_container_width=True,
            )

with result_col:
    st.markdown('<div class="section-title">Machine Health Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Prediction results and estimated risk</div>',
        unsafe_allow_html=True,
    )

    if submitted:
        input_data = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }])

        if process_temperature <= air_temperature:
            st.warning(
                "Process temperature is not higher than air temperature. "
                "Confirm that these input values are intentional."
            )

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])

        if prediction == 1:
            status_class = "status-failure"
            status_text = "Failure Predicted"
            status_icon = "⚠"
            status_description = (
                "The model classified this operating scenario as a "
                "potential machine failure."
            )
        else:
            status_class = "status-normal"
            status_text = "Normal Predicted"
            status_icon = "✓"
            status_description = (
                "The model classified this operating scenario as normal."
            )

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-label">PREDICTED MACHINE STATUS</div>
                <div class="{status_class}">
                    {status_icon} &nbsp; {status_text}
                </div>
                <div class="metric-note">{status_description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-label">ESTIMATED FAILURE PROBABILITY</div>
                <div class="risk-number">{probability:.2%}</div>
                <div class="metric-note">
                    Model-estimated probability for the supplied inputs
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(probability, text="Estimated failure probability")

        risk_label = (
            "Higher model-estimated probability"
            if probability >= 0.5
            else "Lower model-estimated probability"
        )
        st.caption(risk_label)

        with st.expander("View submitted operating conditions"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)

    else:
        st.markdown(
            """
            <div class="glass-card" style="min-height:190px;
                display:flex;flex-direction:column;justify-content:center;">
                <div class="card-label">AWAITING INPUT</div>
                <div style="font-size:24px;font-weight:800;color:#dceaff;">
                    Ready to analyze
                </div>
                <div class="metric-note">
                    Enter machine conditions and select
                    <b>Run AI Prediction</b> to see the results here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass-card">
                <div class="card-label">ANALYSIS PIPELINE</div>
                <div class="small-muted">
                    01 &nbsp; Sensor values entered<br><br>
                    02 &nbsp; Model evaluates input features<br><br>
                    03 &nbsp; Status and probability displayed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------
# Information and footer
# --------------------------------------------------
st.divider()

with st.expander("About this project and model"):
    st.write(
        "This educational project uses the UCI AI4I 2020 Predictive "
        "Maintenance Dataset. The saved Random Forest model predicts "
        "the Machine failure target from six operating-condition features."
    )
    st.write(
        "The benchmark dataset is not live sensor data from physical "
        "industrial equipment. Model probabilities are estimates and "
        "must not be treated as guarantees or used alone for safety-critical "
        "maintenance decisions."
    )

st.markdown(
    """
    <div class="footer">
        NEXUS AI · PREDICTIVE MAINTENANCE<br>
        EDUCATIONAL MACHINE-LEARNING DEMONSTRATION
        &nbsp; | &nbsp; NOT A CERTIFIED INDUSTRIAL SAFETY SYSTEM
    </div>
    """,
    unsafe_allow_html=True,
)
