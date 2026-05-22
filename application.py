# =============================================================================
# Smartphone Price Prediction Dashboard
# -----------------------------------------------------------------------------
# Streamlit application for estimating smartphone market value using a
# machine learning pipeline trained on hardware specifications.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64
from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# Custom Feature Engineering Transformer
# -----------------------------------------------------------------------------
# Handles:
# - Missing value imputation
# - Derived feature generation
# - Removal of unused training columns
# =============================================================================

class Custom_Transform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Replace invalid zero values before computing column medians
        self.medians_ = X.replace(0, np.nan).median()
        return self

    def transform(self, X):
        x_new = X.copy()
        x_new = x_new.replace(0, np.nan).fillna(self.medians_)

        # Composite hardware performance metric
        x_new["performance_score"] = (
            x_new["cpu freq"] * x_new["cpu core"] * x_new["ram"]
        )
        x_new["display_quality"] = x_new["ppi"] * x_new["resoloution"]
        x_new["ram_mem"] = x_new["ram"] * x_new["internal mem"]

        remove_cols = ["Price", "Product_id", "Sale"]
        all_cols = list(x_new.columns)

        for col in remove_cols:
            if col in all_cols:
                x_new = x_new.drop(columns=[col])

        return x_new


# =============================================================================
# Streamlit Application Configuration
# =============================================================================

st.set_page_config(
    page_title="Cellphone Price Predictor", page_icon="📱", layout="centered"
)

# =============================================================================
# Custom UI Styling
# -----------------------------------------------------------------------------
# Contains premium UI styling, animations, typography, sliders, button
# interactions, and responsive visual enhancements.
# =============================================================================

st.markdown(
    """
    <style>
    
    @import url('https://api.fontshare.com/v2/css?f[]=clash-display@600,700,800&f[]=satoshi@400,500,700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&display=swap');

    
    html, body, [class*="css"], .stMarkdown, p, span, div, li {
        font-family: 'Satoshi', sans-serif !important;
    }

    
    h1, h2, h3, .gradient-text, .section-header {
        font-family: 'Clash Display', sans-serif !important;
    }
    
    button, .stButton > button,
    label, .stLabel,
    input, select, textarea,
    div[data-baseweb="slider"] *,
    div[data-baseweb="select"] * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: 0.03em !important; 
    }

    
    
    div[data-testid="stSlider"], div[data-testid="stSelectbox"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 8px 0px !important;
        margin-bottom: 24px !important;
    }
    
    
    div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label {
        color: #B0B0B0 !important; 
        font-size: 0.85rem !important;
        font-weight: 600 !important; 
        letter-spacing: 0.1em !important;
        text-transform: uppercase;
        margin-bottom: 12px !important;
        transition: color 0.4s ease !important;
    }
    div[data-testid="stSlider"]:hover label, div[data-testid="stSelectbox"]:hover label {
        color: #FFFFFF !important; 
    }

    
    div[data-baseweb="select"] > div {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0 4px !important;
        box-shadow: none !important; 
    }
    div[data-baseweb="select"] > div:hover, div[data-baseweb="select"] > div:focus-within {
        box-shadow: none !important;
        border: none !important;
    }
    div[data-baseweb="select"] * {
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        color: #CCCCCC !important; 
        transition: color 0.3s ease !important;
    }
    div[data-baseweb="select"]:hover * {
        color: #FFFFFF !important; 
    }
    
        
    div[data-baseweb="select"] > div > div:first-child,
    div[data-baseweb="select"] > div > div:first-child * {
        color: #f99157 !important;
        -webkit-text-fill-color: #f99157 !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        transition: color 0.3s ease !important;
    }

    
    div[data-baseweb="select"] > div {
        border-radius: 6px !important;
        transition: background 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-baseweb="select"] > div:hover {
        background: rgba(255, 255, 255, 0.055) !important;
    }

    
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div,
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div > div {
        height: 2px !important;
        border-radius: 2px !important;
        transition: width 0.4s cubic-bezier(0.16, 1, 0.3, 1), left 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        box-shadow: none !important;
    }

    
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div:nth-child(2),
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div > div:first-child {
        background: #4A90E2 !important;
        box-shadow: none !important;
    }
    
    
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div:nth-child(3),
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div > div:first-child {
        background: transparent !important;
        box-shadow: none !important;
    }

    
    div[data-baseweb="slider"] div[role="slider"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: left 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important; 
    }
    div[data-baseweb="slider"] div[role="slider"]::before { display: none !important; }

    div[data-baseweb="slider"] div[role="slider"]::after {
        content: "";
        position: absolute;
        width: 6px !important; 
        height: 6px !important;
        background: #f99157 !important; 
        border-radius: 50% !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.4) !important;
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease !important;
        border: none !important;
    }
    
    
    div[data-testid="stSlider"]:hover div[data-baseweb="slider"] div[role="slider"]::after {
        transform: scale(1.66) !important;
        box-shadow: 0 0 10px rgba(249, 145, 87, 0.5) !important;
    }
    
    
    div[data-baseweb="slider"] div[role="slider"]:active::after,
    div[data-baseweb="slider"] div[role="slider"]:focus::after {
        transform: scale(1.33) !important;
        box-shadow: 0 0 12px rgba(249, 145, 87, 0.8) !important;
    }

    
    div[data-testid="stThumbValue"],
    div[data-baseweb="slider"] div[role="slider"] div {
        color: #f99157 !important;
        font-weight: 700 !important; 
        font-size: 0.9rem !important; 
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        transform-origin: bottom center !important;
        -webkit-text-fill-color: #f99157 !important; 
    }
    
    div[data-testid="stThumbValue"] *,
    div[data-baseweb="slider"] div[role="slider"] div * {
        color: #f99157 !important;
        font-weight: 700 !important; 
        -webkit-text-fill-color: #f99157 !important;
    }
    
    
    div[data-testid="stSlider"]:hover div[data-testid="stThumbValue"],
    div[data-testid="stSlider"]:hover div[data-baseweb="slider"] div[role="slider"] div {
        transform: scale(1.05) translateY(-1px) !important; 
    }

    
    div[data-testid="stSliderTickBar"] {
        display: flex !important;
        justify-content: space-between !important;
        padding-top: 10px !important;
    }

    
    div[data-testid="stSliderTickBar"],
    div[data-baseweb="slider"] div[data-baseweb="mark"] {
        opacity: 0 !important; 
        transition: opacity 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    div[data-testid="stSlider"]:hover div[data-testid="stSliderTickBar"],
    div[data-testid="stSlider"]:hover div[data-baseweb="slider"] div[data-baseweb="mark"] {
        opacity: 1 !important; 
    }

    
    div[data-testid="stSliderTickBar"] svg,
    div[data-testid="stSliderTickBar"] path,
    div[data-testid="stSliderTickBar"] div:empty,
    div[data-baseweb="slider"] div[data-baseweb="mark"] > div:first-child,
    div[data-baseweb="slider"] div[data-baseweb="mark"] > div:empty {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        border: none !important;
        background-color: transparent !important;
    }

    
    div[data-testid="stSliderTickBar"]::before,
    div[data-testid="stSliderTickBar"]::after,
    div[data-testid="stSliderTickBar"] div::before,
    div[data-testid="stSliderTickBar"] div::after,
    div[data-baseweb="slider"] div[data-baseweb="mark"]::before,
    div[data-baseweb="slider"] div[data-baseweb="mark"]::after {
        content: none !important;
        display: none !important;
    }
    
    
    div[data-testid="stSliderTickBar"] *,
    div[data-baseweb="slider"] div[data-baseweb="mark"] * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.85rem !important; 
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        color: #CCCCCC !important; 
        background: transparent !important;
        text-shadow: none !important;
        border: none !important; 
    }

    
    .section-header {
        color: #A0A0A0;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 50px;
        margin-bottom: 25px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05); 
        background: none;
        box-shadow: none;
        display: block;
    }
    
    div[data-baseweb="slider"] {
        width: 100% !important;
        margin: 0 auto !important;
    }

    

.element-container:has(.stButton) {
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
    margin-top: 18px !important;
}

div.stButton {
    width: auto !important;
    display: flex !important;
    justify-content: center !important;
}

div.stButton > button {
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: auto !important;
    min-width: 0 !important;
    padding: 13px 30px !important;
    border-radius: 8px !important;
    background: rgba(255,255,255,0.045) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #D8D8D8 !important;
    backdrop-filter: blur(10px) !important;
    overflow: hidden !important;

    transition:
        transform 0.45s cubic-bezier(0.16, 1, 0.3, 1),
        background 0.45s ease,
        border-color 0.45s ease,
        box-shadow 0.45s ease !important;

    box-shadow:
        0 0 0 rgba(74,144,226,0),
        0 8px 30px rgba(0,0,0,0.18);
}

div.stButton > button p,
div.stButton > button span {
    margin: 0 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    transition: color 0.35s ease !important;
}

div.stButton > button::before {
    content: "";
    position: absolute;
    inset: -40%;
    background:
        radial-gradient(
            circle at center,
            rgba(74,144,226,0.18) 0%,
            rgba(74,144,226,0.08) 30%,
            transparent 70%
        );

    opacity: 0;

    transform: translateX(-30%) translateY(10%) scale(0.8);

    transition:
        opacity 0.6s ease,
        transform 0.8s cubic-bezier(0.16,1,0.3,1);

    pointer-events: none;
}

div.stButton > button:hover {
    transform:
        translateY(-2px)
        scale(1.015);

    background: rgba(255,255,255,0.065) !important;

    border-color: rgba(74,144,226,0.22) !important;

    box-shadow:
        0 0 40px rgba(74,144,226,0.10),
        0 12px 40px rgba(0,0,0,0.24);
}

div.stButton > button:hover::before {
    opacity: 1;

    transform:
        translateX(15%)
        translateY(-10%)
        scale(1.15);
}

div.stButton > button:hover p,
div.stButton > button:hover span {
    color: white !important;
}

div.stButton > button:active {
    transform:
        translateY(0px)
        scale(0.985);
}


    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    div[data-testid="stSlider"], div[data-testid="stSelectbox"] {
        animation: fadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    
    @keyframes textShine {
        0% { background-position: 0% center; }
        100% { background-position: 100% center; }
    }
    .gradient-text {
        background: linear-gradient(120deg, #4A90E2 30%, #E0F7FA 50%, #4A90E2 70%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textShine 4s cubic-bezier(0.4, 0, 0.2, 1) infinite alternate;
    }
    
    
    @keyframes bottomAnchoredSway {
        0% { transform: rotate(-2deg); }
        50% { transform: rotate(2deg); }
        100% { transform: rotate(-2deg); }
    }
    .img-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 45px;
        margin-bottom: 25px;
    }
    .premium-hero-img {
        width: 100%;
        max-width: 550px;      
        max-height: 325px;     
        object-fit: contain;   
        border-radius: 12px;
        transform-origin: bottom center; 
        animation: bottomAnchoredSway 6s ease-in-out infinite; 
        filter: drop-shadow(0 15px 25px rgba(0, 0, 0, 0.4)); 
    }
    
    .block-container {
        margin-top: -64px !important; 
    }

    
    div[data-testid="stSlider"], div[data-testid="stSelectbox"] {
        margin-bottom: 32px !important;
    }

    .section-header {
        margin-top: 62px;
    }

    
    body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.018;
        z-index: 0;
        background-image:
            radial-gradient(rgba(255,255,255,0.14) 0.5px, transparent 0.5px);
        background-size: 4px 4px;
    }

    
    .main .block-container {
        animation: pageReveal 900ms cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes pageReveal {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    
    .result-reveal {
        animation: resultReveal 650ms cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes resultReveal {
        from {
            opacity: 0;
            transform: translateY(14px);
            filter: blur(6px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
            filter: blur(0px);
        }
    }

    </style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Model Loading
# -----------------------------------------------------------------------------
# Loads the serialized machine learning pipeline and caches it to improve
# application startup and prediction performance.
# =============================================================================

@st.cache_resource
def load_model():
    with open("phone_price_pipeline.pkl", "rb") as f:
        return pickle.load(f)


try:
    pipeline = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file 'phone_price_pipeline.pkl' not found.")
    st.stop()


st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# Hero Section
# -----------------------------------------------------------------------------
# Main landing section introducing the application branding and visual
# identity.
# =============================================================================

st.markdown(
    """
    <h1 style='font-size: 60px; margin-bottom: 0; line-height: 1.1; text-align: center;'>
        Cellphone<br>
        <span class='gradient-text' style='font-size: 68px; font-weight: 800;'>Value Predictor</span>
    </h1>
    <p style='color: #888; font-size: 16px; margin-top: 20px; text-align: center; font-weight: 500;'>
        Precision smartphone valuation powered by intelligent hardware analysis.
    </p>
""",
    unsafe_allow_html=True,
)


st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)


image_path = "image.png"
if os.path.exists(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <div class="img-wrapper">
                <img class="premium-hero-img" src="data:image/png;base64,{encoded_string}" alt="App Hero Image">
            </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <hr style="
        width:76%;
        margin:-34px auto 26px auto;
        border:none;
        height:0.5px;
        background-color:rgba(255,255,255,0.10);
    ">
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='height: 34px;'></div>",
    unsafe_allow_html=True,
)


# =============================================================================
# User Input Section
# -----------------------------------------------------------------------------
# Collects smartphone hardware specifications used by the prediction model.
# =============================================================================

st.markdown(
    "<div class='section-header'>📐 Physical Dimensions</div>", unsafe_allow_html=True
)
col1, col2 = st.columns(2)
with col1:
    weight = st.slider(
        "Weight (grams)", min_value=100.0, max_value=250.0, value=175.0, step=1.0
    )
with col2:
    thickness = st.slider(
        "Thickness (mm)", min_value=5.0, max_value=12.0, value=7.5, step=0.1
    )

st.markdown("<div class='section-header'>🖥️ Display</div>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    resoloution = st.slider(
        "Screen Size (Inches)", min_value=4.0, max_value=7.0, value=5.8, step=0.1
    )
with col4:
    ppi = st.slider(
        "Pixels Per Inch (PPI)", min_value=150, max_value=600, value=350, step=5
    )

st.markdown(
    "<div class='section-header'>⚙️ Core Performance</div>", unsafe_allow_html=True
)
col5, col6 = st.columns(2)
with col5:
    cpu_core = st.selectbox("CPU Cores", options=[2, 4, 6, 8], index=3)
    ram = st.select_slider("RAM (GB)", options=[1.0, 2.0, 3.0, 4.0, 6.0], value=4.0)
with col6:
    cpu_freq = st.slider(
        "CPU Frequency (GHz)", min_value=1.0, max_value=3.0, value=2.0, step=0.1
    )
    internal_mem = st.select_slider(
        "Internal Memory (GB)", options=[8.0, 16.0, 32.0, 64.0, 128.0], value=64.0
    )

st.markdown(
    "<div class='section-header'>📸 Camera & Battery</div>", unsafe_allow_html=True
)
col7, col8 = st.columns(2)
with col7:
    rear_cam = st.slider(
        "Rear Camera (MP)", min_value=2.0, max_value=48.0, value=12.0, step=1.0
    )
    front_cam = st.slider(
        "Front Camera (MP)", min_value=0.0, max_value=32.0, value=8.0, step=1.0
    )
with col8:
    battery = st.slider(
        "Battery Capacity (mAh)", min_value=1500, max_value=6000, value=3500, step=50
    )


# =============================================================================
# Prediction Execution
# -----------------------------------------------------------------------------
# Builds an input dataframe from UI values and runs inference using the
# trained pipeline.
# =============================================================================

if st.button("Estimate Market Value"):
    input_data = pd.DataFrame(
        {
            "Product_id": [0],
            "Sale": [0],
            "weight": [weight],
            "resoloution": [resoloution],
            "ppi": [ppi],
            "cpu core": [cpu_core],
            "cpu freq": [cpu_freq],
            "internal mem": [internal_mem],
            "ram": [ram],
            "RearCam": [rear_cam],
            "Front_Cam": [front_cam],
            "battery": [battery],
            "thickness": [thickness],
        }
    )

    with st.spinner("Processing specifications through pipeline..."):
        try:
            # Generate price prediction using the trained ML pipeline
            prediction = pipeline.predict(input_data)
            predicted_price = max(50, prediction[0])

            st.markdown(
                """
                <div class="result-reveal" style="text-align:center; margin-top:38px; padding:42px 20px 12px;">
                    <p style="color:#777; font-size:11px; font-weight:700; letter-spacing:0.22em; text-transform:uppercase; margin:0 0 18px;">
                        Estimated Valuation
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<h1 style='text-align:center; font-size:68px; margin:0; font-family:Clash Display, sans-serif; line-height:1;' class='gradient-text'><span style='font-size:32px; vertical-align:super;'>$</span>{predicted_price:,.2f}</h1>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='width:46px; height:1px; background:rgba(74,144,226,0.22); margin:28px auto 0;'></div>",
                unsafe_allow_html=True,
            )

            st.components.v1.html(
                """
                <script>
                    const result = window.parent.document.querySelector('.result-reveal');

                    if (result) {
                        result.scrollIntoView({
                            behavior: 'smooth'
                        });
                    }
                </script>
                """,
                height=0,
            )

        except Exception as e:
            st.error(f"Computation Error: {e}")

st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)


# =============================================================================
# Frontend Interaction Enhancements
# -----------------------------------------------------------------------------
# JavaScript used for hover motion effects and smooth UI interactions.
# =============================================================================

st.components.v1.html(
    """
    <script>
    const doc = window.parent.document;

    function centerButton() {
        doc.querySelectorAll('.stButton').forEach(el => {
            el.style.setProperty('display', 'flex', 'important');
            el.style.setProperty('justify-content', 'center', 'important');
            let parent = el.parentElement;
            while (parent) {
                parent.style.setProperty('display', 'flex', 'important');
                parent.style.setProperty('justify-content', 'center', 'important');
                if (parent.classList.contains('block-container')) break;
                parent = parent.parentElement;
            }
        });
    }

    centerButton();
    new MutationObserver(centerButton).observe(doc.body, { childList: true, subtree: true });

    const buttons = doc.querySelectorAll('.stButton > button');

    buttons.forEach(btn => {

        btn.addEventListener('mousemove', e => {

            const rect = btn.getBoundingClientRect();

            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const moveX = (x - rect.width / 2) * 0.04;
            const moveY = (y - rect.height / 2) * 0.10;

            btn.style.transform = `
                translate(${moveX}px, ${moveY - 2}px)
                scale(1.015)
            `;
        });

        btn.addEventListener('mouseleave', () => {
            btn.style.transform = '';
        });
    });
    </script>
    """,
    height=0,
    width=0,
)
