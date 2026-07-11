import streamlit as st
import pandas as pd

from utils.data_loader import load_data, load_model
from utils.predictor import prepare_input

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="Predict Exoplanet",
    page_icon="🤖",
    layout="wide",
)

# -------------------------------------------------------
# Load CSS
# -------------------------------------------------------
with open("css/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )

# -------------------------------------------------------
# Load Resources
# -------------------------------------------------------
df = load_data()
model = load_model()

LABEL_MAP = {0: "False Positive", 1: "Confirmed Planet"}

# -------------------------------------------------------
# Hero
# -------------------------------------------------------
st.markdown("""
<div class="hero">
<div class="planet">🤖</div>
<h1>AI Exoplanet Prediction</h1>
<p>
Predict whether a Kepler Object of Interest is a
Confirmed Planet or a False Positive.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Machine Learning Prediction Engine
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Input Form
# -------------------------------------------------------
st.markdown("""
<div class="section-title">
📥 Enter Planet Parameters
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

features = [
    "koi_period",
    "koi_time0bk",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]

inputs = {}

for i, feature in enumerate(features):

    minimum = float(df[feature].min())
    maximum = float(df[feature].max())
    default = float(df[feature].median())

    target = col1 if i % 2 == 0 else col2

    with target:
        inputs[feature] = st.number_input(
            feature.replace("_", " ").title(),
            min_value=minimum,
            max_value=maximum,
            value=default,
        )

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------
st.markdown("---")

if st.button("🚀 Predict Exoplanet", use_container_width=True):

    input_df = prepare_input(inputs)

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    confidence = probabilities.max() * 100

    st.markdown("""
    <div class="section-title">
    🌍 Prediction Result
    </div>
    """, unsafe_allow_html=True)

    if prediction == 1:
        st.success("🪐 CONFIRMED PLANET")
    else:
        st.error("❌ FALSE POSITIVE")

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(confidence / 100)

    prob_df = pd.DataFrame({
        "Class": [LABEL_MAP.get(c, str(c)) for c in model.classes_],
        "Probability": probabilities
    })

    st.subheader("Class Probabilities")

    st.bar_chart(
        prob_df.set_index("Class")
    )

    st.subheader("Input Summary")

    st.dataframe(
        input_df,
        use_container_width=True
    )