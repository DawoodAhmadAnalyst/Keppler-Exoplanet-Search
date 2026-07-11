import streamlit as st
from utils.data_loader import load_data

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="ExoVision",
    page_icon="🌌",
    layout="wide",
)

# --------------------------------------------------
# Load CSS
# --------------------------------------------------
with open("css/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = load_data()

confirmed = (df["koi_disposition"] == "CONFIRMED").sum()
candidate = (df["koi_disposition"] == "CANDIDATE").sum()
false_positive = (df["koi_disposition"] == "FALSE POSITIVE").sum()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.markdown("# 🚀 Mission Control")

    st.markdown("---")

    st.success("🟢 Systems Online")

    st.info(f"📊 {len(df):,} Records")

    st.info(f"🧬 {df.shape[1]} Features")

    st.info("🤖 Machine Learning Ready")

    st.markdown("---")

    st.caption("Navigate using the Pages menu above.")

# --------------------------------------------------
# Hero
# --------------------------------------------------
st.markdown("""
<div class="hero">

<div class="planet">🪐</div>

<h1>ExoVision</h1>

<p>

Discover planets beyond our Solar System using
Artificial Intelligence and NASA's Kepler Mission.

</p>

</div>
""", unsafe_allow_html=True)

st.markdown(
"""
<div class="subtitle">

AI Powered Exoplanet Discovery Dashboard

</div>
""",
unsafe_allow_html=True
)

# --------------------------------------------------
# Metric Cards
# --------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

cards = [
    ("🌍", len(df), "Observations"),
    ("🛰", confirmed, "Confirmed"),
    ("⭐", candidate, "Candidates"),
    ("❌", false_positive, "False Positive"),
]

for col, (icon, value, label) in zip((c1, c2, c3, c4), cards):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">

            <h2>{icon}</h2>

            <h1>{value:,}</h1>

            <p>{label}</p>

            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# --------------------------------------------------
# Mission Brief
# --------------------------------------------------

st.markdown(
"""
<div class="section-title">

🛰 Mission Brief

</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="glass">

ExoVision is a professional machine learning dashboard built using Streamlit,
Plotly and Scikit-Learn.

The dashboard enables users to explore NASA's Kepler Exoplanet dataset,
perform exploratory data analysis, predict exoplanets using a trained machine
learning model, evaluate model performance and understand feature importance.

</div>
""",
unsafe_allow_html=True
)

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;color:#9fc9ff;">

Made with ❤️ using
Python • Streamlit • Plotly • Scikit-Learn

</div>
""",
unsafe_allow_html=True
)