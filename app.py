import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="ExoVision",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)


def local_css(file_name):

    with open(file_name) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

local_css("css/style.css")

with st.sidebar:

    st.sidebar.markdown("""
        # 🚀 Mission Control

        ---

        """)

    selected = option_menu(
        menu_title="ExoVision",
        options=[
            "Home",
            "Dataset",
            "EDA",
            "Predict",
            "Performance",
            "Feature Importance",
            "About Kepler"
        ],
        icons=[
            "house",
            "database",
            "bar-chart",
            "rocket",
            "graph-up",
            "stars",
            "globe"
        ],
        default_index=0,
    )

    st.sidebar.markdown("---")

    st.sidebar.info("""
        Mission Status

        🟢 Online

        Kepler Dataset Loaded

        Machine Learning Ready
    """)

if selected=="Home":

    st.markdown("""
        <div class="hero">

        <div class="planet">🌌</div>

        <h1>ExoVision</h1>

        <p>

        Discover planets beyond our Solar System using
        Machine Learning and NASA's Kepler Mission.

        </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class='subtitle'>
        AI Powered Kepler Exoplanet Discovery Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )


c1,c2,c3,c4 = st.columns(4)

with c1:

    st.markdown(
        """
        <div class='metric-card'>

        <h2>🌍</h2>

        <h1>9564</h1>

        <p>Observations</p>

        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class='metric-card'>

        <h2>🚀</h2>

        <h1>92.64%</h1>

        <p>Accuracy</p>

        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        """
        <div class='metric-card'>

        <h2>⭐</h2>

        <h1>97.29%</h1>

        <p>ROC AUC</p>

        </div>
        """,
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        """
        <div class='metric-card'>

        <h2>🤖</h2>

        <h1>5</h1>

        <p>Models</p>

        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


st.markdown("""

    <div class="section-title">

    🛰 Dataset Overview

    </div>

    """, unsafe_allow_html=True)



st.markdown("""
    <div class="section-title">

    🚀 Mission Brief

    </div>
    """, unsafe_allow_html=True)

st.markdown("""

    <div class="glass">

    ExoVision is an AI-powered dashboard designed for exploring NASA's Kepler Exoplanet Search dataset.

    The application combines exploratory data analysis, interactive visualizations, and machine learning prediction into a unified experience inspired by a futuristic mission control center.

    </div>

    """, unsafe_allow_html=True)


st.markdown("---")

st.markdown("""

<div style="text-align:center;color:#8da8d6">

Made with ❤️ using

Python • Streamlit • Plotly • Scikit-Learn

</div>

""", unsafe_allow_html=True)