import streamlit as st

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="About",
    page_icon="🌍",
    layout="wide"
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
# Hero
# --------------------------------------------------
st.markdown("""
<div class="hero">
<div class="planet">🌍</div>
<h1>About ExoVision</h1>
<p>
An AI-powered dashboard for exploring NASA's Kepler mission data
and classifying Kepler Objects of Interest as Confirmed Exoplanets
or False Positives.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Mission, Data & Team
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NASA Kepler Mission
# --------------------------------------------------
st.markdown("""
<div class="section-title">
🛰 The Kepler Mission
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass">
Launched by NASA in 2009, the Kepler Space Telescope monitored over
150,000 stars, watching for tiny dips in brightness caused by planets
passing in front of them — a method known as the transit technique.
Over nearly a decade, Kepler discovered thousands of exoplanet candidates,
transforming our understanding of how common planets are throughout
the galaxy. Each observation was flagged as a Kepler Object of Interest (KOI)
and later vetted to determine whether it represented a real planet or
a false signal caused by noise, eclipsing binaries, or instrumental artifacts.
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Dataset
# --------------------------------------------------
st.markdown("""
<div class="section-title">
📊 The Dataset
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass">
This project uses NASA's Kepler Objects of Interest (KOI) cumulative table,
containing measured and derived properties for each observed object —
orbital period, transit duration and depth, planet radius, equilibrium
temperature, insolation flux, signal-to-noise ratio, and stellar properties
such as temperature, surface gravity, and radius. Candidate rows were
excluded, leaving a binary classification task between two dispositions:
<b>Confirmed Planet</b> and <b>False Positive</b>.
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# ML Workflow
# --------------------------------------------------
st.markdown("""
<div class="section-title">
🤖 Machine Learning Workflow
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="glass">
    <b>1. Data Cleaning & Preprocessing</b><br>
    Handling missing values, removing candidate rows, encoding the target,
    and scaling numeric features.
    <br><br>
    <b>2. Exploratory Data Analysis</b><br>
    Distribution analysis, correlation study, and visual inspection of
    class separability across features.
    <br><br>
    <b>3. Model Training</b><br>
    Comparison of Logistic Regression, Random Forest, Gradient Boosting,
    and SVM classifiers.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass">
    <b>4. Hyperparameter Tuning</b><br>
    Grid/random search to optimize the Gradient Boosting pipeline.
    <br><br>
    <b>5. Evaluation</b><br>
    Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and
    calibration analysis on held-out test data.
    <br><br>
    <b>6. Deployment</b><br>
    The final pipeline is served through this interactive Streamlit
    dashboard for live predictions.
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# Technologies
# --------------------------------------------------
st.markdown("""
<div class="section-title">
🛠 Technologies Used
</div>
""", unsafe_allow_html=True)

tech_cols = st.columns(6)

techs = [
    ("🐍", "Python"),
    ("🎈", "Streamlit"),
    ("📊", "Plotly"),
    ("🐼", "Pandas"),
    ("🤖", "Scikit-learn"),
    ("🎨", "Custom CSS"),
]

for col, (icon, name) in zip(tech_cols, techs):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
            <h2>{icon}</h2>
            <p>{name}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Contributors
# --------------------------------------------------
st.markdown("""
<div class="section-title">
👥 Contributors
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="glass" style="text-align:center;">
    <h2>🧑‍🚀</h2>
    <h3>TODO_NAME_1</h3>
    <p>TODO_ROLE_1</p>
    <p>
    <a href="TODO_GITHUB_URL_1" style="color:#00d4ff;">GitHub</a> •
    <a href="TODO_LINKEDIN_URL_1" style="color:#00d4ff;">LinkedIn</a>
    </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass" style="text-align:center;">
    <h2>🧑‍🚀</h2>
    <h3>TODO_NAME_2</h3>
    <p>TODO_ROLE_2</p>
    <p>
    <a href="TODO_GITHUB_URL_2" style="color:#00d4ff;">GitHub</a> •
    <a href="TODO_LINKEDIN_URL_2" style="color:#00d4ff;">LinkedIn</a>
    </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:#9fc9ff;">
ExoVision • NASA Kepler Mission • Built with Streamlit
</div>
""", unsafe_allow_html=True)