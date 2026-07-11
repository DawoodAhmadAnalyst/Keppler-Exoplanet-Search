import streamlit as st
import pandas as pd

from utils.data_loader import load_data

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="📊",
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

# --------------------------------------------------
# Hero Section
# --------------------------------------------------
st.markdown("""
<div class="hero">

<div class="planet">📊</div>

<h1>Dataset Explorer</h1>

<p>

Explore NASA's Kepler Exoplanet dataset used for
training the machine learning model.

</p>

</div>
""", unsafe_allow_html=True)

st.markdown(
"""
<div class="subtitle">

Cleaned Kepler Exoplanet Dataset

</div>
""",
unsafe_allow_html=True
)

# --------------------------------------------------
# KPI Cards
# --------------------------------------------------

rows = len(df)
columns = len(df.columns)
missing = int(df.isnull().sum().sum())
numeric = len(df.select_dtypes(include="number").columns)

c1, c2, c3, c4 = st.columns(4)

cards = [
    ("📄", rows, "Records"),
    ("🧬", columns, "Features"),
    ("🔢", numeric, "Numeric"),
    ("❌", missing, "Missing"),
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

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.markdown("""
<div class="section-title">

📋 Dataset Preview

</div>
""", unsafe_allow_html=True)

rows_to_show = st.slider(
    "Rows to Display",
    5,
    100,
    10
)

st.dataframe(
    df.head(rows_to_show),
    use_container_width=True,
    height=420
)

# --------------------------------------------------
# Feature Selector
# --------------------------------------------------

st.markdown("""
<div class="section-title">

🎛 Feature Selector

</div>
""", unsafe_allow_html=True)

selected = st.multiselect(
    "Choose columns",
    df.columns,
    default=df.columns[:8]
)

if selected:
    st.dataframe(
        df[selected].head(rows_to_show),
        use_container_width=True
    )

# --------------------------------------------------
# Summary Statistics
# --------------------------------------------------

st.markdown("""
<div class="section-title">

📈 Summary Statistics

</div>
""", unsafe_allow_html=True)

st.dataframe(
    df.describe().T,
    use_container_width=True,
    height=450
)

# --------------------------------------------------
# Data Types
# --------------------------------------------------

st.markdown("""
<div class="section-title">

🧬 Feature Data Types

</div>
""", unsafe_allow_html=True)

dtype_df = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(
    dtype_df,
    use_container_width=True
)

# --------------------------------------------------
# Missing Values
# --------------------------------------------------

st.markdown("""
<div class="section-title">

❌ Missing Values

</div>
""", unsafe_allow_html=True)

missing_df = pd.DataFrame({
    "Feature": df.columns,
    "Missing": df.isnull().sum(),
    "Percentage (%)":
        (df.isnull().sum()/len(df)*100).round(2)
})

st.dataframe(
    missing_df,
    use_container_width=True
)

# --------------------------------------------------
# Download Dataset
# --------------------------------------------------

st.markdown("""
<div class="section-title">

📥 Download Dataset

</div>
""", unsafe_allow_html=True)

st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    "cumulative_cleaned.csv",
    "text/csv"
)

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;color:#9fc9ff;">

NASA Kepler Mission • Dataset Explorer • ExoVision

</div>
""",
unsafe_allow_html=True
)