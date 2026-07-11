import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# Load CSS
# =====================================================
with open("css/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# =====================================================
# Load Dataset
# =====================================================
df = load_data()

# =====================================================
# Hero Section
# =====================================================
st.markdown("""
<div class="hero">

<div class="planet">📈</div>

<h1>Exploratory Data Analysis</h1>

<p>
Understand the characteristics of NASA's Kepler Exoplanet dataset
through interactive visualizations.
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Interactive Data Visualization Dashboard
</div>
""", unsafe_allow_html=True)

# =====================================================
# KPI Cards
# =====================================================
rows = len(df)
columns = len(df.columns)
numeric = len(df.select_dtypes(include="number").columns)
classes = df["koi_disposition"].nunique()

c1, c2, c3, c4 = st.columns(4)

cards = [
    ("📄", rows, "Records"),
    ("🧬", columns, "Features"),
    ("🔢", numeric, "Numeric Features"),
    ("🌍", classes, "Planet Classes")
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

# =====================================================
# Planet Class Distribution
# =====================================================
st.markdown("""
<div class="section-title">
🌍 Planet Class Distribution
</div>
""", unsafe_allow_html=True)

counts = (
    df["koi_disposition"]
    .value_counts()
    .rename_axis("Disposition")
    .reset_index(name="Count")
)

fig = px.bar(
    counts,
    x="Disposition",
    y="Count",
    color="Disposition",
    text="Count",
)

fig.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_title="Planet Class",
    yaxis_title="Count"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Histogram
# =====================================================
st.markdown("""
<div class="section-title">
📊 Feature Distribution
</div>
""", unsafe_allow_html=True)

numeric_columns = df.select_dtypes(include="number").columns.tolist()

feature = st.selectbox(
    "Select Feature",
    numeric_columns
)

hist = px.histogram(
    df,
    x=feature,
    nbins=40,
    color_discrete_sequence=["#00D4FF"]
)

hist.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(hist, use_container_width=True)

# =====================================================
# Scatter Plot
# =====================================================
st.markdown("""
<div class="section-title">
⭐ Feature Relationship
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    x_axis = st.selectbox(
        "X Axis",
        numeric_columns,
        index=0,
        key="x"
    )

with col2:
    y_axis = st.selectbox(
        "Y Axis",
        numeric_columns,
        index=1,
        key="y"
    )

scatter = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color="koi_disposition",
    opacity=0.75,
    hover_data=["koi_disposition"]
)

scatter.update_layout(
    template="plotly_dark",
    height=650
)

st.plotly_chart(scatter, use_container_width=True)

# =====================================================
# Correlation Heatmap
# =====================================================
st.markdown("""
<div class="section-title">
🔥 Correlation Heatmap
</div>
""", unsafe_allow_html=True)

corr = df[numeric_columns].corr()

heat = px.imshow(
    corr,
    aspect="auto",
    color_continuous_scale="Viridis",
    text_auto=".2f"
)

heat.update_layout(
    template="plotly_dark",
    height=800
)

st.plotly_chart(heat, use_container_width=True)

# =====================================================
# Summary Statistics
# =====================================================
st.markdown("""
<div class="section-title">
📈 Summary Statistics
</div>
""", unsafe_allow_html=True)

st.dataframe(
    df.describe().T,
    use_container_width=True,
    height=500
)

# =====================================================
# Footer
# =====================================================
st.markdown("---")

st.markdown("""
<div style="text-align:center;color:#9fc9ff;">
ExoVision • Exploratory Data Analysis • NASA Kepler Mission
</div>
""", unsafe_allow_html=True)