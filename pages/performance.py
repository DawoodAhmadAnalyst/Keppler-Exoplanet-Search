import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Model Performance",
    page_icon="📉",
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
# Load Results
# --------------------------------------------------
metrics = pd.read_csv("results/metrics.csv")
report = pd.read_csv("results/classification_report.csv", index_col=0)
cm = pd.read_csv("results/confusion_matrix.csv", index_col=0)
importance = pd.read_csv("results/feature_importance.csv")

LABELS = {"0": "False Positive", "1": "Confirmed"}

# --------------------------------------------------
# Hero
# --------------------------------------------------
st.markdown("""
<div class="hero">
<div class="planet">📉</div>
<h1>Model Performance</h1>
<p>
Evaluating the Gradient Boosting pipeline trained to distinguish
Confirmed Exoplanets from False Positives.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Model Evaluation Dashboard
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# KPI Cards
# --------------------------------------------------
st.markdown("""
<div class="section-title">
🏆 Performance Metrics
</div>
""", unsafe_allow_html=True)

cols = st.columns(len(metrics))

icon_map = {
    "Accuracy": "🎯",
    "Precision": "🔎",
    "Recall": "📡",
    "F1 Macro": "⚖️",
    "ROC-AUC": "📈"
}

for col, (_, row) in zip(cols, metrics.iterrows()):
    icon = icon_map.get(row["Metric"], "📊")
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
            <h2>{icon}</h2>
            <h1>{row['Score']*100:.2f}%</h1>
            <p>{row['Metric']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
st.markdown("""
<div class="section-title">
🔥 Confusion Matrix
</div>
""", unsafe_allow_html=True)

cm.index = [LABELS.get(str(i), str(i)) for i in cm.index]
cm.columns = [LABELS.get(str(c), str(c)) for c in cm.columns]

fig_cm = go.Figure(data=go.Heatmap(
    z=cm.values,
    x=cm.columns,
    y=cm.index,
    colorscale="Viridis",
    text=cm.values,
    texttemplate="%{text}",
    textfont={"size": 20},
))

fig_cm.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_title="Predicted",
    yaxis_title="Actual"
)

st.plotly_chart(fig_cm, use_container_width=True)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------
st.markdown("""
<div class="section-title">
📋 Classification Report
</div>
""", unsafe_allow_html=True)

report_display = report.rename(index=LABELS)

st.dataframe(
    report_display.round(3),
    use_container_width=True
)

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
st.markdown("""
<div class="section-title">
⭐ Feature Importance
</div>
""", unsafe_allow_html=True)

fig_imp = px.bar(
    importance.sort_values("Importance"),
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="Viridis"
)

fig_imp.update_layout(
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig_imp, use_container_width=True)

# --------------------------------------------------
# MODEL COMPARISON SECTION GOES HERE
# --------------------------------------------------

# --------------------------------------------------
# Summary
# --------------------------------------------------
st.markdown("""
<div class="section-title">
🎯 Summary
</div>
""", unsafe_allow_html=True)

acc = metrics.loc[metrics["Metric"] == "Accuracy", "Score"].values[0]

st.markdown(f"""
<div class="glass">
The Gradient Boosting model achieves <b>{acc*100:.2f}% accuracy</b> distinguishing
Confirmed Exoplanets from False Positives in NASA's Kepler dataset. The model
balances precision and recall effectively, making it suitable for automated
triage of Kepler Objects of Interest before manual vetting.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:#9fc9ff;">
ExoVision • Model Performance • NASA Kepler Mission
</div>
""", unsafe_allow_html=True)