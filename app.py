import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

from utils.data_loader import load_data, load_model
from utils.predictor import prepare_input

# ==================================================
# Page Config
# ==================================================
st.set_page_config(
    page_title="KOI Classifier — Kepler Exoplanet Search",
    page_icon="🪐",
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

# ==================================================
# Shared Resources
# ==================================================
df = load_data()
model = load_model()

LABELS = {"0": "False Positive", "1": "Confirmed"}
LABEL_MAP = {0: "False Positive", 1: "Confirmed Planet"}

FEATURES = [
    "koi_period", "koi_time0bk", "koi_impact", "koi_duration",
    "koi_depth", "koi_prad", "koi_teq", "koi_insol",
    "koi_model_snr", "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]

# ==================================================
# Sidebar Navigation
# ==================================================
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

    st.sidebar.info(f"""
        Mission Status

        🟢 Online

        {len(df):,} Records Loaded

        Machine Learning Ready
    """)

# ==================================================
# HOME
# ==================================================
if selected == "Home":

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

    st.markdown("""
        <div class='subtitle'>
        AI Powered Kepler Exoplanet Discovery Dashboard
        </div>
        """, unsafe_allow_html=True)

    confirmed = int((df["koi_disposition"] == 1).sum()) if df["koi_disposition"].dtype != object \
        else int((df["koi_disposition"] == "CONFIRMED").sum())
    false_pos = len(df) - confirmed

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
            <div class='metric-card'>
            <h2>🌍</h2>
            <h1>{len(df):,}</h1>
            <p>Observations</p>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class='metric-card'>
            <h2>🪐</h2>
            <h1>{confirmed:,}</h1>
            <p>Confirmed Planets</p>
            </div>
            """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class='metric-card'>
            <h2>❌</h2>
            <h1>{false_pos:,}</h1>
            <p>False Positives</p>
            </div>
            """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
            <div class='metric-card'>
            <h2>🤖</h2>
            <h1>5</h1>
            <p>Models Compared</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
        <div class="section-title">
        🚀 Mission Brief
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="glass">
        ExoVision is an AI-powered dashboard designed for exploring NASA's
        Kepler Exoplanet Search dataset. The application combines exploratory
        data analysis, interactive visualizations, and machine learning
        prediction into a unified experience inspired by a futuristic
        mission control center.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
        <div style="text-align:center;color:#8da8d6">
        Made with ❤️ using
        Python • Streamlit • Plotly • Scikit-Learn
        </div>
        """, unsafe_allow_html=True)

# ==================================================
# DATASET
# ==================================================
elif selected == "Dataset":

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

    rows, columns = len(df), len(df.columns)
    missing = int(df.isnull().sum().sum())
    numeric = len(df.select_dtypes(include="number").columns)

    c1, c2, c3, c4 = st.columns(4)
    cards = [("📄", rows, "Records"), ("🧬", columns, "Features"),
             ("🔢", numeric, "Numeric"), ("❌", missing, "Missing")]

    for col, (icon, value, label) in zip((c1, c2, c3, c4), cards):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                <h2>{icon}</h2>
                <h1>{value:,}</h1>
                <p>{label}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""<div class="section-title">📋 Dataset Preview</div>""", unsafe_allow_html=True)

    rows_to_show = st.slider("Rows to Display", 5, 100, 10)
    st.dataframe(df.head(rows_to_show), use_container_width=True, height=420)

    st.markdown("""<div class="section-title">🎛 Feature Selector</div>""", unsafe_allow_html=True)

    selected_cols = st.multiselect("Choose columns", df.columns, default=list(df.columns[:8]))
    if selected_cols:
        st.dataframe(df[selected_cols].head(rows_to_show), use_container_width=True)

    st.markdown("""<div class="section-title">📈 Summary Statistics</div>""", unsafe_allow_html=True)
    st.dataframe(df.describe().T, use_container_width=True, height=450)

    st.markdown("""<div class="section-title">❌ Missing Values</div>""", unsafe_allow_html=True)
    missing_df = pd.DataFrame({
        "Feature": df.columns,
        "Missing": df.isnull().sum(),
        "Percentage (%)": (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(missing_df, use_container_width=True)

    st.markdown("""<div class="section-title">📥 Download Dataset</div>""", unsafe_allow_html=True)
    st.download_button("Download CSV", df.to_csv(index=False), "cumulative_cleaned.csv", "text/csv")

# ==================================================
# EDA
# ==================================================
elif selected == "EDA":

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

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    st.markdown("""<div class="section-title">🌍 Planet Class Distribution</div>""", unsafe_allow_html=True)
    counts = (df["koi_disposition"].value_counts()
              .rename_axis("Disposition").reset_index(name="Count"))
    fig = px.bar(counts, x="Disposition", y="Count", color="Disposition", text="Count")
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="section-title">📊 Feature Distribution</div>""", unsafe_allow_html=True)
    feature = st.selectbox("Select Feature", numeric_columns)
    hist = px.histogram(df, x=feature, nbins=40, color_discrete_sequence=["#00D4FF"])
    hist.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(hist, use_container_width=True)

    st.markdown("""<div class="section-title">⭐ Feature Relationship</div>""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X Axis", numeric_columns, index=0, key="x")
    with col2:
        y_axis = st.selectbox("Y Axis", numeric_columns, index=1, key="y")

    scatter = px.scatter(df, x=x_axis, y=y_axis, color="koi_disposition", opacity=0.75)
    scatter.update_layout(template="plotly_dark", height=650)
    st.plotly_chart(scatter, use_container_width=True)

    st.markdown("""<div class="section-title">🔥 Correlation Heatmap</div>""", unsafe_allow_html=True)
    corr = df[numeric_columns].corr()
    heat = px.imshow(corr, aspect="auto", color_continuous_scale="Viridis", text_auto=".2f")
    heat.update_layout(template="plotly_dark", height=800)
    st.plotly_chart(heat, use_container_width=True)

# ==================================================
# PREDICT
# ==================================================
elif selected == "Predict":

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

    st.markdown("""<div class="section-title">📥 Enter Planet Parameters</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    inputs = {}

    for i, feat in enumerate(FEATURES):
        minimum = float(df[feat].min())
        maximum = float(df[feat].max())
        default = float(df[feat].median())
        target = col1 if i % 2 == 0 else col2
        with target:
            inputs[feat] = st.number_input(
                feat.replace("_", " ").title(),
                min_value=minimum, max_value=maximum, value=default,
            )

    st.markdown("---")

    if st.button("🚀 Predict Exoplanet", use_container_width=True):

        input_df = prepare_input(inputs)
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = probabilities.max() * 100

        st.markdown("""<div class="section-title">🌍 Prediction Result</div>""", unsafe_allow_html=True)

        if prediction == 1:
            st.success("🪐 CONFIRMED PLANET")
        else:
            st.error("❌ FALSE POSITIVE")

        st.metric("Prediction Confidence", f"{confidence:.2f}%")
        st.progress(confidence / 100)

        prob_df = pd.DataFrame({
            "Class": [LABEL_MAP.get(c, str(c)) for c in model.classes_],
            "Probability": probabilities
        })

        st.subheader("Class Probabilities")
        st.bar_chart(prob_df.set_index("Class"))

        st.subheader("Input Summary")
        st.dataframe(input_df, use_container_width=True)

# ==================================================
# PERFORMANCE
# ==================================================
elif selected == "Performance":

    metrics = pd.read_csv("results/metrics.csv")
    report = pd.read_csv("results/classification_report.csv", index_col=0)
    cm = pd.read_csv("results/confusion_matrix.csv", index_col=0)

    try:
        comparison = pd.read_csv("results/model_comparison.csv")
    except FileNotFoundError:
        comparison = None

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

    st.markdown("""<div class="section-title">🏆 Performance Metrics</div>""", unsafe_allow_html=True)

    metric_col, value_col = metrics.columns[0], metrics.columns[1]
    icon_map = {"Accuracy": "🎯", "Precision": "🔎", "Recall": "📡",
                "F1 Macro": "⚖️", "F1": "⚖️", "ROC-AUC": "📈", "ROC AUC": "📈"}

    cols = st.columns(len(metrics))
    for col, (_, row) in zip(cols, metrics.iterrows()):
        icon = icon_map.get(row[metric_col], "📊")
        score = float(row[value_col])
        with col:
            st.markdown(f"""
                <div class="metric-card">
                <h2>{icon}</h2>
                <h1>{score*100:.2f}%</h1>
                <p>{row[metric_col]}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""<div class="section-title">🔥 Confusion Matrix</div>""", unsafe_allow_html=True)

    cm_display = cm.copy()
    cm_display.index = [LABELS.get(str(i), str(i)) for i in cm_display.index]
    cm_display.columns = [LABELS.get(str(c), str(c)) for c in cm_display.columns]

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_display.values, x=cm_display.columns, y=cm_display.index,
        colorscale="Viridis", text=cm_display.values,
        texttemplate="%{text}", textfont={"size": 20},
    ))
    fig_cm.update_layout(template="plotly_dark", height=500,
                          xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("""<div class="section-title">📋 Classification Report</div>""", unsafe_allow_html=True)
    st.dataframe(report.rename(index=LABELS).round(3), use_container_width=True)

    st.markdown("""<div class="section-title">📊 Model Comparison</div>""", unsafe_allow_html=True)

    if comparison is not None:
        model_col = comparison.columns[0]
        numeric_cols = comparison.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            metric_choice = st.selectbox("Compare models by", numeric_cols)
            fig_comp = px.bar(
                comparison.sort_values(metric_choice, ascending=False),
                x=model_col, y=metric_choice, color=model_col, text=metric_choice
            )
            fig_comp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_comp.update_layout(template="plotly_dark", height=500, showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

        st.dataframe(comparison, use_container_width=True)
    else:
        st.info("Model comparison data not found.")

    acc_row = metrics[metrics[metric_col] == "Accuracy"]
    acc = float(acc_row[value_col].values[0]) if not acc_row.empty else None

    st.markdown("""<div class="section-title">🎯 Summary</div>""", unsafe_allow_html=True)
    if acc is not None:
        st.markdown(f"""
            <div class="glass">
            The Gradient Boosting model achieves <b>{acc*100:.2f}% accuracy</b>
            distinguishing Confirmed Exoplanets from False Positives.
            </div>
            """, unsafe_allow_html=True)

# ==================================================
# FEATURE IMPORTANCE
# ==================================================
elif selected == "Feature Importance":

    importance = pd.read_csv("results/feature_importance.csv")

    st.markdown("""
        <div class="hero">
        <div class="planet">⭐</div>
        <h1>Feature Importance</h1>
        <p>
        Which measurements matter most to the model when
        distinguishing planets from false positives.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<div class="section-title">⭐ Feature Ranking</div>""", unsafe_allow_html=True)

    fig_imp = px.bar(
        importance.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Viridis"
    )
    fig_imp.update_layout(template="plotly_dark", height=650)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""<div class="section-title">🏅 Top 5 Features</div>""", unsafe_allow_html=True)
    top5 = importance.sort_values("Importance", ascending=False).head(5)
    st.dataframe(top5, use_container_width=True)

# ==================================================
# ABOUT KEPLER
# ==================================================
elif selected == "About Kepler":

    st.markdown("""
        <div class="hero">
        <div class="planet">🌍</div>
        <h1>About ExoVision</h1>
        <p>
        An AI-powered dashboard for exploring NASA's Kepler mission data
        and classifying Kepler Objects of Interest.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<div class="section-title">🛰 The Kepler Mission</div>""", unsafe_allow_html=True)
    st.markdown("""
        <div class="glass">
        Launched by NASA in 2009, the Kepler Space Telescope monitored over
        150,000 stars, watching for tiny dips in brightness caused by planets
        passing in front of them — a method known as the transit technique.
        Over nearly a decade, Kepler discovered thousands of exoplanet
        candidates, transforming our understanding of how common planets
        are throughout the galaxy.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<div class="section-title">📊 The Dataset</div>""", unsafe_allow_html=True)
    st.markdown("""
        <div class="glass">
        This project uses NASA's Kepler Objects of Interest (KOI) cumulative
        table. Candidate rows were excluded, leaving a binary classification
        task between two dispositions: <b>Confirmed Planet</b> and
        <b>False Positive</b>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<div class="section-title">👥 Contributors</div>""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="glass" style="text-align:center;">
            <h2>🧑‍🚀</h2>
            <h3>Hadeed Qamar</h3>
            <p>Lead</p>
            <p>
            <a href="https://github.com/Hadeed07" style="color:#00d4ff;">GitHub</a> •
            <a href="TODO_LINKEDIN_URL_1" style="color:#00d4ff;">LinkedIn</a>
            </p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="glass" style="text-align:center;">
            <h2>🧑‍🚀</h2>
            <h3>Dawood Ahmad</h3>
            <p>Partner</p>
            <p>
            <a href="https://github.com/DawoodAhmadAnalyst" style="color:#00d4ff;">GitHub</a> •
            <a href="https://www.linkedin.com/in/dawood-ahmad-90676b319/" style="color:#00d4ff;">LinkedIn</a>
            </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
        <div style="text-align:center;color:#9fc9ff;">
        ExoVision • NASA Kepler Mission • Built with Streamlit
        </div>
        """, unsafe_allow_html=True)