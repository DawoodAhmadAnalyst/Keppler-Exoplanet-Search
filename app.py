import streamlit as st
import pandas as pd
import textwrap
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    accuracy_score, f1_score, roc_auc_score
)

from utils.data_loader import load_data, load_model, load_params
from utils.predictor import prepare_input

# ==================================================
# Page Config
# ==================================================
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



@st.cache_resource
def build_pipeline_artifacts(_df_hash):
    """
    Reconstruct the exact preprocessing used in Modelling.ipynb:
    stratified 70/30 split (random_state=42) -> log1p on all 13 features -> StandardScaler
    fit on the training fold. Verified to reproduce the notebook's reported test-set
    metrics (Accuracy 0.9264 / F1-macro 0.9169 / ROC-AUC 0.9729) exactly.
    """
    df = load_data()
    X = df[FEATURES]
    y = df['koi_disposition']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[FEATURES] = np.log1p(X_train[FEATURES])
    X_test[FEATURES] = np.log1p(X_test[FEATURES])
 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    model = load_model()
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
 
    return {
        "scaler": scaler,
        "X_test_raw": X.loc[X_test.index],
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
 
def predict_one(raw_row: dict, scaler, model):
    x = pd.DataFrame([raw_row])[FEATURES]
    x[FEATURES] = np.log1p(x[FEATURES])
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    proba = model.predict_proba(x_scaled)[0, 1]
    return pred, proba


local_css("css/style.css")

# ==================================================
# Shared Resources
# ==================================================




LABELS = {"0": "False Positive", "1": "Confirmed"}
LABEL_MAP = {0: "False Positive", 1: "Confirmed Planet"}

FEATURES = [
    "koi_period", "koi_time0bk", "koi_impact", "koi_duration",
    "koi_depth", "koi_prad", "koi_teq", "koi_insol",
    "koi_model_snr", "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]

FEATURE_META = {
    'koi_period':    {'label': 'Orbital Period',        'unit': 'days',        'group': 'signal', 'help': 'Time for one full orbit around the host star.'},
    'koi_time0bk':   {'label': 'Transit Epoch',          'unit': 'BKJD',        'group': 'signal', 'help': 'Time of the first observed transit (Barycentric Kepler Julian Day).'},
    'koi_impact':    {'label': 'Impact Parameter',       'unit': '',            'group': 'signal', 'help': 'How centrally the planet crosses the stellar disk. 0 = through the center, 1 = grazing the edge.'},
    'koi_duration':  {'label': 'Transit Duration',       'unit': 'hours',       'group': 'signal', 'help': 'How long the transit lasts, start to finish.'},
    'koi_depth':     {'label': 'Transit Depth',          'unit': 'ppm',         'group': 'signal', 'help': 'How much the star\u2019s brightness dips during transit.'},
    'koi_model_snr': {'label': 'Transit Signal-to-Noise','unit': '',            'group': 'signal', 'help': 'Strength of the transit signal relative to noise in the light curve.'},
    'koi_prad':      {'label': 'Planetary Radius',       'unit': 'Earth radii', 'group': 'planet', 'help': 'Radius of the candidate planet, in units of Earth\u2019s radius.'},
    'koi_teq':       {'label': 'Equilibrium Temperature','unit': 'K',           'group': 'planet', 'help': 'Estimated surface temperature assuming a simple energy balance.'},
    'koi_insol':     {'label': 'Insolation Flux',        'unit': '\u2295 flux', 'group': 'planet', 'help': 'Radiation the planet receives, relative to what Earth receives from the Sun.'},
    'koi_steff':     {'label': 'Stellar Eff. Temperature','unit': 'K',          'group': 'star',   'help': 'Surface temperature of the host star.'},
    'koi_slogg':     {'label': 'Stellar Surface Gravity','unit': 'log\u2081\u2080(cm/s\u00b2)', 'group': 'star', 'help': 'Surface gravity of the host star.'},
    'koi_srad':      {'label': 'Stellar Radius',         'unit': 'Solar radii', 'group': 'star',   'help': 'Radius of the host star, in units of the Sun\u2019s radius.'},
    'koi_kepmag':    {'label': 'Kepler-band Magnitude',  'unit': 'mag',         'group': 'star',   'help': 'Apparent brightness of the host star as seen by Kepler. Lower = brighter.'},
}
 
GROUP_LABELS = {
    'signal': 'Transit Signal — measured directly from the light curve',
    'planet': 'Planet Properties — derived quantities',
    'star':   'Host Star — stellar properties',
}

df = load_data()
model = load_model()
artifacts = build_pipeline_artifacts(len(df))

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
        <div class="eyebrow">
            NASA KEPLER MISSION • AI POWERED DISCOVERY
        </div>
        <div class="hero-title">
            🌌 ExoVision
        </div>
        <div class="hero-sub">
            Explore thousands of Kepler Objects of Interest, analyze planetary
            characteristics through interactive visualizations, and predict
            potential exoplanets using a machine learning model trained on
            NASA's Kepler mission data.
        </div>

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

    st.markdown(f"""
        <div class="stat-strip">

        <div class="stat-block">
            <div class="stat-value">{len(df):,}</div>
            <div class="stat-label">Observations</div>
        </div>

        <div class="stat-block">
            <div class="stat-value">{confirmed:,}</div>
            <div class="stat-label">Confirmed</div>
        </div>

        <div class="stat-block">
            <div class="stat-value">{false_pos:,}</div>
            <div class="stat-label">False Positives</div>
        </div>

        <div class="stat-block">
            <div class="stat-value">{len(FEATURES)}</div>
            <div class="stat-label">Features</div>
        </div>

        <div class="stat-block">
            <div class="stat-value">92.6%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
        <div class="panel">

        <div class="panel-title">
        🚀 Mission Brief
        </div>

        ExoVision is an AI-powered dashboard designed for exploring NASA's
        Kepler Exoplanet Search dataset. It combines exploratory data analysis,
        interactive visualizations, and machine learning prediction into a
        single mission-control inspired experience for discovering worlds
        beyond our Solar System.

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
        <div class="planet"></div>
        <h1>📈Exploratory Data Analysis</h1>
        <p>
        Understand the characteristics of NASA's Kepler Exoplanet dataset
        through interactive visualizations.
        </p>
        </div>
        """, unsafe_allow_html=True)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    common_layout = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500, margin=dict(l=20, r=20, t=20, b=20), font=dict(family="Space Grotesk",color="#eef5ff", size=14))

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("""<div class="panel"><div class="panel-title">Planet Class Distribution </div>""", unsafe_allow_html=True)

        header_left, header_right = st.columns([4,1])
        with header_left:
            st.caption("Kepler KOI Disposition")
        with header_right:
            st.write("")

        counts = (df["koi_disposition"].value_counts().rename_axis("Disposition").reset_index (name="Count"))

        fig = px.bar(counts, x="Disposition", y="Count", color="Disposition", text="Count", color_continuous_scale='Blues')
        fig.update_traces(textposition='outside')
        fig.update_layout(**common_layout, showlegend=False, bargap=0.35)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("""<div class="panel"><div class="panel-title">Features Distribution </div>""", unsafe_allow_html=True)

        control1, control2 = st.columns([2, 1])
        with control1:
            feature = st.selectbox("Feature", numeric_columns, label_visibility="collapsed")
        with control2:
            log_transform = st.toggle("Log Transformation", value=False)

        plot_data = df[feature].copy()
        if log_transform:
            plot_data = np.log1p(plot_data)
        skewness = df[feature].skew()

        hist = px.histogram(df, x=plot_data, nbins=40, color_discrete_sequence=["#00D4FF"])
        hist.update_layout(**common_layout)
        st.plotly_chart(hist, use_container_width=True)
        
        if abs(skewness) > 1:
            color = "#ff7070"
            status = "Highly Skewed"
        elif abs(skewness) > 0.5:
            color = "#f5c76b"
            status = "Moderately Skewed"
        else:
            color = "#5ee5b8"
            status = "Nearly Symmetric"

        st.markdown(f"""
        <div class="metric-caption"><div><b>Skewness</b><br><small>{status}</small></div> <div style=" font-size:20px; font-weight:700; color:{color}; ">{skewness:.2f} </div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="panel"><div class="panel-title">Features Relationships </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X Axis", numeric_columns, label_visibility="collapsed", index=0, key="x")
    with col2:
        y_axis = st.selectbox("Y Axis", numeric_columns, label_visibility="collapsed", index=1, key="y")

    scatter = px.scatter(df, x=x_axis, y=y_axis, color="koi_disposition", opacity=0.75)
    scatter.update_layout(template="plotly_dark", height=650)
    st.plotly_chart(scatter, use_container_width=True)
    

    st.markdown("""<div class="panel"><div class="panel-title">Correlation Heatmap</div>""", unsafe_allow_html=True)
    corr = df[numeric_columns].corr()
    show_values = st.toggle("Show Values", value=True)
    heat = px.imshow(corr, aspect="auto", color_continuous_scale="Viridis", text_auto=".2f" if show_values else False)
    heat.update_layout(template="plotly_dark", height=800)
    st.plotly_chart(heat, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# PREDICT
# ==================================================
elif selected == "Predict":

    st.markdown("""
        <div class="hero">
        <div class="planet"></div>
        <h1>🤖 AI Exoplanet Prediction</h1>
        <p>
        Predict whether a Kepler Object of Interest is a
        Confirmed Planet or a False Positive.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="panel">
        <div class="panel-title">
            ⚙ Candidate Parameters
        </div>

        <p style="margin-top:10px; color:#b7c9e2;">
            Adjust the numerical characteristics of the selected Kepler candidate.
            These parameters will be analyzed by the trained ExoVision classifier.
        </p>
        </div>
        """, unsafe_allow_html=True)


    X_test = artifacts["X_test_raw"]
    y_test = artifacts["y_test"]

    def load_example(label=None):
        if label is None:
            idx = np.random.choice(X_test.index)
        else:
            idx_pool = y_test[y_test == label].index
            idx = np.random.choice(idx_pool)

        row = X_test.loc[idx]
        for feat in FEATURES:
            st.session_state[f"inp_{feat}"] = float(row[feat])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🟢 Confirmed", use_container_width=True):
            load_example(1)

    with col2:
        if st.button("🔴 False Positive", use_container_width=True):
            load_example(0)

    with col3:
        if st.button("🎲 Random", use_container_width=True):
            load_example()

    with col4:
        if st.button("♻ Reset", use_container_width=True):
            for feature in FEATURES:
                st.session_state.pop(f"inp_{feature}", None)


    fcols = st.columns(3)
    raw_input = {}

    for i, grp in enumerate(['signal', 'planet', 'star']):

        with fcols[i]:
            st.markdown(f'<div class="group-label">{GROUP_LABELS[grp]}</div>', unsafe_allow_html=True)

            for feat, meta in FEATURE_META.items():
                if meta['group'] != grp:
                    continue
            
                col_data = df[feat]
                lo, hi = float(col_data.min()), float(col_data.max())
                step = (hi - lo) / 200 if hi > lo else 0.1
                unit = f" ({meta['unit']})" if meta['unit'] else ""
                st.session_state.setdefault(f"inp_{feat}", float(col_data.median()))
                raw_input[feat] = st.number_input(
                    f"{meta['label']}{unit}", min_value=lo, max_value=hi, step=step,
                    key=f"inp_{feat}", help=meta["help"],
                )


    

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="panel">
        <div class="panel-title">
        🚀 AI Analysis Console
        </div>

        <p style="color:#b7c9e2;margin-top:10px;">
        When ready, launch the neural classifier to evaluate the selected
        Kepler candidate.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    run = st.button("Launch AI Analysis", use_container_width=True)
    if run:
        scaler = artifacts['scaler']
        pred, proba = predict_one(raw_input, scaler, model)
        if pred == 1:
            st.markdown(textwrap.dedent(f"""
            <div class="result-card confirmed">
                <div class="result-verdict confirmed">🟢 Confirmed Exoplanet</div>
                <div class="result-prob">Model confidence: {proba:.1%}</div>
            </div>
            """), unsafe_allow_html=True)
        else:
            st.markdown(textwrap.dedent(f"""
            <div class="result-card fp">
                <div class="result-verdict fp">🔴 False Positive</div>
                <div class="result-prob">Confirmed-planet probability: {proba:.1%}</div>
            </div>
            """), unsafe_allow_html=True)
        
        figgauge = go.Figure(go.Bar(
            x=[proba], y=["P(confirmed)"], orientation="h",
            marker=dict(color="#4fd1ae" if proba >= 0.5 else "#e0685a"),
        ))
        figgauge.add_vline(x=0.5, line=dict(color="#8c96ad", dash="dot"))
        figgauge.update_layout(template='plotly_dark', height=110, xaxis_range=[0, 1],
                                margin=dict(t=10, b=25, l=10, r=10), showlegend=False)
        st.plotly_chart(figgauge, width='stretch', config={"displayModeBar": False})


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
        <div class="planet"></div>
        <h1>📈 Model Performance</h1>
        <p>
        Evaluating the Gradient Boosting pipeline trained to distinguish
        Confirmed Exoplanets from False Positives.
        </p>
        </div>
        """, unsafe_allow_html=True)
    

    st.markdown("""
        <div class="panel">
            <div class="panel-title">
                🏆 Performance Overview
            </div>
        </div>
        """, unsafe_allow_html=True)

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

    st.markdown("""
        <div class="panel">
            <div class="panel-title">
                🔥 Confusion Matrix
            </div>
        </div>
        """, unsafe_allow_html=True)

    cm_display = cm.copy()
    cm_display.index = [LABELS.get(str(i), str(i)) for i in cm_display.index]
    cm_display.columns = [LABELS.get(str(c), str(c)) for c in cm_display.columns]

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_display.values, x=cm_display.columns, y=cm_display.index,
        colorscale="Blues", text=cm_display.values,
        texttemplate="%{text}", textfont=dict(size=18)))

    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk",color="#eef5ff",size=14),
        xaxis=dict(title="Predicted Class",gridcolor="rgba(255,255,255,.08)",zeroline=False),
        yaxis=dict(title="Actual Class",gridcolor="rgba(255,255,255,.08)",zeroline=False),
        margin=dict(l=20,r=20,t=20,b=20),
        height=480
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("""
        <div class="panel">
            <div class="panel-title">
                📋 Classification Report
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.dataframe(report.rename(index=LABELS).round(3), use_container_width=True)

    st.markdown("""
        <div class="panel">
        <div class="panel-title">
            📊 Model Comparison
        </div>

        <p style="margin-top:10px;color:#b7c9e2;">
            Compare the Gradient Boosting classifier against the alternative
            machine learning models evaluated during experimentation.
        </p>
        </div>
        """, unsafe_allow_html=True)

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

    st.markdown(f"""
        <div class="panel">
        <div class="panel-title">
            🚀 Mission Assessment
        </div>

        <p style="margin-top:12px;line-height:1.8;color:#d8e7ff;">
            The Gradient Boosting classifier achieved an overall accuracy of
            <strong>{acc*100:.2f}%</strong> on the held-out test set,
            demonstrating strong generalization for distinguishing confirmed
            exoplanets from false positive detections.
        </p>
        </div>
        """, unsafe_allow_html=True)

# ==================================================
# FEATURE IMPORTANCE
# ==================================================
elif selected == "Feature Importance":

    importance = pd.read_csv("results/feature_importance.csv")
    importance = importance.sort_values("Importance", ascending=False)

    st.markdown("""
        <div class="hero">
        <div class="planet"></div>
        <h1>Feature Importance</h1>
        <p>
        Which measurements matter most to the model when
        distinguishing planets from false positives.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="panel">
            <div class="panel-title">
                ⭐ Feature Importance Ranking
            </div>
            <p style="margin-top:10px;color:#b7c9e2;">
            Larger importance values indicate that the feature contributes
            more strongly to the model's decision-making process.
            </p>
        </div>
        """, unsafe_allow_html=True)

    fig_imp = px.bar(importance.sort_values("Importance", ascending=True), x="Importance", y="Feature", orientation="h", color="Importance", color_discrete_sequence=["#41d6ff"])
    fig_imp.update_traces(texttemplate="%{x:.3f}", textposition="outside")

    fig_imp.update_layout(template="plotly_dark", height=650)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    top5 = importance.head(5)
    st.markdown("""
        <div class="stat-strip">
        """, unsafe_allow_html=True)

    cols = st.columns(5)
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

    for col, medal, (_, row) in zip(cols, medals, top5.iterrows()):

        feature_name = FEATURE_META.get(
            row["Feature"],
            {}
        ).get("label", row["Feature"])

        with col:
            st.markdown(f"""
            <div class="stat-block">
            <div style="font-size:1.8rem; margin-bottom:8px;">
                {medal}
            </div>
            <div class="stat-label" style="
                min-height:42px;
                font-size:0.9rem;
                line-height:1.35;
                margin-bottom:10px;
                color:#eef5ff;
                font-weight:600;
            ">
                {feature_name}
            </div>

            <div class="stat-value">
                {row["Importance"]:.3f}
            </div>
            <div class="stat-label">
                Importance
            </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""</div>""", unsafe_allow_html=True)

    top_names = [
        FEATURE_META.get(f, {}).get("label", f)
        for f in top5["Feature"][:3]
    ]

    st.markdown(f"""
        <div class="panel">
        <div class="panel-title">
            💡 Model Interpretation
        </div>

        <p style="margin-top:12px;line-height:1.8;color:#d8e7ff;">

        The Gradient Boosting classifier relies most heavily on
        <strong>{top_names[0]}</strong>,
        <strong>{top_names[1]}</strong>, and
        <strong>{top_names[2]}</strong> when evaluating Kepler Objects of
        Interest.

        Features with higher importance values contribute more strongly to
        the final prediction, while lower-ranked variables provide
        complementary information that helps refine the classifier's
        decision boundary.

        </p>
        </div>
        """, unsafe_allow_html=True)

# ==================================================
# ABOUT KEPLER
# ==================================================
elif selected == "About Kepler":

    st.markdown("""
        <div class="hero">
        <div class="planet"></div>
        <h1>About ExoVision</h1>
        <p>
        An AI-powered dashboard for exploring NASA's Kepler mission data
        and classifying Kepler Objects of Interest.
        </p>
        </div>
        """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:

        st.markdown("""
        <div class="panel">
        <div class="panel-title">
            🛰 The Kepler Mission
        </div>

        <p style="margin-top:12px;line-height:1.8;color:#d8e7ff;">
        Launched by NASA in 2009, the Kepler Space Telescope continuously
        monitored more than <strong>150,000 stars</strong> searching for
        tiny decreases in stellar brightness caused by planetary transits.

        During nearly a decade of observations, the mission identified
        thousands of exoplanet candidates and fundamentally changed our
        understanding of planetary systems throughout the Milky Way.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="panel">
        <div class="panel-title">
            📊 The Dataset
        </div>

        <p style="margin-top:12px;line-height:1.8;color:#d8e7ff;">
        ExoVision uses NASA's Kepler Objects of Interest (KOI) cumulative
        dataset.

        Candidate observations were removed to formulate a binary
        classification problem in which the model predicts whether a
        Kepler candidate represents a
        <strong>Confirmed Exoplanet</strong> or a
        <strong>False Positive</strong>.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown("""
    <div class="panel">
    <div class="panel-title">
        ⚙ Technology Stack
    </div>

    <p style="margin-top:10px;color:#b7c9e2;">
    ExoVision combines modern data science, machine learning,
    and interactive visualization libraries to deliver an intuitive
    exploration experience.
    </p>
    </div>
    """, unsafe_allow_html=True)


    # Technology Stack

    st.markdown("""
        <div class="stat-strip">

        <div class="stat-block">
            <div class="stat-value">🐍</div>
            <div class="stat-label">Python</div>
        </div>

        <div class="stat-block">
            <div class="stat-value">🚀</div>
            <div class="stat-label">Streamlit</div>
        </div>

        <div class="stat-block">
            <div class="stat-value">📊</div>
            <div class="stat-label">Plotly</div>
        </div>

        <div class="stat-block">
            <div class="stat-value">🤖</div>
            <div class="stat-label">Scikit-Learn</div>
        </div>

        <div class="stat-block">
            <div class="stat-value">🌌</div>
            <div class="stat-label">NASA Kepler</div>
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Contributers
    st.markdown("""
    <div class="panel">
    <div class="panel-title">
        👨‍🚀 Development Team
    </div>

    <p style="margin-top:10px;color:#b7c9e2;">
    ExoVision was developed as an undergraduate machine learning
    and scientific visualization project.
    </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        <div class="panel" style="text-align:center;min-height:270px;">

        <div style="font-size:3rem;">👨‍💻</div>

        <h2 style="margin-top:10px;">
        Hadeed Qamar
        </h2>

        <p style="color:#41d6ff;font-weight:600;">
        ML Developer
        </p>

        <p style="line-height:1.7;color:#d8e7ff;">
        Machine Learning • Data Science •
        Dashboard Design
        </p>

        <p>
        <a href="https://github.com/Hadeed07" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/hadeed-qamar-40338a235/">LinkedIn</a>
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="panel" style="text-align:center;min-height:270px;">

        <div style="font-size:3rem;">👨‍💻</div>

        <h2 style="margin-top:10px;">
        Dawood Ahmad
        </h2>

        <p style="color:#41d6ff;font-weight:600;">
        Data Analyst
        </p>

        <p style="line-height:1.7;color:#d8e7ff;">
        Machine Learning • Data Analysis •
        Feature Engineering
        </p>

        <p>
        <a href="https://github.com/DawoodAhmadAnalyst" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/dawood-ahmad-90676b319/" target="_blank">LinkedIn</a>
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
        <div class="panel">
        <div class="panel-title">
            🚀 Project Highlights
        </div>

        <p style="margin-top:12px;line-height:1.9;color:#d8e7ff;">
        ✅ Interactive exploration of NASA's Kepler dataset.<br>
        ✅ Comprehensive exploratory data analysis and visualization.<br>
        ✅ Gradient Boosting classifier for exoplanet prediction.<br>
        ✅ AI-powered prediction console with live inference.<br>
        ✅ Performance evaluation and explainable machine learning tools.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Footer

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="
            text-align:center;
            padding:30px 10px;
            color:#9fb4c8;
            line-height:1.8;
        ">

        <h3 style="color:#eef5ff;">
        🌌 ExoVision
        </h3>

        An interactive AI dashboard inspired by NASA's Kepler Mission,
        combining machine learning, scientific visualization,
        and modern web technologies to explore the search for worlds
        beyond our Solar System.

        <br><br>

        Built with ❤️ using
        <strong>Python</strong>,
        <strong>Streamlit</strong>,
        <strong>Plotly</strong>,
        and
        <strong>Scikit-Learn</strong>.

        </div>
        """, unsafe_allow_html=True)