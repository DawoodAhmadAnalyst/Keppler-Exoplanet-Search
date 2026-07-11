import streamlit as st
import textwrap
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    accuracy_score, f1_score, roc_auc_score
)

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="KOI Classifier — Kepler Exoplanet Search",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FEATURES = ['koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth',
            'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
            'koi_slogg', 'koi_srad', 'koi_kepmag']

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

# ----------------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown(textwrap.dedent("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    """), unsafe_allow_html=True)

    with open("css/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def light_curve_svg(width=1180, height=90, dip_frac=0.5, color="var(--gold)"):
    """A stylised transit light curve: flat brightness, then a dip, then recovery."""
    w, h = width, height
    baseline = h * 0.32
    dip = h * 0.72
    x0 = w * dip_frac - 90
    x1 = w * dip_frac - 45
    x2 = w * dip_frac + 45
    x3 = w * dip_frac + 90
    path = f"M0,{baseline} L{x0},{baseline} L{x1},{dip} L{x2},{dip} L{x3},{baseline} L{w},{baseline}"
    svg = f"""<div class="lightcurve-wrap">
    <svg width="100%" height="{h}" viewBox="0 0 {w} {h}" preserveAspectRatio="none" style="display:block;">
        <path d="{path}" fill="none" stroke="{color}" stroke-width="2" opacity="0.85"/>
        <circle cx="{x1 - 15}" cy="{dip}" r="3.5" fill="{color}">
            <animate attributeName="cx" values="0;{w}" dur="7s" repeatCount="indefinite" />
            <animate attributeName="cy" values="{baseline};{baseline};{dip};{dip};{baseline};{baseline}"
                     keyTimes="0;{max(x0/w-0.02,0):.3f};{x0/w:.3f};{x3/w:.3f};{min(x3/w+0.02,1):.3f};1"
                     dur="7s" repeatCount="indefinite" />
        </circle>
    </svg>
    </div>"""
    return textwrap.dedent(svg)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", color="#c7cede", size=12),
        xaxis=dict(gridcolor="#1e2940", zerolinecolor="#1e2940", linecolor="#1e2940"),
        yaxis=dict(gridcolor="#1e2940", zerolinecolor="#1e2940", linecolor="#1e2940"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=40, l=50, r=20),
        colorway=["#4fd1ae", "#e0685a", "#e8a94c", "#7aa2f7"],
    )
)

# ----------------------------------------------------------------------------
# DATA / MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Datasets/cumulative_cleaned.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

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

# ----------------------------------------------------------------------------
# STATIC RESULTS FROM THE NOTEBOOK (Modelling.ipynb) — for cross-validated
# and held-out comparison tables across all 5 candidate pipelines.
# ----------------------------------------------------------------------------
TEST_SET_COMPARISON = pd.DataFrame([
    {"Model": "Tuned Gradient Boosting", "Accuracy": 0.926366, "Precision (macro)": 0.914152, "Recall (macro)": 0.919883, "F1 (macro)": 0.916913, "ROC-AUC": 0.972871},
    {"Model": "Gradient Boosting",       "Accuracy": 0.919240, "Precision (macro)": 0.906675, "Recall (macro)": 0.910851, "F1 (macro)": 0.908707, "ROC-AUC": 0.966850},
    {"Model": "SVM (RBF)",               "Accuracy": 0.907363, "Precision (macro)": 0.888049, "Recall (macro)": 0.918481, "F1 (macro)": 0.899112, "ROC-AUC": 0.968498},
    {"Model": "Random Forest",           "Accuracy": 0.910214, "Precision (macro)": 0.896427, "Recall (macro)": 0.900782, "F1 (macro)": 0.898541, "ROC-AUC": 0.958094},
    {"Model": "Logistic Regression",     "Accuracy": 0.831829, "Precision (macro)": 0.811552, "Recall (macro)": 0.844056, "F1 (macro)": 0.819935, "ROC-AUC": 0.916127},
])

CV_COMPARISON = pd.DataFrame([
    {"Model": "Gradient Boosting",   "Accuracy": 0.899359, "Precision (macro)": 0.901200, "Recall (macro)": 0.890404, "F1 (macro)": 0.887031, "ROC-AUC": 0.969086},
    {"Model": "Random Forest",       "Accuracy": 0.895225, "Precision (macro)": 0.895663, "Recall (macro)": 0.882841, "F1 (macro)": 0.880826, "ROC-AUC": 0.956882},
    {"Model": "SVM (RBF)",           "Accuracy": 0.816108, "Precision (macro)": 0.813298, "Recall (macro)": 0.841164, "F1 (macro)": 0.807254, "ROC-AUC": 0.916756},
    {"Model": "Logistic Regression", "Accuracy": 0.796721, "Precision (macro)": 0.793392, "Recall (macro)": 0.823744, "F1 (macro)": 0.787755, "ROC-AUC": 0.889092},
])

BEST_PARAMS = {
    "n_estimators": 500, "learning_rate": 0.1, "max_depth": 5,
    "min_samples_split": 2, "min_samples_leaf": 4, "subsample": 1.0, "max_features": "log2",
}

# ============================================================================
# APP
# ============================================================================
inject_css()
df = load_data()
model = load_model()
artifacts = build_pipeline_artifacts(len(df))

n_total = len(df)
n_confirmed = int((df['koi_disposition'] == 1).sum())
n_fp = int((df['koi_disposition'] == 0).sum())

# ---------------- HERO ----------------
st.markdown(textwrap.dedent(f"""
<div class="hero">
    <div class="eyebrow">Kepler Objects of Interest &middot; Binary Classification</div>
    <div class="hero-title">Exoplanet Candidate Classifier</div>
    <div class="hero-sub">
        Every row below is a signal the Kepler space telescope flagged as a possible planet —
        a periodic dip in a star's brightness. This dashboard explores that dataset and the models
        trained to separate genuine <b>confirmed exoplanets</b> from <b>false positives</b> using nothing
        but the shape of the transit and the properties of the host star.
    </div>
    <div class="stat-strip">
        <div class="stat-block"><div class="stat-value">{n_total:,}</div><div class="stat-label">Objects of Interest</div></div>
        <div class="stat-block"><div class="stat-value">{n_confirmed:,}</div><div class="stat-label">Confirmed Planets</div></div>
        <div class="stat-block"><div class="stat-value">{n_fp:,}</div><div class="stat-label">False Positives</div></div>
        <div class="stat-block"><div class="stat-value">13</div><div class="stat-label">Transit &amp; Stellar Features</div></div>
        <div class="stat-block"><div class="stat-value">92.6%</div><div class="stat-label">Best Model Accuracy</div></div>
    </div>
    {light_curve_svg()}
</div>
"""), unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Mission Overview", "Explore the Data", "Model Comparison", "Classify a Candidate"
])

# ============================================================================
# TAB 1 — MISSION OVERVIEW
# ============================================================================
with tab1:
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown(textwrap.dedent("""
        <div class="panel">
            <div class="panel-title">How Kepler finds a planet</div>
            <p style="color:var(--dim); line-height:1.7; font-size:0.93rem;">
            The Kepler telescope stared at roughly 150,000 stars for years, measuring brightness
            every 30 minutes. When a planet crosses in front of its star from our point of view — a
            <i>transit</i> — the star dims by a tiny, periodic amount. A repeating dip like that gets
            logged as a <b>Kepler Object of Interest (KOI)</b>.
            <br><br>
            Not every dip is a planet. Eclipsing binary stars, instrument noise, and background
            stars blending into the same pixel can all fake the same signature. Telling a real
            planet from an imitator is exactly the classification problem this dashboard is built
            around: <b>koi_disposition = 1</b> means the candidate was confirmed as a genuine planet,
            <b>koi_disposition = 0</b> means it was dispositioned as a false positive.
            </p>
        </div>
        """), unsafe_allow_html=True)

    with col2:
        fig = go.Figure(data=[go.Pie(
            labels=["Confirmed", "False Positive"],
            values=[n_confirmed, n_fp],
            hole=0.62,
            marker=dict(colors=["#4fd1ae", "#e0685a"]),
            textfont=dict(family="IBM Plex Mono", size=13, color="#f2f0e8"),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=280, showlegend=True,
                           legend=dict(orientation="h", y=-0.1),
                           annotations=[dict(text=f"{n_confirmed/n_total:.0%}<br><span style='font-size:10px'>confirmed</span>",
                                              x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="#4fd1ae"))])
        st.markdown('<div class="panel"><div class="panel-title">Class Balance</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Feature Glossary</div>', unsafe_allow_html=True)
    gcols = st.columns(3)
    for i, grp in enumerate(["signal", "planet", "star"]):
        with gcols[i]:
            st.markdown(f'<div class="group-label">{GROUP_LABELS[grp]}</div>', unsafe_allow_html=True)
            for feat, meta in FEATURE_META.items():
                if meta["group"] == grp:
                    unit = f" &nbsp;<span style='color:var(--dim); font-size:0.75rem;'>({meta['unit']})</span>" if meta['unit'] else ""
                    st.markdown(textwrap.dedent(f"""
                    <div style="margin-bottom:0.7rem;">
                        <span style="font-family:'IBM Plex Mono'; font-size:0.82rem; color:var(--gold);">{feat}</span>{unit}<br>
                        <span style="font-size:0.82rem; color:var(--dim);">{meta['help']}</span>
                    </div>
                    """), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 2 — EXPLORE THE DATA
# ============================================================================
with tab2:
    st.markdown(textwrap.dedent("""
    <div class="panel">
        <div class="panel-title">The classic exoplanet diagram: Radius vs. Orbital Period</div>
        <p style="color:var(--dim); font-size:0.88rem; margin-bottom:1rem;">
        This log-log view is the standard way exoplanet catalogs are visualised. Confirmed planets
        and false positives occupy noticeably different regions — that separability is what the
        models below are learning.
        </p>
    """), unsafe_allow_html=True)

    plot_df = df.copy()
    plot_df["Disposition"] = plot_df["koi_disposition"].map({1: "Confirmed", 0: "False Positive"})
    fig = px.scatter(
        plot_df, x="koi_period", y="koi_prad", color="Disposition",
        color_discrete_map={"Confirmed": "#4fd1ae", "False Positive": "#e0685a"},
        log_x=True, log_y=True, opacity=0.55,
        labels={"koi_period": "Orbital Period (days, log scale)", "koi_prad": "Planetary Radius (Earth radii, log scale)"},
        hover_data={"koi_depth": True, "koi_teq": True},
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=440, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel"><div class="panel-title">Distribution by Feature</div>', unsafe_allow_html=True)
        feat_choice = st.selectbox(
            "Feature", FEATURES,
            format_func=lambda f: f"{FEATURE_META[f]['label']} ({f})",
            key="dist_feat"
        )
        log_scale = st.checkbox("Log scale (recommended — most features are heavily skewed)", value=True)
        plot_vals = plot_df.copy()
        x_col = feat_choice
        if log_scale:
            plot_vals["_x"] = np.log1p(plot_vals[feat_choice])
            x_label = f"log(1 + {feat_choice})"
        else:
            plot_vals["_x"] = plot_vals[feat_choice]
            x_label = feat_choice
        fig2 = px.histogram(
            plot_vals, x="_x", color="Disposition", barmode="overlay", nbins=50,
            color_discrete_map={"Confirmed": "#4fd1ae", "False Positive": "#e0685a"},
            labels={"_x": x_label},
        )
        fig2.update_traces(opacity=0.6)
        fig2.update_layout(template=PLOTLY_TEMPLATE, height=340, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel"><div class="panel-title">Feature Correlation</div>', unsafe_allow_html=True)
        corr = df[FEATURES].corr()
        fig3 = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, "#e0685a"], [0.5, "#0f172c"], [1, "#4fd1ae"]],
            zmid=0, zmin=-1, zmax=1,
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
            colorbar=dict(thickness=12, tickfont=dict(size=9)),
        ))
        fig3.update_layout(template=PLOTLY_TEMPLATE, height=340,
                            xaxis=dict(tickfont=dict(size=8), tickangle=45),
                            yaxis=dict(tickfont=dict(size=8)))
        st.plotly_chart(fig3, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 3 — MODEL COMPARISON
# ============================================================================
with tab3:
    st.markdown(textwrap.dedent("""
    <div class="panel">
        <div class="panel-title">Held-out Test Set — 5 Candidate Pipelines</div>
        <p style="color:var(--dim); font-size:0.85rem; margin-bottom:1rem;">
        Macro-F1 and ROC-AUC are the metrics that matter most here — the classes are imbalanced
        (roughly 2:1 false positives to confirmed planets), so plain accuracy alone would flatter a
        model that just leans toward the majority class.
        </p>
    """), unsafe_allow_html=True)

    display_df = TEST_SET_COMPARISON.set_index("Model")
    st.dataframe(
        display_df.style.format("{:.3f}").background_gradient(cmap="Greens", vmin=0.79, vmax=0.98),
        width='stretch'
    )

    fig4 = px.bar(
        TEST_SET_COMPARISON.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_sequence=["#4fd1ae", "#e8a94c", "#7aa2f7", "#e0685a", "#c792ea"],
    )
    fig4.update_layout(template=PLOTLY_TEMPLATE, height=380, yaxis_range=[0.75, 1.0],
                        legend=dict(orientation="h", y=1.15), xaxis_tickangle=-15)
    st.plotly_chart(fig4, width='stretch', config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("5-fold cross-validation results (more robust than a single test split)"):
        st.dataframe(
            CV_COMPARISON.set_index("Model").style.format("{:.3f}").background_gradient(cmap="Greens", vmin=0.79, vmax=0.98),
            width='stretch'
        )
        st.caption("Cross-validation repeatedly re-splits the full dataset so every row is used for both training and evaluation, giving a less split-dependent estimate than a single train/test hold-out.")

    st.markdown("---")
    st.markdown(textwrap.dedent(f"""
    <div class="eyebrow" style="margin-top:0;">Winning Model</div>
    <div style="font-size:1.3rem; font-weight:600; margin-bottom:0.3rem;">Tuned Gradient Boosting</div>
    <div style="color:var(--dim); font-size:0.85rem; margin-bottom:1rem;">
        Selected via <code style="color:var(--gold);">RandomizedSearchCV</code> (5-fold, 50 candidate configurations) on macro-F1 and ROC-AUC.
    </div>
    """), unsafe_allow_html=True)

    pcols = st.columns(len(BEST_PARAMS))
    for c, (k, v) in zip(pcols, BEST_PARAMS.items()):
        with c:
            st.markdown(textwrap.dedent(f"""
            <div style="text-align:center;">
                <div style="font-family:'IBM Plex Mono'; font-size:1.1rem; color:var(--gold); font-weight:600;">{v}</div>
                <div style="font-family:'IBM Plex Mono'; font-size:0.62rem; color:var(--dim); text-transform:uppercase;">{k}</div>
            </div>
            """), unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.markdown('<div class="panel"><div class="panel-title">Confusion Matrix (test set, n=2105)</div>', unsafe_allow_html=True)
        cm = confusion_matrix(artifacts["y_test"], artifacts["y_pred"])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        st.markdown(textwrap.dedent(f"""
        <div class="cm-grid">
            <div></div>
            <div class="cm-axis">Pred: False Pos.</div>
            <div class="cm-axis">Pred: Confirmed</div>
            <div class="cm-axis">Actual:<br>False Pos.</div>
            <div class="cm-cell cm-correct"><div class="cm-num">{tn}</div><div class="cm-sub">True Negative</div></div>
            <div class="cm-cell cm-wrong"><div class="cm-num">{fp}</div><div class="cm-sub">False Positive</div></div>
            <div class="cm-axis">Actual:<br>Confirmed</div>
            <div class="cm-cell cm-wrong"><div class="cm-num">{fn}</div><div class="cm-sub">False Negative</div></div>
            <div class="cm-cell cm-correct"><div class="cm-num">{tp}</div><div class="cm-sub">True Positive</div></div>
        </div>
        <p style="color:var(--dim); font-size:0.8rem; margin-top:0.9rem;">
        In this domain a missed planet (false negative) is generally costlier than a false alarm —
        a flagged false positive just wastes follow-up telescope time, but a missed real planet is
        a lost discovery.
        </p>
        """), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel"><div class="panel-title">Feature Importance</div>', unsafe_allow_html=True)
        gb_model = model.named_steps.get("model", model)
        importances = pd.Series(gb_model.feature_importances_, index=FEATURES)
        importances = importances.sort_values(ascending=True)
        labels = [FEATURE_META[f]['label'] for f in importances.index]
        fig5 = go.Figure(go.Bar(
            x=importances.values, y=labels, orientation="h",
            marker=dict(color=importances.values, colorscale=[[0, "#e8a94c"], [1, "#4fd1ae"]]),
        ))
        fig5.update_layout(template=PLOTLY_TEMPLATE, height=420, xaxis_title="Importance")
        st.plotly_chart(fig5, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("ROC and Precision–Recall curves"):
        c1, c2 = st.columns(2)
        fpr, tpr, _ = roc_curve(artifacts["y_test"], artifacts["y_proba"])
        roc_auc = auc(fpr, tpr)
        with c1:
            figroc = go.Figure()
            figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#4fd1ae", width=2.5),
                                         name=f"ROC (AUC = {roc_auc:.3f})"))
            figroc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#4a5568", dash="dash"), name="Chance"))
            figroc.update_layout(template=PLOTLY_TEMPLATE, height=320, title="ROC Curve",
                                  xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                  legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(figroc, width='stretch', config={"displayModeBar": False})
        with c2:
            prec, rec, _ = precision_recall_curve(artifacts["y_test"], artifacts["y_proba"])
            figpr = go.Figure()
            figpr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", line=dict(color="#e8a94c", width=2.5), name="Precision-Recall"))
            figpr.update_layout(template=PLOTLY_TEMPLATE, height=320, title="Precision–Recall Curve",
                                 xaxis_title="Recall", yaxis_title="Precision", legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(figpr, width='stretch', config={"displayModeBar": False})

# ============================================================================
# TAB 4 — CLASSIFY A CANDIDATE
# ============================================================================
with tab4:
    st.markdown(textwrap.dedent("""
    <div class="panel">
        <div class="panel-title">Run the classifier on a candidate</div>
        <p style="color:var(--dim); font-size:0.88rem;">
        Enter transit and stellar measurements below — or load a real example from the held-out
        test set — and the tuned Gradient Boosting model will classify it as a confirmed planet
        or a false positive.
        </p>
    </div>
    """), unsafe_allow_html=True)

    bcol1, bcol2, bcol3, _ = st.columns([1.3, 1.3, 1, 2])
    X_test_raw = artifacts["X_test_raw"]
    y_test = artifacts["y_test"]

    def load_example(label_value):
        idx_pool = y_test[y_test == label_value].index
        chosen = np.random.choice(idx_pool)
        row = X_test_raw.loc[chosen]
        for f in FEATURES:
            st.session_state[f"inp_{f}"] = float(row[f])

    with bcol1:
        if st.button("Load confirmed example", width='stretch'):
            load_example(1)
    with bcol2:
        if st.button("Load false-positive example", width='stretch'):
            load_example(0)
    with bcol3:
        if st.button("Reset", width='stretch'):
            for f in FEATURES:
                st.session_state.pop(f"inp_{f}", None)

    st.markdown("<br>", unsafe_allow_html=True)

    raw_input = {}
    fcols = st.columns(3)
    for i, grp in enumerate(["signal", "planet", "star"]):
        with fcols[i]:
            st.markdown(f'<div class="group-label">{GROUP_LABELS[grp]}</div>', unsafe_allow_html=True)
            for feat, meta in FEATURE_META.items():
                if meta["group"] != grp:
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
    run = st.button("Run Classification", type="primary", width='stretch')

    if run:
        scaler = artifacts["scaler"]
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
        figgauge.update_layout(template=PLOTLY_TEMPLATE, height=110, xaxis_range=[0, 1],
                                margin=dict(t=10, b=25, l=10, r=10), showlegend=False)
        st.plotly_chart(figgauge, width='stretch', config={"displayModeBar": False})

st.markdown(textwrap.dedent("""
<div style="text-align:center; color:var(--dim); font-family:'IBM Plex Mono'; font-size:0.7rem; margin-top:2.5rem; padding-top:1.2rem; border-top:1px solid var(--grid);">
KOI CUMULATIVE TABLE &middot; TUNED GRADIENT BOOSTING CLASSIFIER &middot; BUILT WITH STREAMLIT
</div>
"""), unsafe_allow_html=True)