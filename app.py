import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


def home():
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



def dataset():
    st.title("🛰 Dataset Explorer")
    st.caption("Explore the NASA Kepler Exoplanet dataset interactively.")

    st.divider()

    # =============================
    # Dataset Metrics
    # =============================

    rows, cols = df.shape
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    memory = df.memory_usage(deep=True).sum() / 1024**2

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{rows:,}")
    c2.metric("Columns", cols)
    c3.metric("Missing Values", missing)
    c4.metric("Memory Usage", f"{memory:.2f} MB")

    st.divider()

    # =============================
    # Preview
    # =============================

    st.subheader("Dataset Preview")
    n_rows = st.slider(
        "Rows to display",
        5,
        100,
        10
    )

    st.dataframe(
        df.head(n_rows),
        use_container_width=True,
        height=400
    )

    st.divider()

    # =============================
    # Column Explorer
    # =============================

    st.subheader("Column Explorer")
    column = st.selectbox(
        "Choose a feature",
        df.columns
    )

    left, right = st.columns(2)

    with left:
        st.write("Datatype")
        st.code(df[column].dtype)

        st.write("Missing")
        st.code(df[column].isnull().sum())

        st.write("Unique Values")
        st.code(df[column].nunique())

    with right:
        st.write("Statistics")
        st.dataframe(
            df[column].describe().to_frame(),
            use_container_width=True
        )

    st.divider()

    # =============================
    # Data Types
    # =============================

    st.subheader("Feature Types")

    dtype_df = pd.DataFrame({
        "Feature": df.columns,
        "Type": df.dtypes.astype(str)
    })

    st.dataframe(
        dtype_df,
        use_container_width=True
    )

    st.divider()

    # =============================
    # Missing Values
    # =============================

    st.subheader("Missing Values")

    missing_df = (
        df.isnull()
          .sum()
          .reset_index()
    )

    missing_df.columns = [
        "Feature",
        "Missing"
    ]

    st.dataframe(
        missing_df.sort_values(
            "Missing",
            ascending=False
        ),
        use_container_width=True
    )

    st.divider()

    # =============================
    # Summary Statistics
    # =============================

    st.subheader("Summary Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.download_button(
        "⬇ Download Dataset",
        df.to_csv(index=False),
        file_name="kepler_dataset.csv",
        mime="text/csv"
    )



def eda():
    st.title("📊 Exploratory Data Analysis")
    st.caption(
        "Interactively explore the Kepler Exoplanet dataset."
    )

    st.markdown("""
        <div class="hero">
        <h1>📊 Exploratory Data Analysis</h1>
        <p>
        Investigate distributions, relationships, correlations,
        and hidden patterns within the NASA Kepler dataset.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()


    # -------------------------------
    # Distribution Analysis
    # -------------------------------

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Create the tabs here
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distribution",
        "☁ Relationships",
        "🔥 Correlations",
        "🛰 Missing Data",
        "🎯 Target Analysis"
    ])


    with tab1:
        feature = st.selectbox(
            "Select Numerical Feature",
            numeric_cols
        )


        st.markdown("""
            <div class="section-header">
            <h2>📈 Distribution Analysis</h2>
            <p>Explore the statistical distribution of every numerical feature.</p>
            </div>
            """, unsafe_allow_html=True
        )


        col1, col2, col3 = st.columns(3)

        # Histogram
        with col1:
            fig = px.histogram(df, x=feature, nbins=40, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)


            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{df[feature].mean():.2f}")
            c2.metric("Median", f"{df[feature].median():.2f}")
            c3.metric("Std", f"{df[feature].std():.2f}")
            c4.metric("Skew", f"{df[feature].skew():.2f}")

        
        # Box Plot
        with col2:
            fig = px.box(df, y=feature, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

            Q1 = df[feature].quantile(.25)
            Q3 = df[feature].quantile(.75)
            IQR = Q3 - Q1

            c1, c2, c3 = st.columns(3)
            c1.metric("Q1", f"{Q1:.2f}")
            c2.metric("Q3", f"{Q3:.2f}")
            c3.metric("IQR", f"{IQR:.2f}")

        
        # Violin Plot
        with col3:
            fig = px.violin(df, y=feature, box=True, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Min", f"{df[feature].min():.2f}")
            c2.metric("Max", f"{df[feature].max():.2f}")
            c3.metric("Kurtosis", f"{df[feature].kurt():.2f}")


    with tab2:
        st.subheader("☁ Feature Relationships")
        col1, col2 = st.columns(2)
        x_axis = col1.selectbox("X-axis", numeric_cols, index=0)
        y_axis = col2.selectbox("Y-axis", numeric_cols, index=1)

        # Scatter Plot
        color_feature = st.selectbox("Color by", ["None"] + numeric_cols)
        if color_feature == "None":
            fig = px.scatter(df, x=x_axis, y=y_axis, opacity=0.7, title=f"{x_axis} vs {y_axis}")
        else:
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_feature)

        st.plotly_chart(fig, use_container_width=True)


    with tab3:
        # Correlation Heatmap
        st.subheader("🔥 Correlation Heatmap")
        corr = df[numeric_cols].corr()

        fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=1,
                zmax=1,
                aspect="auto"
            )
        
        fig.update_layout(
            template="plotly_dark",
            font_color="white",
            coloraxis_colorbar=dict(title='Correlation')
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        corr_values = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack()
        c1.metric( "Strongest +ve", f"{corr_values.max():.2f}")
        c2.metric("Strongest -ve", f"{corr_values.min():.2f}")
        c3.metric("Mean |Correlation|", f"{corr_values.abs().mean():.2f}")
        c4.metric("Features", len(numeric_cols))

        


    with tab4:
        # Missing Values
        st.subheader("🛰 Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing):
            fig = px.bar(
                x=missing.index,
                y=missing.values,
                labels={
                    "x":"Feature",
                    "y":"Missing Values"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.success("No missing values found.")


    with tab5:
        # Target Distribution
        st.subheader("🪐 Target Distribution")
        fig = px.pie(
            df,
            names="koi_disposition",
            title="Class Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


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


# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Datasets/cumulative_cleaned.csv")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

df = load_data()
model = load_model()





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


if selected == 'Home':
    home()


elif selected == 'Dataset':
    dataset()


elif selected == 'EDA':
    eda()


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