import pandas as pd
import joblib
import streamlit as st


@st.cache_data
def load_data():
    """
    Load the cleaned Kepler dataset.
    """
    return pd.read_csv("Datasets/cumulative_cleaned.csv")


@st.cache_resource
def load_model():
    """
    Load the trained machine learning model.
    """
    return joblib.load("best_model.joblib")