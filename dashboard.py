import streamlit as st
from components.eda import show_eda
from components.training import show_training
from components.inference import show_inference

st.title("Resume Classification Dashboard")
st.sidebar.title("Dashboard Modes")

sidebar_options = st.sidebar.selectbox("Options", ("EDA", "Training", "Inference"))

if sidebar_options == "EDA":
    show_eda()
elif sidebar_options == "Training":
    show_training()
else:
    show_inference()
