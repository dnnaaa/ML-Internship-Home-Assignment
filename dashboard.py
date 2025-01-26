import streamlit as st
import streamlit as st
from modules.eda import render_eda_section
from modules.training import render_training_section
from modules.inference import render_inference_section


st.title("Resume Classification Dashboard")
st.sidebar.title("Dashboard Modes")

sidebar_options = st.sidebar.selectbox("Options", ("EDA", "Training", "Inference"))

if sidebar_options == "EDA":
    render_eda_section("data/raw/resume.csv")
elif sidebar_options == "Training":
    render_training_section("data/raw/resume.csv")
else:
    render_inference_section()
