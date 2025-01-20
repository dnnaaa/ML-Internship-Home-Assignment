import streamlit as st
from eda import render_eda
from training import render_training
from inference import render_inference

def main():
    st.title("Resume Classification Dashboard")
    st.sidebar.title("Dashboard Modes")

    # Sidebar navigation
    sidebar_options = st.sidebar.selectbox("Options", ("EDA", "Training", "Inference"))

    if sidebar_options == "EDA":
        render_eda()
    elif sidebar_options == "Training":
        render_training()
    elif sidebar_options == "Inference":
        render_inference()

if __name__ == "__main__":
    main()