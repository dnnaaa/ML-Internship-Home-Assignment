# dashboard/dashboard.py
import streamlit as st
from dashboard.components.eda import display_eda_section
from dashboard.components.training import training_section
from dashboard.components.inference import inference_section
def main():
    st.title("Resume Classification Dashboard")
    st.sidebar.title("Dashboard Modes")

    sidebar_options = st.sidebar.selectbox("Options", ("EDA", "Training", "Inference"))

    if sidebar_options == "EDA":
        display_eda_section()
    elif sidebar_options == "Training":
        training_section()
    elif sidebar_options == "Inference":  
        inference_section()  
    
if __name__ == "__main__":
    main()
