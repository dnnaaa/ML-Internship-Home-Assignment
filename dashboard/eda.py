import streamlit as st
from data_ml_assignment.constants import CM_PLOT_PATH, LABELS_MAP, SAMPLES_PATH

class EDACombonent:
    def render(self):
        st.header("Exploratory Data Analysis")
        st.info("In this section, you are invited to create insightful graphs about the resume dataset that you were provided.")
        # EDA-specific logic goes here