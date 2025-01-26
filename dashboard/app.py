import streamlit as st
from dashboard.eda import EDACombonent
from dashboard.training import TrainingComponent
from dashboard.inference import InferenceComponent

class DashboardComposer:
    def __init__(self):
        self.eda_component = EDACombonent()
        self.training_component = TrainingComponent()
        self.inference_component = InferenceComponent()

    def run(self):
        sidebar_options = st.sidebar.selectbox("Options", ("EDA", "Training", "Inference"))

        if sidebar_options == "EDA":
            self.eda_component.render()
        elif sidebar_options == "Training":
            self.training_component.render()
        else:
            self.inference_component.render()

if __name__ == "__main__":
    st.set_page_config(page_title="Resume Classification Dashboard")
    dashboard = DashboardComposer()
    dashboard.run()