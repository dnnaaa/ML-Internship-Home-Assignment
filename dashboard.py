import streamlit as st

from data_ml_assignment.dashboard.components.eda import EDAComponent
from data_ml_assignment.dashboard.components.training import TrainingComponent
from data_ml_assignment.dashboard.components.inference import InferenceComponent

def main():
    st.title("Resume Classification Dashboard")
    st.sidebar.title("Dashboard Modes")

    # Initialize components
    components = {
        "EDA": EDAComponent(),
        "Training": TrainingComponent(),
        "Inference": InferenceComponent()
    }

    # Render selected component
    selected = st.sidebar.selectbox("Options", tuple(components.keys()))
    components[selected].render()

if __name__ == "__main__":
    main()
