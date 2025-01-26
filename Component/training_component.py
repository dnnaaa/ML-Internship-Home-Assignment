# components/training_component.py
import streamlit as st
from PIL import Image
from .base_component import DashboardComponent
from utils.helpers import display_metrics
from data_ml_assignment.training.train_pipeline import TrainingPipeline
from data_ml_assignment.constants import CM_PLOT_PATH

#Class used inside the dashboard.py
class TrainingComponent(DashboardComponent):
    def render(self):
        st.header("Pipeline Training")
        st.info(
            "Before you proceed to training your pipeline, ensure your "
            "training pipeline code is set properly."
        )

        name = st.text_input("Pipeline name", placeholder="Naive Bayes")
        serialize = st.checkbox("Save pipeline")
        train = st.button("Train pipeline")

        if train:
            with st.spinner("Training pipeline, please wait..."):
                try:
                    tp = TrainingPipeline()
                    tp.train(serialize=serialize, model_name=name)
                    accuracy, f1 = tp.get_model_perfomance()
                    display_metrics(accuracy, f1)
                    tp.render_confusion_matrix()
                except Exception as e:
                    st.error("Failed to train the pipeline!")
                    st.exception(e)
