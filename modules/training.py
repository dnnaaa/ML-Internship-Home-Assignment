import streamlit as st
from PIL import Image
from data_ml_assignment.training.train_pipeline import TrainingPipeline
from data_ml_assignment.constants import CM_PLOT_PATH

def render_training_section():
    "In this section, we can train a pipeline and evaluate its performance."
    return 0