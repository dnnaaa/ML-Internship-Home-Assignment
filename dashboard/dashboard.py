import time
import streamlit as st
from PIL import Image
import requests

from data_ml_assignment.training.train_pipeline import TrainingPipeline
from data_ml_assignment.constants import CM_PLOT_PATH, LABELS_MAP, SAMPLES_PATH
from eda import display_eda_section
from training import display_training_section
from inference import display_inference_section

# Create the database and the 'predictions' table
from data_ml_assignment.database import create_database

create_database()


st.title("Resume Classification Dashboard")
st.sidebar.title("Dashboard Modes")

sidebar_options = st.sidebar.selectbox("Options", ("EDA", "Training", "Inference"))

if sidebar_options == "EDA":
    display_eda_section()

elif sidebar_options == "Training":
    display_training_section()

else:
    display_inference_section()
