import time
import streamlit as st
from PIL import Image
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



from data_ml_assignment.components.eda import show_eda
from data_ml_assignment.training.train_pipeline import TrainingPipeline
from data_ml_assignment.constants import CM_PLOT_PATH, LABELS_MAP, SAMPLES_PATH, RAW_DATASET_PATH, DATA_PATH, API_BASE_URL, INFERENCE_API_URL, PREDICTIONS_BASE_URL



st.title("Resume Classification Dashboard")
st.sidebar.title("Dashboard Modes")


# API_URL = "http://localhost:9002/api"

def display_predictions():
    try:
        response = requests.get(PREDICTIONS_BASE_URL)
        response.raise_for_status()
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df)
        else:
            st.write("No predictions found.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch predictions: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")



sidebar_options = st.sidebar.selectbox("Options", ("EDA", "Training", "Inference"))

if sidebar_options == "EDA":
    st.header("Exploratory Data Analysis")
    st.info("In this section, you are invited to create insightful graphs about the resume dataset that you were provided.")

    dataset = pd.read_csv(RAW_DATASET_PATH)
    show_eda(dataset)

    
elif sidebar_options == "Training":
    st.header("Pipeline Training")
    st.info(
        "Before you proceed to training your pipeline. Make sure you "
        "have checked your training pipeline code and that it is set properly."
    )

    name = st.text_input("Pipeline name", placeholder="Naive Bayes")
    serialize = st.checkbox("Save pipeline")
    train = st.button("Train pipeline")

    if train:
        with st.spinner("Training pipeline, please wait..."):
            try:
                tp = TrainingPipeline()
                tp.train(serialize=serialize, model_name=name)
                tp.render_confusion_matrix()
                train_accuracy,test_accuracy, f1 = tp.get_model_perfomance()
                
                
                col1, col2 = st.columns(2)

                col1.metric(label="Accuracy Train score", value=str(round(train_accuracy, 4)))
                col1.metric(label="Accuracy Test score", value=str(round(test_accuracy, 4)))
                col2.metric(label="F1 score", value=str(round(f1, 4)))

                st.image(Image.open(CM_PLOT_PATH), width=850)
                

            except Exception as e:
                st.error("Failed to train the pipeline!")
                st.exception(e)

else:
    st.header("Resume Inference")
    st.info("This section simplifies the inference process. Choose a test resume and observe the label that your trained pipeline will predict.")

    sample = st.selectbox(
        "Resume samples for inference",
        tuple(LABELS_MAP.values()),
        index=None,
        placeholder="Select a resume sample",
    )
    infer = st.button("Run Inference")

    if infer:
        with st.spinner("Running inference..."):
            try:
                sample_file = "_".join(sample.upper().split()) + ".txt"
                with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                    sample_text = file.read()

                result = requests.post(
                    INFERENCE_API_URL, json={"text": sample_text}
                )
                result.raise_for_status()
                st.success("Done!")
                label = LABELS_MAP.get(int(float(result.text)))
                st.metric(label="Status", value=f"Resume label: {label}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error in API request: {e}")
            except Exception as e:
                st.error("Failed to call Inference API!")
                st.exception(e)

    if st.button("View Saved Predictions"):
        display_predictions()

