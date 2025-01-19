# import time
# import streamlit as st
# from PIL import Image
# import requests

# from data_ml_assignment.training.train_pipeline import TrainingPipeline
# from data_ml_assignment.constants import CM_PLOT_PATH, LABELS_MAP, SAMPLES_PATH


# st.title("Resume Classification Dashboard")
# st.sidebar.title("Dashboard Modes")

# sidebar_options = st.sidebar.selectbox("Options", ("EDA", "Training", "Inference"))

# if sidebar_options == "EDA":
#     st.header("Exploratory Data Analysis")
#     st.info(
#         "In this section, you are invited to create insightful graphs "
#         "about the resume dataset that you were provided."
#     )
# elif sidebar_options == "Training":
#     st.header("Pipeline Training")
#     st.info(
#         "Before you proceed to training your pipeline. Make sure you "
#         "have checked your training pipeline code and that it is set properly."
#     )

#     name = st.text_input("Pipeline name", placeholder="Naive Bayes")
#     serialize = st.checkbox("Save pipeline")
#     train = st.button("Train pipeline")

#     if train:
#         with st.spinner("Training pipeline, please wait..."):
#             try:
#                 tp = TrainingPipeline()
#                 tp.train(serialize=serialize, model_name=name)
#                 tp.render_confusion_matrix()
#                 accuracy, f1 = tp.get_model_perfomance()
#                 col1, col2 = st.columns(2)

#                 col1.metric(label="Accuracy score", value=str(round(accuracy, 4)))
#                 col2.metric(label="F1 score", value=str(round(f1, 4)))

#                 st.image(Image.open(CM_PLOT_PATH), width=850)
#             except Exception as e:
#                 st.error("Failed to train the pipeline!")
#                 st.exception(e)

# else:
#     st.header("Resume Inference")
#     st.info(
#         "This section simplifies the inference process. "
#         "Choose a test resume and observe the label that your trained pipeline will predict."
#     )

#     sample = st.selectbox(
#         "Resume samples for inference",
#         tuple(LABELS_MAP.values()),
#         index=None,
#         placeholder="Select a resume sample",
#     )
#     infer = st.button("Run Inference")

#     if infer:
#         with st.spinner("Running inference..."):
#             try:
#                 sample_file = "_".join(sample.upper().split()) + ".txt"
#                 with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
#                     sample_text = file.read()

#                 result = requests.post(
#                     "http://localhost:9000/api/inference", json={"text": sample_text}
#                 )
#                 st.success("Done!")
#                 label = LABELS_MAP.get(int(float(result.text)))
#                 st.metric(label="Status", value=f"Resume label: {label}")
#             except Exception as e:
#                 st.error("Failed to call Inference API!")
#                 st.exception(e)

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