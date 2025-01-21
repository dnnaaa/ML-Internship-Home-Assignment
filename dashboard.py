import time
import streamlit as st
from PIL import Image
import requests

from eda import EDA
from train_pipeline import TrainingPipeline
from data_ml_assignment.constants import CM_PLOT_PATH, LABELS_MAP, SAMPLES_PATH
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define the SQLite database URL
DATABASE_URL = "sqlite:///predictions.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the Prediction model
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    resume_name = Column(String, index=True)
    prediction_result = Column(String)
    confidence_score = Column(Float)

Base.metadata.create_all(bind=engine)

st.title("Resume Classification Dashboard")
st.sidebar.title("Dashboard Modes")

sidebar_options = st.sidebar.selectbox("Options", ("EDA", "Training", "Inference"))

if sidebar_options == "EDA":
    st.header("Exploratory Data Analysis")
    eda = EDA()
    eda.show_summary()
    eda.show_missing_values()
    eda.show_category_distribution()
    eda.show_word_cloud()
    
elif sidebar_options == "Training":
    st.header("Pipeline Training")

    name = st.text_input("Pipeline name", placeholder="xgb_pipline")
    serialize = st.checkbox("Save pipeline")
    train = st.button("Train pipeline")

    if train:
        with st.spinner("Training pipeline, please wait..."):
            try:
                tp = TrainingPipeline()
                tp.train(serialize=serialize, model_name=name)
                tp.render_confusion_matrix()
                accuracy, f1 = tp.get_model_perfomance()
                col1, col2 = st.columns(2)

                col1.metric(label="Accuracy score", value=str(round(accuracy, 4)))
                col2.metric(label="F1 score", value=str(round(f1, 4)))

                st.image(Image.open(CM_PLOT_PATH), width=850)
            except Exception as e:
                st.error("Failed to train the pipeline!")
                st.exception(e)

else:
    st.header("Resume Inference")

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
                sample_path = SAMPLES_PATH / sample_file
                print(f"Looking for file: {sample_path}")
                with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                    sample_text = file.read()

                result = requests.post(
                    "http://localhost:9000/api/inference", json={"text": sample_text}
                )
                st.success("Done!")
                label = LABELS_MAP.get(int(float(result.text)))

                db = SessionLocal()
                db_prediction = Prediction(
                    resume_name=sample,
                    prediction_result=label,
                    confidence_score=float(result.text)  # the API returns confidence
                )
                db.add(db_prediction)
                db.commit()
                db.refresh(db_prediction)

                st.metric(label="Status", value=f"Resume label: {label}")

                st.subheader("All Predictions")
                predictions = db.query(Prediction).all()
                for pred in predictions:
                    st.write(f"Resume: {pred.resume_name}, Prediction: {pred.prediction_result}, Confidence: {pred.confidence_score}")
                    
            except Exception as e:
                st.error("Failed to call Inference API!")
                st.exception(e)
