import streamlit as st
import requests
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from data_ml_assignment.constants import LABELS_MAP, SAMPLES_PATH
from data_ml_assignment.constants import SQLITE_DB_URI
from data_ml_assignment.constants import INFERENCE_ENDPOINT

# SQLite Database Setup
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    resume_text = Column(String)
    predicted_label = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create SQLite database and session
engine = create_engine(SQLITE_DB_URI, echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

def render_inference():
    st.header("Resume Inference")
    st.info(
        "This section simplifies the inference process. "
        "Choose a test resume and observe the label that your trained pipeline will predict."
    )

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

                result = requests.post(INFERENCE_ENDPOINT, json={"text": sample_text})

                result.raise_for_status()  # Raise an exception for HTTP errors

                label = LABELS_MAP.get(int(float(result.text)))
                st.success("Done!")
                st.metric(label="Status", value=f"Resume label: {label}")

                prediction = Prediction(
                    resume_text=sample_text,
                    predicted_label=label,
                )
                session.add(prediction)
                session.commit()

            except requests.exceptions.RequestException as e:
                st.error("Failed to connect to the Inference API. Please ensure the API server is running.")
                st.exception(e)
            except FileNotFoundError:
                st.error("Resume sample not found. Please check the sample file.")
            except Exception as e:
                st.error("An error occurred during inference.")
                st.exception(e)

    # Display Prediction History
    st.subheader("Prediction History")
    predictions = session.query(Prediction).order_by(Prediction.timestamp.desc()).all()
    if predictions:
        prediction_data = [
            {
                "Resume Text": p.resume_text[:100] + "...",  # Display first 100 characters
                "Predicted Label": p.predicted_label,
                "Timestamp": p.timestamp,
            }
            for p in predictions
        ]
        st.table(prediction_data)
    else:
        st.info("No predictions have been made yet.")