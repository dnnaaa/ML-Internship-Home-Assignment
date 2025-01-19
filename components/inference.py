import time
import streamlit as st
import requests
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from data_ml_assignment.constants import LABELS_MAP, SAMPLES_PATH

# Define the SQLite database URL
DATABASE_URL = "sqlite:///resume_predictions.db"

# Create a SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Define the base class for declarative models
Base = declarative_base()


# Define the table to store inference results
class PredictionResult(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    resume_text = Column(String, nullable=False)
    predicted_label = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)


# Create the table in the database
Base.metadata.create_all(engine)

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()


def save_prediction(resume_text, predicted_label):
    """
    Save the inference result into the SQLite database.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    prediction = PredictionResult(
        resume_text=resume_text,
        predicted_label=predicted_label,
        timestamp=timestamp
    )
    session.add(prediction)
    session.commit()


def show_inference():
    """
    Display the Resume Inference section in the Streamlit app.
    """
    st.header("Resume Inference")
    st.info(
        "This section allows you to test the trained machine learning model on sample resumes. "
        "Select a resume sample from the dropdown menu, and the model will predict its corresponding job label. "
        "The predicted label will be displayed below, and all predictions are saved to a database for future reference."
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

                if not sample_text:
                    st.error("Resume text is empty. Please select a valid sample.")
                    return

                # Call the API
                result = requests.post(
                    "http://localhost:9000/api/inference", json={"text": sample_text}
                )
                st.success("Done!")

                # Parse the API response as JSON
                response_data = result.json()

                # Handle the response
                if isinstance(response_data, dict) and "error" in response_data:
                    # If the response contains an error, display it
                    st.error(f"API Error: {response_data['error']}")
                    return

                # Get the predicted label
                predicted_label = int(float(response_data))
                label = LABELS_MAP.get(predicted_label)

                # Save the prediction result
                save_prediction(sample_text, label)

                # Display the predicted label
                st.metric(label="Status", value=f"Resume label: {label}")

                # Query the SQLite table and display the prediction history
                st.subheader("Prediction History")
                predictions = session.query(PredictionResult).all()
                if predictions:
                    prediction_data = [
                        {
                            "Resume Text": p.resume_text[:50] + "...",  # Display first 50 characters
                            "Predicted Label": p.predicted_label,
                            "Timestamp": p.timestamp
                        }
                        for p in predictions
                    ]
                    st.table(prediction_data)  # Display the predictions as a table
                else:
                    st.write("No predictions found in the database.")
            except Exception as e:
                st.error("Failed to call Inference API!")
                st.exception(e)
