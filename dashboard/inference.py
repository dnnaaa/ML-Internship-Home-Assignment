import streamlit as st
import requests
from data_ml_assignment.database import save_prediction_to_db, get_predictions
from data_ml_assignment.constants import LABELS_MAP, SAMPLES_PATH

def run_inference(sample_text: str) -> str:
    """Make a prediction for the given sample text."""
    try:
        result = requests.post(
            "http://localhost:9000/api/inference", json={"text": sample_text}
        )
        if result.status_code == 200:
            # Return the label from the model
            return LABELS_MAP.get(int(float(result.text)))
        else:
            st.error("API returned an error. Check logs.")
            return None
    except Exception as e:
        st.error("Failed to call Inference API!")
        st.exception(e)
        return None

def display_inference_section() -> None:
    """Display the inference section for resume classification."""
    st.header("Resume Inference")
    st.info("Choose a resume sample and observe the predicted label from your trained pipeline.")

    # Dropdown for selecting a sample file
    sample = st.selectbox(
        "Resume samples for inference",
        tuple(LABELS_MAP.values()),
        index=None,
        placeholder="Select a resume sample",
    )
    
    # Button to trigger the inference
    infer = st.button("Run Inference")

    if infer:
        with st.spinner("Running inference..."):
            sample_file = "_".join(sample.upper().split()) + ".txt"
            try:
                with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                    sample_text = file.read()

                label = run_inference(sample_text)
                if label:
                    # Save the result to the database
                    save_prediction_to_db(sample_file, label)  # Passing both parameters
                    
                    st.success("Done!")
                    st.metric(label="Status", value=f"Resume label: {label}")

                    # Display all predictions from the database
                    predictions = get_predictions()
                    st.write("Prediction History:")
                    for pred in predictions:
                        st.write(f"Sample: {pred.sample_name}, Label: {pred.prediction_label}")
            except Exception as e:
                st.error("Failed to read sample file!")
                st.exception(e)
