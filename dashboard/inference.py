import streamlit as st
import requests
from pathlib import Path
import pandas as pd
from data_ml_assignment.database import save_prediction_to_db, get_predictions
from data_ml_assignment.constants import LABELS_MAP, SAMPLES_PATH, MODELS_PATH

def run_inference(text: str) -> str:
    """Get prediction from API endpoint"""
    try:
        response = requests.post(
            "http://localhost:9000/api/inference",
            json={"text": text},
            timeout=10
        )
        return LABELS_MAP.get(int(response.text)) if response.ok else "Error"
    except Exception as e:
        return f"API Error: {str(e)}"

def display_inference_section():
    st.header("🔍 Prediction Interface")
    
    # Check for trained models
    if not list(Path(MODELS_PATH).glob("*.joblib")):
        st.error("No trained models found. Please train a model first.")
        return
    
    # Sample selection
    sample = st.selectbox(
        "Choose resume sample",
        options=list(LABELS_MAP.values()),
        index=0,
        help="Pre-loaded resume samples for quick testing"
    )
    
    if st.button("Analyze Resume", type="primary"):
        with st.status("Processing...", expanded=True) as status:
            try:
                # File handling
                sample_file = SAMPLES_PATH / f"{'_'.join(sample.upper().split())}.txt"
                if not sample_file.exists():
                    raise FileNotFoundError(f"Sample {sample} not found")
                
                # Read sample text
                with open(sample_file, "r", encoding="utf-8") as f:
                    sample_text = f.read()
                
                # Run inference
                status.update(label="Running model prediction...")
                prediction = run_inference(sample_text)
                
                # Save to database
                status.update(label="Saving results...")
                save_prediction_to_db(sample_file.name, prediction)
                
                # Display results
                status.update(label="Analysis Complete", state="complete")
                st.success(f"Predicted Label: **{prediction}**")
                
                # Show history
                st.subheader("📜 Prediction History")
                history = get_predictions()
                if history:
                    st.dataframe(
                        pd.DataFrame([
                            {"Sample": h.sample_name, "Prediction": h.prediction_label}
                            for h in history
                        ]),
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("No previous predictions found")
                    
            except Exception as e:
                status.update(label="Analysis Failed", state="error")
                st.error(f"Error: {str(e)}")
                st.exception(e) 