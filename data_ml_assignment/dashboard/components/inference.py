import streamlit as st
import requests
import pandas as pd
from datetime import datetime

from data_ml_assignment.dashboard.base import DashboardComponent
from data_ml_assignment.constants import LABELS_MAP, SAMPLES_PATH
from data_ml_assignment.models.prediction import Session, Prediction

class InferenceComponent(DashboardComponent):
    def __init__(self, api_url: str = "http://localhost:9000/api/inference"):
        self.api_url = api_url
        self.session = Session()

    def render(self):
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
            self._handle_inference(sample)
            
        # Afficher l'historique des prédictions
        self._display_prediction_history()

    def _handle_inference(self, sample: str):
        with st.spinner("Running inference..."):
            try:
                sample_text = self._read_sample_file(sample)
                result = self._call_inference_api(sample_text)
                label = LABELS_MAP.get(int(float(result.text)))
                
                # Sauvegarder la prédiction
                self._save_prediction(sample, label)
                
                st.success("Done!")
                st.metric(label="Status", value=f"Resume label: {label}")
            except Exception as e:
                self.show_error("Failed to call Inference API!", e)

    def _save_prediction(self, resume_type: str, predicted_label: str):
        """Save prediction to SQLite database."""
        prediction = Prediction(
            resume_type=resume_type,
            predicted_label=predicted_label
        )
        self.session.add(prediction)
        self.session.commit()

    def _display_prediction_history(self):
        """Display prediction history from database."""
        st.subheader("Prediction History")
        
        # Récupérer les prédictions de la base de données
        predictions = self.session.query(Prediction).order_by(Prediction.prediction_time.desc()).all()
        
        if predictions:
            # Convertir en DataFrame pour un meilleur affichage
            df = pd.DataFrame([
                {
                    'Resume Type': p.resume_type,
                    'Predicted Label': p.predicted_label,
                    'Prediction Time': p.prediction_time.strftime('%Y-%m-%d %H:%M:%S')
                }
                for p in predictions
            ])
            st.dataframe(df)
        else:
            st.info("No predictions have been made yet.")

    def _read_sample_file(self, sample: str) -> str:
        sample_file = "_".join(sample.upper().split()) + ".txt"
        with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
            return file.read()

    def _call_inference_api(self, text: str) -> requests.Response:
        return requests.post(self.api_url, json={"text": text}) 