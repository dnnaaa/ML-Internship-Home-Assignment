# dashboard/components/inference.py
import streamlit as st
import requests
from data_ml_assignment.constants import LABELS_MAP, SAMPLES_PATH

def inference_section():
    st.header("Resume Inference")
    st.info(
        "This section simplifies the inference process. "
        "Choose a test resume and observe the label that your trained pipeline will predict."
    )

    # Sélectionner un échantillon de résumé pour l'inférence
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
                # Construire le nom du fichier du résumé
                sample_file = "_".join(sample.upper().split()) + ".txt"
                
                # Lire le fichier de résumé
                with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                    sample_text = file.read()

                # Envoyer la requête à l'API FastAPI pour l'inférence
                response = requests.post(
                    "http://localhost:9000/api/inference", json={"text": sample_text}
                )

                if response.status_code == 200:
                    result = response.json()
                    label = LABELS_MAP.get(int(result), "Unknown label")
                    st.success("Done!")
                    st.metric(label="Predicted Resume Label", value=label)
                else:
                    st.error("Failed to call Inference API! Check if the API is running.")
            
            except Exception as e:
                st.error("An error occurred during inference!")
                st.exception(e)
