import streamlit as st
from .base_component import DashboardComponent
from utils.helpers import load_sample_text, run_inference , save_inference , show_inference
from data_ml_assignment.constants import LABELS_MAP
import pandas as pd
import json



#Class used inside the dashboard.py
class InferenceComponent(DashboardComponent):
    def render(self):
        st.header("Resume Inference")
        st.info(
            "This section simplifies the inference process. "
            "Choose a test resume and observe the label that your trained pipeline will predict."
        )

        # Dropdown to select a sample resume
        sample = st.selectbox(
            "Resume samples for inference",
            options=tuple(LABELS_MAP.values()),  # Available labels
            index=0  # Default selection
        )
        infer = st.button("Run Inference")

        if infer:
            with st.spinner("Running inference..."):
                try:
                    # Load the sample text
                    sample_text = load_sample_text(sample)

                    if sample_text != "Not Exist" :
                        # Call the API to get predictions
                        prediction = run_inference(sample_text)

                        #Map prediction to label
                        label = LABELS_MAP.get(int(float(prediction)))
                        
                        # Display results
                        st.success("Done!")
                        st.metric(label="Status", value=f"Resume label: {label}")

                        #Save on database
                        save_inference(sample_text , label)

                        #Get From Database
                        data=show_inference()
                        # Parse the JSON string into a dictionary
                        data = json.loads(data)
                        df = pd.DataFrame(data['predictions'])
                        #Show all the data
                        st.dataframe(df)
                    else:
                        st.write("The Resume samples for inference does not exist on files , please select another one ! ")

                except Exception as e:
                    st.error("Failed to call Inference API!")
                    st.exception(e)
