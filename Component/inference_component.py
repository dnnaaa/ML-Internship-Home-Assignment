import streamlit as st
from .base_component import DashboardComponent
from utils.helpers import load_sample_text, run_inference , save_inference , show_inference , delete_inference
from data_ml_assignment.constants import LABELS_MAP
import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import Counter


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

                        # Extract predictions
                        predictions = [item["prediction"] for item in data["predictions"]]
                        prediction_counts = Counter(predictions)

            
                        labels = list(prediction_counts.keys())
                        sizes = list(prediction_counts.values())

                        st.title("Visualization of Total Predictions")
                        # Generate the donut chart
                        fig, ax = plt.subplots(figsize=(6, 6))
                        wedges, texts, autotexts = ax.pie(
                            sizes,
                            labels=labels,
                            autopct='%1.1f%%',
                            startangle=140,
                            wedgeprops={'width': 0.4}
                        )

                        # Add title and format
                        ax.set_title("Prediction Distribution")
                        plt.setp(autotexts, size=10, weight="bold")

                        # Display the chart in Streamlit
                        st.pyplot(fig)
                    else:
                        st.write("The Resume samples for inference does not exist on files , please select another one ! ")

                except Exception as e:
                    st.error("Failed to call Inference API!")
        
                    st.exception(e)
                    
        # Create a form to delete an inference based on the ID
        with st.form("delete_form"):
                            st.write("Enter the ID of the inference to delete:")
                            id_to_delete = st.text_input("Inference ID")
                            delete_button = st.form_submit_button("Delete Inference")

                            if delete_button:
                                try:
                                    if id_to_delete:
                                        delete_inference(int(id_to_delete))  # Call delete function
                                        st.success(f"Inference with ID {id_to_delete} has been deleted.")
                                    else:
                                        st.warning("Please enter a valid ID.")
                                except Exception as e:
                                    st.error("Failed to delete the inference!")
                                    st.exception(e)