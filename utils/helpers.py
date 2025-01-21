#This helper function is used inside the training_component.py ; their role is to show interactive metrics on the screen
def display_metrics(accuracy, f1):
    import streamlit as st
    col1, col2 = st.columns(2)
    col1.metric(label="Accuracy score", value=str(round(accuracy, 4)))
    col2.metric(label="F1 score", value=str(round(f1, 4)))

#This helper function is used inside the  inference_component ; their role is to load the text from the file_path
def load_sample_text(sample_name):
    from pathlib import Path
    from data_ml_assignment.constants import SAMPLES_PATH

    sample_file = "_".join(sample_name.upper().split()) + ".txt"
    sample_path = Path(SAMPLES_PATH) / sample_file
    
    # Check if the file exists
    if not sample_path.exists():
        return "Not Exist"

    # Load and return the content of the file
    with open(sample_path, encoding="utf-8") as file:
        return file.read()


#This helper function is used inside the  inference_component ; their role is to run the predict functionlity
def run_inference(sample_text):
    import requests
    API_URL_inference = "http://localhost:9000/api/inference"
    response = requests.post(API_URL_inference, json={"text": sample_text})
    response.raise_for_status()
    return response.text

#This helper function is used inside the  inference_component ; their role is to add the inference or the predicted label on the database
def save_inference(sample_text , label):
    import requests
    API_URL_save= "http://localhost:9000/api/save"
    response=requests.post(API_URL_save, json={"text": sample_text, "predict": label})
    response.raise_for_status()
    return "Data saved on database"

##This helper function is used inside the  inference_component ; their role is to show all the predicted labels or the inferences from the database
def show_inference():
    import requests
    API_URL_get="http://localhost:9000/api/get"
    response=requests.post(API_URL_get)
    response.raise_for_status()
    return response.text
