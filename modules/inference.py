#! Import necessary libraries
import streamlit as st
import pickle
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, DateTime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect
from datetime import datetime
import pandas as pd
import os

# Labels without numbers
LABELS = [
    '.Net Developer', 'Business Analyst', 'Business Intelligence', 
    'Help Desk and Support', 'Informatica Developer', 'Java Developer', 
    'Network and System Administrator', 'Oracle DBA', 'Project Manager', 
    'Quality Assurance', 'SAP', 'SQL Developer', 
    'Sharepoint Developer', 'Web Developer'
]

#! Load pipeline components from disk
def load_pipeline(pipeline_name):
    try:
        with open(f"models/{pipeline_name}_vectorizer.pkl", "rb") as vec_file:
            vectorizer = pickle.load(vec_file)
        with open(f"models/{pipeline_name}_svd.pkl", "rb") as svd_file:
            svd = pickle.load(svd_file)
        with open(f"models/{pipeline_name}_scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        with open(f"models/{pipeline_name}_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        st.success(f"Pipeline '{pipeline_name}' loaded successfully.")
        return vectorizer, svd, scaler, model
    except FileNotFoundError:
        st.error(f"One or more pipeline files for '{pipeline_name}' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading pipeline '{pipeline_name}': {e}")
        st.stop()

#! Connect to the database and load the table
def setup_database(db_path="results_Db.db", table_name="results_table"):
    if not os.path.exists(db_path):
        st.error(f"Database file '{db_path}' does not exist.")
        st.stop()
    try:
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"timeout": 30, "check_same_thread": False}
        )
        metadata = MetaData()
        inspector = inspect(engine)

        # Load table if exists
        if inspector.has_table(table_name):
            results_table = Table(table_name, metadata, autoload_with=engine)
        else:
            st.error(f"Table '{table_name}' does not exist.")
            st.stop()
        return engine, results_table
    except SQLAlchemyError as e:
        st.error(f"Database connection or table loading failed: {e}")
        st.stop()

#! Save prediction results into the database
def save_prediction(engine, results_table, resume_text, predicted_label):
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(results_table.insert().values(
                resume_text=resume_text,
                predicted_label=predicted_label,
                timestamp=datetime.utcnow()
            ))
        st.success("Prediction saved successfully")
    except SQLAlchemyError as e:
        st.error(f"Failed to save prediction: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

#! Streamlit UI for running and saving predictions
def render_inference_section():
    st.header("Resume Classification Dashboard")
    st.subheader("CV Inference")
    st.info("Enter resume text and see the predicted label from the trained pipeline.")

    ## Set up database connection
    engine, results_table = setup_database()

    ## Load pipeline
    if "pipeline_name" not in st.session_state:
        st.session_state.pipeline_name = "naive_bayes" 
    st.text_input("Pipeline Name", value=st.session_state.pipeline_name, key="pipeline_name")

    if st.button("Load Pipeline"):
        vectorizer, svd, scaler, model = load_pipeline(st.session_state.pipeline_name)
        st.session_state["vectorizer"] = vectorizer
        st.session_state["svd"] = svd
        st.session_state["scaler"] = scaler
        st.session_state["model"] = model

    ## Run inference if pipeline is loaded
    if all(key in st.session_state for key in ["vectorizer", "svd", "scaler", "model"]):
        st.subheader("Enter Resume Text")
        sample_resume_text = st.text_area("Resume Text", height=200)

        if st.button("Run Inference"):
            if not sample_resume_text.strip():
                st.error("Please enter resume text for prediction.")
            else:
                try:
                    # Sequential transformation: Vectorizer -> SVD -> Scaler
                    sample_vectorized = st.session_state["vectorizer"].transform([sample_resume_text])
                    sample_svd = st.session_state["svd"].transform(sample_vectorized)
                    sample_scaled = st.session_state["scaler"].transform(sample_svd)
                    
                    # Prediction
                    predicted_label = str(st.session_state["model"].predict(sample_scaled)[0])

                    # Display prediction and save it
                    st.metric("Predicted Label", predicted_label)
                    save_prediction(engine, results_table, sample_resume_text, predicted_label)
                except Exception as e:
                    st.error(f"Inference or saving failed: {e}")

    ## Show prediction history
    st.subheader("Prediction History")
    try:
        with engine.connect() as conn:
            results_df = pd.read_sql(results_table.select().order_by(results_table.c.timestamp.desc()), conn)
        st.dataframe(results_df)
    except SQLAlchemyError as e:
        st.error(f"Failed to load prediction history: {e}")

#! Main section
if __name__ == "__main__":
    render_inference_section()
