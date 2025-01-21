import streamlit as st
import pickle
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, DateTime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect
from datetime import datetime
import pandas as pd
import os

# Mapping des labels sans nombres
LABELS = [
    '.Net Developer', 'Business Analyst', 'Business Intelligence', 
    'Help Desk and Support', 'Informatica Developer', 'Java Developer', 
    'Network and System Administrator', 'Oracle DBA', 'Project Manager', 
    'Quality Assurance', 'SAP', 'SQL Developer', 
    'Sharepoint Developer', 'Web Developer'
]

#! Load vectorizer and model from disk
def load_pipeline(pipeline_name):
    try:
        with open(f"models/{pipeline_name}_vectorizer.pkl", "rb") as vec_file:
            vectorizer = pickle.load(vec_file)
        with open(f"models/{pipeline_name}_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        st.success(f"Pipeline '{pipeline_name}' loaded successfully.")
        return vectorizer, model
    except FileNotFoundError:
        st.error(f"Pipeline files for '{pipeline_name}' not found.")
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
    st.info("Entrez le texte d'un CV et voyez comment le pipeline entraîné prédit son étiquette.")

    ## Set up database connection
    engine, results_table = setup_database()

    ## Load pipeline
    if "pipeline_name" not in st.session_state:
        st.session_state.pipeline_name = "naive_bayes" 
    st.text_input("Pipeline Name", value=st.session_state.pipeline_name, key="pipeline_name")

    if st.button("Load Pipeline"):
        vectorizer, model = load_pipeline(st.session_state.pipeline_name)
        st.session_state["vectorizer"] = vectorizer
        st.session_state["model"] = model

    ## Run inference if pipeline is loaded
    if "vectorizer" in st.session_state and "model" in st.session_state:
        # Option 1: Sélectionner une étiquette et générer un texte de CV factice
        # resume_samples = LABELS  # Affiche un sous-ensemble d'étiquettes
        # selected_resume = st.selectbox("Select a sample CV for inference", resume_samples)

        # Option 2: Saisir un texte de CV réel
        st.subheader("Entrer le Texte du CV")
        sample_resume_text = st.text_area("Texte du CV", height=200)

        if st.button("Run Inference"):
            if not sample_resume_text.strip():
                st.error("Veuillez entrer le texte du CV pour effectuer une prédiction.")
            else:
                try:
                    sample_vectorized = st.session_state["vectorizer"].transform([sample_resume_text])
                    predicted_label = str(st.session_state["model"].predict(sample_vectorized)[0])

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

if __name__ == "__main__":
    render_inference_section()
