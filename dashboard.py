import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from joblib import load

# Import constants
from data_ml_assignment.constants import SAMPLES_PATH, MODELS_PATH
from database import add_prediction_to_db, get_all_predictions

class EDAComponent:
    def render(self):
        st.header("Exploratory Data Analysis")
        st.info("Explore the dataset with statistical descriptions and visualizations.")

        # Load dataset with error handling
        try:
            resumes_df = pd.read_csv(SAMPLES_PATH / 'resumes.csv')
            st.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Statistical Summary", "Visualizations"])

        with tab1:
            st.subheader("Resume Dataset")
            rows = st.slider("Select number of rows to preview:", min_value=5, max_value=len(resumes_df), value=10)
            st.dataframe(resumes_df.head(rows))

        with tab2:
            st.subheader("Statistical Descriptions")
            st.write(resumes_df.describe())
            st.write("Categorical Column Distribution:")
            st.write(resumes_df['Label'].value_counts())

        with tab3:
            st.subheader("Visualizations")
            
            # Distribution of labels
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            resumes_df['Label'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Distribution of Resume Labels')
            ax1.set_xlabel('Label')
            ax1.set_ylabel('Count')
            st.pyplot(fig1)

            # Length of resumes by label
            st.markdown("#### Resume Length by Label")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            resumes_df['resume_length'] = resumes_df['resume'].str.len()
            sns.boxplot(x='Label', y='resume_length', data=resumes_df, ax=ax2)
            ax2.set_title('Resume Length Distribution by Label')
            st.pyplot(fig2)

class InferenceComponent:
    def __init__(self):
        # Load serialized model and vectorizer
        self.model = load(MODELS_PATH / "improved_model.joblib")
        self.vectorizer = load(MODELS_PATH / "improved_model_vectorizer.joblib")

    def render(self):
        st.header("Resume Prediction")
        st.info("Predict resume labels and view prediction history.")

        # Resume text input
        resume_text = st.text_area("Enter resume text:", height=300)

        # Prediction button
        if st.button("Predict Label"):
            if resume_text.strip():
                try:
                    # Vectorization and prediction
                    vectorized_text = self.vectorizer.transform([resume_text])
                    predicted_label = self.model.predict(vectorized_text)[0]
                    probabilities = self.model.predict_proba(vectorized_text)[0]
                    confidence = max(probabilities)

                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"Predicted Label: {predicted_label}")
                    with col2:
                        st.info(f"Confidence: {confidence:.2%}")

                    # Save to database
                    add_prediction_to_db(resume_text, predicted_label)
                    st.success("Prediction saved to database.")

                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.warning("Please enter resume text.")

        # Prediction history
        st.subheader("Prediction History")
        predictions = get_all_predictions()
        if predictions:
            for pred in predictions:
                with st.expander(f"Prediction for {pred.predicted_label}"):
                    st.text(pred.resume[:500] + "..." if len(pred.resume) > 500 else pred.resume)
                    st.write(f"**Predicted Label:** {pred.predicted_label}")
        else:
            st.info("No predictions stored yet.")

def main():
    st.set_page_config(page_title="Resume Analysis", page_icon="📊")
    st.sidebar.title("Resume Analysis Dashboard")
    
    menu = ["EDA", "Inference"]
    choice = st.sidebar.radio("Select Section", menu)

    if choice == "EDA":
        EDAComponent().render()
    elif choice == "Inference":
        InferenceComponent().render()

if __name__ == "__main__":
    main()