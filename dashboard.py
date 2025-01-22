import time
import streamlit as st
from PIL import Image
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from data_ml_assignment.training.train_pipeline import TrainingPipeline
from data_ml_assignment.constants import CM_PLOT_PATH, LABELS_MAP, SAMPLES_PATH, ROC_PLOT_PATH

@st.cache_data
def load_data():
    # Example: Load a CSV file
    data = pd.read_csv("./data/raw/resume.csv")  # Replace with your dataset path
    data['label_name'] = data['label'].map(LABELS_MAP)
    data['resume_length'] = data['resume'].str.split().str.len()

    return data

# Load the dataset
data = load_data()

st.title("Resume Classification Dashboard")
st.sidebar.title("Dashboard Modes")

sidebar_options = st.sidebar.selectbox("Options", ("EDA", "Training", "Inference"))

if sidebar_options == "EDA":
    st.header("Exploratory Data Analysis")

    # Dataset Overview in grid
    st.subheader("Dataset Overview")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Total Resumes", len(data))
    with metrics_col2:
        st.metric("Total Categories", len(LABELS_MAP))
    with metrics_col3:
        st.metric("Avg Words per Resume", int(data['resume_length'].mean()))

    # Raw Data Expander
    with st.expander("View Raw Data"):
        st.dataframe(
            data,
            column_config={
                "resume": st.column_config.TextColumn(
                    "Resume Content",
                    width="medium",
                    help="Content of the resume"
                ),
                "label": st.column_config.TextColumn(
                    "Category ID",
                    width="small"
                ),
                "label_name": st.column_config.TextColumn(
                    "Category Name",
                    width="small"
                ),
                "resume_length": st.column_config.NumberColumn(
                    "Word Count",
                    help="Number of words in resume"
                )
            }
        )

    # Create two columns for main visualizations
    left_col, right_col = st.columns(2)

    with left_col:
        # Distribution of Labels
        st.subheader("Category Distribution")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        label_counts = data["label_name"].value_counts()
        sns.barplot(x=label_counts.values, y=label_counts.index, ax=ax1, palette="viridis")
        plt.xlabel("Number of Resumes")
        plt.ylabel("Job Category")
        ax1.set_title("Resume Categories")
        st.pyplot(fig1)

        # Distribution table below the graph
        dist_df = pd.DataFrame({
            'Category': label_counts.index,
            'Count': label_counts.values,
            'Percentage': (label_counts.values / len(data) * 100).round(2)
        })
        st.dataframe(dist_df, hide_index=True, height=200)

    with right_col:
        # Length Distribution
        st.subheader("Length Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=data, x='resume_length', y='label_name', ax=ax2, palette='viridis')
        plt.xlabel("Number of Words")
        plt.ylabel("Job Category")
        ax2.set_title("Resume Length by Category")
        st.pyplot(fig2)

        # Length statistics below the graph
        length_stats = data.groupby('label_name')['resume_length'].agg([
            ('Min', 'min'),
            ('Max', 'max'),
            ('Median', 'median'),
            ('Mean', 'mean')
        ]).round(0)
        st.dataframe(length_stats, height=200)

    # Word Analysis in bottom section
    st.subheader("Word Analysis by Category")
    analysis_col1, analysis_col2 = st.columns([1, 3])

    with analysis_col1:
        selected_label = st.selectbox(
            "Select a job category:",
            options=sorted(LABELS_MAP.values()),
            index=0,
        )

        # Add some category stats
        category_data = data[data['label_name'] == selected_label]
        st.metric("Category Size", len(category_data))
        st.metric("Avg Length", int(category_data['resume_length'].mean()))

    with analysis_col2:
        # TF-IDF Analysis
        label_resumes = data[data["label_name"] == selected_label]["resume"]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
        tfidf_matrix = vectorizer.fit_transform(label_resumes)
        tfidf_scores = pd.DataFrame({
            'Term': vectorizer.get_feature_names_out(),
            'Score': tfidf_matrix.toarray().mean(axis=0)
        }).sort_values('Score', ascending=False).head(10)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=tfidf_scores, x='Score', y='Term', palette='viridis')
        plt.xlabel("TF-IDF Score")
        plt.ylabel("Term")
        ax3.set_title(f"Top Terms in {selected_label} Resumes")
        st.pyplot(fig3)


elif sidebar_options == "Training":
    st.header("Pipeline Training")
    st.info(
        "Before you proceed to training your pipeline, make sure you "
        "have checked your training pipeline code and that it is set properly."
    )

    name = st.text_input("Pipeline name", placeholder="Naive Bayes")
    serialize = st.checkbox("Save pipeline")
    train = st.button("Train pipeline")

    if train:
        with st.spinner("Training pipeline, please wait..."):
            try:
                tp = TrainingPipeline()
                tp.train(serialize=serialize, model_name=name)
                tp.render_confusion_matrix()
                accuracy, f1, precision, recall = tp.get_model_performance()  # Updated to return more metrics

                # Display metrics in columns
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric(label="Accuracy", value=str(round(accuracy, 4)))
                col2.metric(label="F1 Score", value=str(round(f1, 4)))
                col3.metric(label="Precision", value=str(round(precision, 4)))
                col4.metric(label="Recall", value=str(round(recall, 4)))

                # Display confusion matrix
                st.image(Image.open(CM_PLOT_PATH), width=850, caption="Confusion Matrix")


                # Display classification report
                st.subheader("Classification Report")
                st.dataframe(tp.get_classification_report())

            except Exception as e:
                st.error("Failed to train the pipeline!")
                st.exception(e)

else:
    st.header("Resume Inference")
    st.info(
        "This section allows you to classify resumes using the trained model. "
        "You can either select a sample resume or input your own text."
    )

    # Create tabs for different input methods
    input_tab, sample_tab = st.tabs(["Custom Input", "Sample Resumes"])

    with input_tab:
        # Text area for custom resume input
        custom_text = st.text_area(
            "Enter resume text",
            height=300,
            placeholder="Paste the resume content here...",
            help="Enter the resume text you want to classify"
        )

        # Add file uploader for text files
        uploaded_file = st.file_uploader(
            "Or upload a text file",
            type=['txt', 'doc', 'docx'],
            help="Upload a resume file to classify"
        )

        if uploaded_file:
            try:
                custom_text = uploaded_file.read().decode('utf-8')
                st.success("File loaded successfully!")
            except Exception as e:
                st.error("Error reading file. Please ensure it's a valid text file.")
                st.exception(e)

        infer_custom = st.button("Classify Custom Resume", disabled=not custom_text)

        if infer_custom:
            with st.spinner("Analyzing resume..."):
                try:
                    result = requests.post(
                        "http://localhost:9000/api/inference",
                        json={"text": custom_text}
                    )
                    if result.status_code == 200:
                        # Get prediction and confidence scores
                        prediction = LABELS_MAP.get(int(float(result.text)))

                        # Display results in a nice format
                        st.success("Analysis Complete!")

                        # Create three columns for metrics
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(
                                label="Predicted Category",
                                value=prediction
                            )

                        # Display the analyzed text in an expander
                        with st.expander("View Analyzed Text"):
                            st.text_area(
                                "Resume Content",
                                value=custom_text,
                                height=200,
                                disabled=True
                            )
                    else:
                        st.error(f"API Error: {result.status_code}")

                except Exception as e:
                    st.error("Failed to call Inference API!")
                    st.exception(e)

    with sample_tab:
        # Add a description for sample resumes
        st.markdown("""
            Select from pre-defined sample resumes to test the model.
            These samples represent different job categories and writing styles.
        """)

        col1, col2 = st.columns(2)

        with col1:
            sample = st.selectbox(
                "Sample Resumes",
                tuple(LABELS_MAP.values()),
                index=None,
                placeholder="Select a resume sample",
            )

        with col2:
            # Add expected label display
            if sample:
                st.info(f"Expected Category: {sample}")

        infer_sample = st.button("Classify Sample Resume", disabled=not sample)

        if infer_sample:
            with st.spinner("Analyzing sample resume..."):
                try:
                    sample_file = "_".join(sample.upper().split()) + ".txt"
                    with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                        sample_text = file.read()

                    result = requests.post(
                        "http://localhost:9000/api/inference",
                        json={"text": sample_text}
                    )

                    if result.status_code == 200:
                        # Get prediction
                        prediction = LABELS_MAP.get(int(float(result.text)))

                        # Display results
                        st.success("Analysis Complete!")

                        # Create columns for metrics
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(
                                label="Predicted Category",
                                value=prediction
                            )

                        with col2:
                            # Show if prediction matches expected category
                            is_correct = prediction == sample
                            st.metric(
                                label="Prediction Status",
                                value="Correct" if is_correct else "Incorrect",
                                delta="✓" if is_correct else "✗",
                                delta_color="normal" if is_correct else "inverse"
                            )

                        # Display the sample text in an expander
                        with st.expander("View Sample Resume"):
                            st.text_area(
                                "Resume Content",
                                value=sample_text,
                                height=200,
                                disabled=True
                            )
                    else:
                        st.error(f"API Error: {result.status_code}")

                except Exception as e:
                    st.error("Failed to call Inference API!")
                    st.exception(e)