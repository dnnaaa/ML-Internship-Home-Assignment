import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud  
import re  
from data_ml_assignment.constants import RAW_DATASET_PATH


def render_eda():
    st.header("Exploratory Data Analysis")
    st.info(
        "In this section, you are invited to create insightful graphs "
        "about the resume dataset that you were provided."
    )

    # Load data
    data = pd.read_csv(RAW_DATASET_PATH)  # Replace with your dataset path

    # Display basic statistics
    st.subheader("Dataset Overview")
    st.write(data.head())

    # Label Distribution
    st.subheader("Label Distribution")
    label_counts = data["label"].value_counts()
    st.bar_chart(label_counts)

    # Resume Length Analysis
    st.subheader("Resume Length Analysis")
    data["resume_length"] = data["resume"].apply(len)
    fig, ax = plt.subplots()
    sns.histplot(data["resume_length"], bins=50, ax=ax)
    st.pyplot(fig)

    # Summary Statistics for Resume Lengths
    st.subheader("Summary Statistics for Resume Lengths")
    st.write(data["resume_length"].describe())

    # Distribution of Resume Lengths by Label
    st.subheader("Distribution of Resume Lengths by Label")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="label", y="resume_length", data=data, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Correlation Between Resume Length and Label
    st.subheader("Correlation Between Resume Length and Label")
    corr = data[["label", "resume_length"]].corr()
    st.write(corr)

    # Word Cloud for Common Terms
    st.subheader("Word Cloud for Common Terms")
    all_text = " ".join(data["resume"].tolist())

    # Basic Text Cleaning
    all_text = re.sub(r"[^\w\s]", "", all_text)  # Remove punctuation
    all_text = all_text.lower()  # Convert to lowercase

    # Generate Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Data Preprocessing (Optional)
    st.subheader("Data Preprocessing")
    if st.checkbox("Show Preprocessed Text"):
        data["cleaned_resume"] = data["resume"].apply(
            lambda x: re.sub(r"[^\w\s]", "", x.lower())  # Remove punctuation and convert to lowercase
        )
        st.write(data[["resume", "cleaned_resume"]].head())