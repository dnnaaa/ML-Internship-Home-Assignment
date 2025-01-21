import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import io

def show_eda(dataset):
    """
    Perform exploratory data analysis (EDA) on the provided dataset and display results using Streamlit.

    Args:
        dataset (pd.DataFrame): The dataset to analyze.
    """
    
    st.write("## Dataset Overview")
    st.write("### Dataset Preview")
    st.dataframe(dataset.head())

    st.write("### Dataset Dimensions")
    st.text(f"Number of rows: {dataset.shape[0]} | Number of columns: {dataset.shape[1]}")

    st.write("### Column Types")
    st.dataframe(dataset.dtypes)

    buffer = io.StringIO()
    dataset.info(buf=buffer)
    details = buffer.getvalue()
    st.write("### Dataset Information")
    st.text(details)

    st.write("### Label Distribution")
    label_distribution = dataset["label"].value_counts()
    st.bar_chart(label_distribution)

    st.write("### the content of a specific row")
    content = dataset[dataset['label'] == 0]['resume'].iloc[0]
    st.write(content[:1000] + "...")

    st.write("## Descriptive Statistics")
    st.dataframe(dataset.describe())

    st.write("## Missing Values Analysis")
    missing_values = dataset.isnull().sum()
    st.bar_chart(missing_values)

    st.write("### Label Distribution (Countplot)")
    plt.figure(figsize=(10, 6))
    sns.countplot(x="label", data=dataset)
    st.pyplot(plt)

    if "resume" in dataset.columns:
        st.write("### Text Length Distribution")
        text_lengths = dataset["resume"].str.len()
        plt.figure(figsize=(10, 6))
        sns.histplot(text_lengths, kde=True, bins=30)
        st.pyplot(plt)
    
    st.write("### Label Distribution by Category")
    label = st.selectbox("Choose a label to explore", options=dataset['label'].unique())
    filtered_data = dataset[dataset['label'] == label]
    sns.countplot(x='label', data=filtered_data)
    st.pyplot(plt)