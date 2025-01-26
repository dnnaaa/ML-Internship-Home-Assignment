import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter

#! EDA
def render_eda_section(data_path):
    # Section EDA
    st.header("Exploratory Data Analysis")
    st.info("Analyze dataset for resume classification.")

    # Load dataset
    data = pd.read_csv(data_path)

    # Dataset Overview
    st.subheader("Dataset Overview")
    st.dataframe(data.head())
    st.write(f"**Rows:** {data.shape[0]}, **Columns:** {data.shape[1]}")

    # Label Distribution
    st.subheader("Label Distribution")
    label_counts = data['label'].value_counts().sort_index()
    label_names = {0: '.Net Developer', 1: 'Business Analyst', 2: 'Business Intelligence', 3: 'Help Desk and Support',
                   4: 'Informatica Developer', 5: 'Java Developer', 6: 'Network and System Administrator',
                   7: 'Oracle DBA', 8: 'Project Manager', 9: 'Quality Assurance', 10: 'SAP', 11: 'SQL Developer',
                   12: 'Sharepoint Developer', 13: 'Web Developer'}
    label_counts.index = label_counts.index.map(label_names)

    # Plot label distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_counts.values, y=label_counts.index, palette="viridis")
    plt.title("Label Distribution")
    st.pyplot(plt)

    st.write(f"**Unique labels:** {label_counts.shape[0]}")
    st.write(f"- **Most common:** {label_counts.idxmax()} ({label_counts.max()})")
    st.write(f"- **Least common:** {label_counts.idxmin()} ({label_counts.min()})")

    # Resume Length Distribution
    st.subheader("Resume Length Distribution")
    data['resume_length'] = data['resume'].apply(len)
    plt.figure(figsize=(8, 5))
    sns.histplot(data['resume_length'], bins=20, color='skyblue')
    plt.title("Resume Length Distribution")
    st.pyplot(plt)

    st.write(data['resume_length'].describe())

    # Average Resume Length by Label
    st.subheader("Average Resume Length by Label")
    avg_length_by_label = data.groupby('label')['resume_length'].mean().sort_values()
    avg_length_by_label.index = avg_length_by_label.index.map(label_names)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_length_by_label.values, y=avg_length_by_label.index, palette="magma")
    plt.title("Average Resume Length by Label")
    st.pyplot(plt)

    # Word Cloud for All Resumes
    st.subheader("Word Cloud of All Resumes")
    all_text = " ".join(data['resume'])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # Explore Resumes by Class
    st.subheader("Explore Resumes by Class")
    selected_label = st.selectbox("Select a label:", options=data['label'].unique().tolist())

    if selected_label is not None:
        filtered_data = data[data['label'] == selected_label]
        st.dataframe(filtered_data[['resume']].head(5))

        # Word Cloud for selected label
        class_text = " ".join(filtered_data['resume'])
        class_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(class_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(class_wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        # Most common words for selected label
        word_counts = Counter(class_text.split()).most_common(10)
        st.write(word_counts)

#! Run Streamlit App
if __name__ == "__main__":
    st.title("Resume Classification EDA")
    data_path = st.text_input("Dataset Path:", value="data/resume.csv")
    if st.button("Run EDA"):
        render_eda_section(data_path)
