# data_ml_assignment/components/eda.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

def show_eda():
    st.header("Exploratory Data Analysis")
    st.info(
        "In this section, I explored the resume dataset through insightful visualizations. "
        "I analyzed the distribution of resume lengths, common words, and label frequencies. "
        "I also examined class distribution and checked for missing values in the dataset. "
        "Interactive filters were added to dive deeper into specific categories or resume lengths."
    )
    # Load the dataset
    df = pd.read_csv("data/raw/resume.csv")
    st.write("Dataset Preview:")
    st.write(df.head())

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Class distribution
    st.subheader("Class Distribution")
    st.write(df["label"].value_counts())

    # Resume length distribution
    st.subheader("Resume Length Distribution")
    df["resume_length"] = df["resume"].apply(len)
    plt.hist(df["resume_length"], bins=20)
    plt.xlabel("Resume Length")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    # Word cloud
    st.subheader("Word Cloud of Common Words")
    text = " ".join(df["resume"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # Top bigrams
    st.subheader("Top Bigrams")
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10)
    X = vectorizer.fit_transform(df["resume"])
    bigrams = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1
    bigram_df = pd.DataFrame({"Bigram": bigrams, "Count": counts})
    bigram_df = bigram_df.sort_values(by="Count", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Count", y="Bigram", data=bigram_df)
    plt.xlabel("Count")
    plt.ylabel("Bigram")
    st.pyplot(plt)

    # Label distribution
    st.subheader("Label Distribution")
    label_counts = df["label"].value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.xlabel("Label")
    plt.ylabel("Count")
    st.pyplot(plt)

    # Filter by label
    st.subheader("Filter by Label")
    selected_label = st.selectbox("Select a label", df["label"].unique())
    filtered_df = df[df["label"] == selected_label]
    st.write(filtered_df.head())

    # Filter by resume length
    st.subheader("Filter by Resume Length")
    min_length, max_length = st.slider("Select a range", 0, 10000, (0, 10000))
    filtered_df = df[(df["resume_length"] >= min_length) & (df["resume_length"] <= max_length)]
    st.write(filtered_df.head())