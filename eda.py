import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from data_ml_assignment.constants import RAW_DATASET_PATH


class EDA:
    def __init__(self, data_path = RAW_DATASET_PATH):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)


    def show_summary(self):
        st.subheader("Dataset Summary")
        st.write(f"Number of rows: {self.df.shape[0]}")
        st.write(f"Number of columns: {self.df.shape[1]}")
        st.write("First 5 rows:")
        st.dataframe(self.df.head())
        st.write("Basic Statistics:")
        st.write(self.df.describe())

    def show_category_distribution(self):
        st.subheader("Resume Category Distribution")
        fig = px.bar(self.df['label'].value_counts(), title="Resume Categories")
        st.plotly_chart(fig, key="unique_key_1")

    def show_missing_values(self):
        st.subheader("Missing Values")
        st.write(self.df.isnull().sum())

    def show_word_cloud(self, text_column="resume"):
        st.subheader("Word Cloud")
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt

            text = " ".join(self.df[text_column].dropna())
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)
        except ImportError:
            st.warning("WordCloud library is not installed. Install it using `pip install wordcloud`.")

