import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import re

from data_ml_assignment.dashboard.base import DashboardComponent
from data_ml_assignment.constants import RAW_DATASET_PATH, LABELS_MAP

class EDAComponent(DashboardComponent):
    def __init__(self):
        # Load and preprocess data
        self.df = pd.read_csv(RAW_DATASET_PATH)
        self.df['label_name'] = self.df['label'].map(LABELS_MAP)
        self.df['word_count'] = self.df['resume'].apply(lambda x: len(str(x).split()))
        self.df['char_count'] = self.df['resume'].apply(len)
        
    def preprocess_text(self, text):
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        # Simple word splitting
        return text.split()

    def get_common_words(self, n=20, min_length=3):
        all_words = []
        for text in self.df['resume']:
            words = self.preprocess_text(text)
            # Filter out short words and common English words
            words = [w for w in words if len(w) >= min_length and w not in COMMON_STOP_WORDS]
            all_words.extend(words)
        return Counter(all_words).most_common(n)

    def render(self):
        st.header("Exploratory Data Analysis")
        st.info(
            "This section provides insights about the resume dataset through "
            "statistical descriptions and visualizations."
        )

        # Basic Dataset Statistics
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Resumes", len(self.df))
        col2.metric("Job Categories", len(self.df['label'].unique()))
        col3.metric("Average Words per Resume", int(self.df['word_count'].mean()))

        # Distribution of Job Categories
        st.subheader("Distribution of Job Categories")
        fig_categories = px.bar(
            self.df['label_name'].value_counts().reset_index(),
            x='label_name',
            y='count',
            title="Number of Resumes per Job Category"
        )
        fig_categories.update_layout(
            xaxis_title="Job Category",
            yaxis_title="Number of Resumes",
            xaxis={'tickangle': 45}
        )
        st.plotly_chart(fig_categories)

        # Word Count Distribution
        st.subheader("Resume Length Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_word_dist = px.box(
                self.df,
                y='word_count',
                title="Distribution of Resume Lengths (Words)"
            )
            st.plotly_chart(fig_word_dist)

        with col2:
            avg_words = self.df.groupby('label_name')['word_count'].mean().sort_values(ascending=True)
            fig_avg_words = px.bar(
                avg_words,
                orientation='h',
                title="Average Words per Category"
            )
            fig_avg_words.update_layout(
                xaxis_title="Average Word Count",
                yaxis_title="Job Category"
            )
            st.plotly_chart(fig_avg_words)

        # Most Common Words
        st.subheader("Most Common Words in Resumes")
        try:
            common_words = self.get_common_words()
            words, counts = zip(*common_words)
            
            fig_words = px.bar(
                x=words,
                y=counts,
                title="20 Most Common Words Across All Resumes"
            )
            fig_words.update_layout(
                xaxis_title="Word",
                yaxis_title="Frequency",
                xaxis={'tickangle': 45}
            )
            st.plotly_chart(fig_words)
        except Exception as e:
            st.warning("Could not generate common words visualization. Error: " + str(e))

        # Detailed Statistics
        st.subheader("Detailed Statistics")
        stats_df = self.df.groupby('label_name').agg({
            'word_count': ['mean', 'min', 'max', 'std'],
            'char_count': ['mean', 'min', 'max', 'std']
        }).round(2)
        
        stats_df.columns = ['Avg Words', 'Min Words', 'Max Words', 'Std Words',
                          'Avg Chars', 'Min Chars', 'Max Chars', 'Std Chars']
        st.dataframe(stats_df)

        # Raw Data Preview
        st.subheader("Raw Data Preview")
        if st.checkbox("Show Raw Data"):
            st.write(self.df[['label_name', 'resume', 'word_count', 'char_count']])

# Common English stop words to filter out
COMMON_STOP_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
    'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
    'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
    'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
    'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
    'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
    'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
    'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
    'give', 'day', 'most', 'us', 'is', 'was', 'are', 'were', 'been',
    'has', 'had', 'did', 'doing', 'does', 'done'
} 