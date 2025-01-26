from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

#This function is used inside the eda_component
def Dataset_wordscloudHelperF(dataset , label_counts_key): ### For Data
    st.subheader("Word Cloud of Resumes")
    all_text = ' '.join(dataset['cleaned_resume'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Top 10 Most Common Words per resume
    st.subheader("Top 10 Most Frequent Words")
    all_words = Counter(all_text.split())
    top_words = all_words.most_common(10)
    st.bar_chart(pd.DataFrame(top_words, columns=["Word", "Frequency"]).set_index("Word"))

    # Top 10 Most Common Words per Job Title
    st.subheader("Top 10 Most Frequent Job Title")
    all_words2 = Counter(label_counts_key)
    top_words = all_words2.most_common(10)
    st.bar_chart(pd.DataFrame(top_words, columns=["Word", "Frequency"]).set_index("Word"))
###


