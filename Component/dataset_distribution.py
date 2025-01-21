import streamlit as st
import matplotlib.pyplot as plt

#This function is used inside the eda_component
def Dataset_distributionHelperF(dataset , LABELS_MAP): ### For Data
    # Label Distribution
    dataset['label_name'] = dataset['label'].map(LABELS_MAP)

    # Calculate label distribution
    st.subheader("Label Distribution")
    label_counts = dataset['label_name'].value_counts()

    # Display label distribution as a bar chart
    st.bar_chart(label_counts)

    # Histogram of Resume Lengths
    st.subheader("Resume Length Distribution")
    plt.hist(dataset['resume_length'], bins=30, color='blue', alpha=0.7)
    plt.title("Histogram of Resume Lengths")
    plt.xlabel("Resume Length (characters)")
    plt.ylabel("Frequency")
    st.pyplot(plt)
###
