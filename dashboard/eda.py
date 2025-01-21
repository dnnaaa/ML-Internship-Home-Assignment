import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file) -> pd.DataFrame:
    """Load the dataset from the uploaded CSV file."""
    try:
        data = pd.read_csv(file)
        if data.empty:
            raise ValueError("The uploaded file is empty.")
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error


def display_summary(data: pd.DataFrame) -> None:
    """Display the summary statistics of the dataset."""
    st.write("### Summary Statistics")
    st.write(data.describe(include="all"))


def plot_histogram(data: pd.DataFrame, column: str, kde: bool, binwidth: int) -> None:
    """Plot a histogram for the given column in the dataset."""
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=kde, binwidth=binwidth)
    st.pyplot(fig)


def display_eda_section() -> None:
    """Display the exploratory data analysis (EDA) section for the resume dataset."""
    st.header("Exploratory Data Analysis")
    st.info(
        "In this section, you can create insightful graphs "
        "about the resume dataset that you were provided."
    )

    file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if file:
        data = load_data(file)
        
        if data.empty:
            st.warning("The uploaded file is empty or invalid. Please try again.")
            return

        display_summary(data)

        numeric_columns = data.select_dtypes(include=["number"]).columns
        if not numeric_columns.any():
            st.warning("No numeric columns available for visualization.")
            return

        column = st.selectbox("Select column for histogram", numeric_columns)
        kde = st.checkbox("Show KDE", value=True)
        binwidth = st.slider("Select bin width", min_value=1, max_value=100, value=1)
        plot_histogram(data, column, kde, binwidth)
