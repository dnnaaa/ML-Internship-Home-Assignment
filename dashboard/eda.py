import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

def load_data(uploaded_file):
    """Secure data loading with format validation"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(BytesIO(uploaded_file.read()))
        else:
            st.error("Unsupported file format. Please upload CSV or Parquet.")
            return None
            
        required_cols = {'resume', 'label'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None
            
        return df.dropna(subset=['resume', 'label'])
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return None

def display_eda_section():
    st.header("📊 Advanced Data Analysis")
    
    with st.expander("📥 Data Upload", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload Dataset", 
            type=["csv", "parquet"],
            help="Supported formats: CSV, Parquet (must contain 'resume' and 'label' columns)"
        )
    
    if not uploaded_file:
        st.info("👋 Please upload a dataset to begin analysis")
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    # Dashboard Metrics
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Resumes", df.shape[0])
        with col2:
            st.metric("Unique Labels", df['label'].nunique())
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())

    # Interactive Analysis
    tab1, tab2, tab3 = st.tabs(["Distribution", "Text Analysis", "Advanced"])
    
    with tab1:
        st.subheader("Label Distribution")
        fig = px.pie(df, names='label', hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Text Characteristics")
        df['text_length'] = df['resume'].str.len()
        fig = px.histogram(df, x='text_length', color='label', marginal="rug")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.subheader("Correlation Matrix")
        numeric_df = df.select_dtypes(include='number')
        if not numeric_df.empty:
            fig = px.imshow(numeric_df.corr(), text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns for correlation analysis")