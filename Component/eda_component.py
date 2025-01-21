import streamlit as st
from .base_component import DashboardComponent
import pandas as pd
from .helperfunction.dataset_overview import Dataset_overviewHelperF
from .helperfunction.dataset_advanced_analysis import Dataset_advanced_analysisHelperF
from .helperfunction.dataset_wordscloud import Dataset_wordscloudHelperF 
from .helperfunction.dataset_preprocessing import Dataset_preprocessingHelperF
from .helperfunction.dataset_distribution import Dataset_distributionHelperF
from data_ml_assignment.constants import LABELS_MAP


#Class used inside the dashboard.py
class EDAComponent(DashboardComponent):
    def render(self):
        st.header("Exploratory Data Analysis")
        path = "./data/raw/resume.csv"
        try:
            dataset = pd.read_csv(path)
            st.success("Dataset loaded successfully.")  

            #For showing the dataset overview 
            Dataset_overviewHelperF(dataset)

            #For preprocessing the dataset  
            dataset_cleaned , label_counts_key =Dataset_preprocessingHelperF(dataset  , LABELS_MAP)

            #For showing the distribution of the dataset  
            Dataset_distributionHelperF(dataset , LABELS_MAP)
            
            #For showing the wordsclouds and some stats of the dataset  
            Dataset_wordscloudHelperF(dataset_cleaned , label_counts_key)

            #For showing the advanced stats & analysis of the dataset  
            Dataset_advanced_analysisHelperF(dataset_cleaned)


        except FileNotFoundError:
            st.error("File not found! Check the path.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
