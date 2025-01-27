import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer  
def display_eda_section():
    st.header("Exploratory Data Analysis")
    st.info(
    "In this section, we will explore our resume dataset and extract relevant information "
    "to better understand the trends and characteristics of the different profiles."
        )

    
    # Charger le dataset (remplacer "ton_fichier.csv" par le chemin de ton dataset)
    data = pd.read_csv("C:\\Users\\hp\\Documents\\ML-Internship-Home-Assignment\\data\\raw\\resume.csv")
 
    # Afficher un aperçu du dataset
    st.subheader("Data Overview")
    st.write(data.head())
    
    # Afficher des informations statistiques
    st.subheader("Statistical Summary")
    st.write(data.describe())

    # Exemple de distribution des labels
    st.subheader("Label Distribution")
    # Crée une figure et des axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Trace le graphique avec seaborn
    sns.countplot(x='label', data=data, ax=ax)  # Remplace 'label' par le nom de ta colonne de labels
    ax.set_title("Distribution des Labels")

    # Affiche le graphique dans Streamlit en passant la figure
    st.pyplot(fig)
    
   # Ajouter une colonne de longueur de texte
    data['resume_length'] = data['resume'].apply(lambda x: len(x.split()))

    # Afficher un histogramme de la longueur des CV
    st.subheader("Distribution of Resume Lengths")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['resume_length'], kde=True, ax=ax)
    ax.set_title("Resume Lengths")
    st.pyplot(fig)
                                                                                                                                         
                                                                                                                                           
    # Vectorisation du texte pour extraire les mots les plus fréquents
    vectorizer = CountVectorizer(stop_words='english', max_features=20)
    X = vectorizer.fit_transform(data['resume'])

    # Créer un DataFrame des mots les plus fréquents
    word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    word_sum = word_freq.sum(axis=0).sort_values(ascending=False)

    # Afficher les 10 mots les plus fréquents
    st.subheader("Most Frequent Keywords in the Resumes")
    st.write(word_sum.head(10))                                                                                                                                           