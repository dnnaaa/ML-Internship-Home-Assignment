from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

#This function is used inside the eda_component
def Dataset_advanced_analysisHelperF(dataset): ### For Data
    # PCA
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(dataset['cleaned_resume'])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    # K-Means Clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X)
    dataset['cluster'] = labels

    # Visualize PCA
    st.subheader("PCA Visualization")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=50)
    plt.title("PCA and K-Means Clustering")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(plt)
###
