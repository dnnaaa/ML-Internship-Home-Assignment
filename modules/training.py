# Import necessary libraries for data processing and modeling
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTETomek
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#! Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

#! Function to preprocess text data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove digits
    text = ''.join([char for char in text if not char.isdigit()])
    # Tokenize text
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Rejoin tokens into a single string
    return ' '.join(tokens)

#! Function to train the model and display results
def render_training_section(data_path):
    # Load and preprocess data
    data = pd.read_csv(data_path)
    X = data['resume'].fillna("").apply(preprocess_text)  # Clean the text data
    y = data['label']

    # Display class distribution before balancing
    st.subheader("Class Distribution Before Balancing")
    class_counts = y.value_counts().sort_index()
    st.bar_chart(class_counts)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Transform text data using TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=10000,
        stop_words='english'
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Balance the training data using SMOTETomek
    smote_tomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_vec, y_train)

    # Display class distribution after balancing
    st.subheader("Class Distribution After Balancing")
    balanced_class_counts = pd.Series(y_train_resampled).value_counts().sort_index()
    st.bar_chart(balanced_class_counts)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_resampled),
        y=y_train_resampled
    )
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train_resampled), class_weights)}

    # Hyperparameter tuning -> GridSearchCV
    st.subheader("Hyperparameter Tuning")
    param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
    nb_model = MultinomialNB()
    grid_search = GridSearchCV(
        nb_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
    )
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Best hyperparameter
    st.write(f"Best alpha: {grid_search.best_params_['alpha']}")
    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Evaluation metrics
    st.subheader("Evaluation Metrics")
    st.write(f"**Accuracy**: {acc:.4f}")
    st.write(f"**F1 Score**: {f1:.4f}")
    st.text("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    st.subheader("Confusion Matrix")
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique())
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    st.pyplot(plt)

    # Section pour sauvegarder le modèle et le vectorizer avec un nom personnalisé
    st.subheader("Save Model and Vectorizer")
    
    # Champ de saisie pour le nom du pipeline
    pipeline_name = st.text_input("Enter Pipeline Name", value="naive_bayes_pipeline")
    
    # Bouton de sauvegarde
    if st.button("Save"):
        if pipeline_name.strip() == "":
            st.error("Please enter a valid pipeline name.")
        else:
            # Nettoyer le nom pour éviter les caractères invalides
            valid_name = re.sub(r'[^a-zA-Z0-9_-]', '_', pipeline_name.strip())
            
            # Chemin du dossier models
            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)
            
            # Chemins complets des fichiers
            vectorizer_path = os.path.join(models_dir, f"{valid_name}_vectorizer.pkl")
            model_path = os.path.join(models_dir, f"{valid_name}_model.pkl")
            
            # Sauvegarder le vectorizer
            with open(vectorizer_path, "wb") as vec_file:
                pickle.dump(vectorizer, vec_file)
            
            # Sauvegarder le modèle
            with open(model_path, "wb") as model_file:
                pickle.dump(best_model, model_file)
            
            st.success(f"Training complete. Models saved as '{vectorizer_path}' and '{model_path}'.")

    return best_model, vectorizer

#! Main Streamlit application
if __name__ == "__main__":
    st.title("Resume Classification Training Pipeline")
    data_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if data_file:
        render_training_section(data_file)
