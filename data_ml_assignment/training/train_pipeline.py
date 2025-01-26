import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC



class NaiveBayesPipeline:
    def __init__(self):
        self.model = None
        self.vectorizer = CountVectorizer()  # Using CountVectorizer as an example
        self.cm_plot_path = "static/confusion_matrix.png"
        self.model_save_path = "C:/Users/hp/Documents/ML-Internship-Home-Assignment/models"
        self.f1 = None  # Add f1 attribute to store the F1 score

    def load_data(self, data_path):
        # Load the resume dataset
        try:
            data = pd.read_csv(data_path)
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    def preprocess_data(self, data):
        # Extract features and labels
        X = data['resume']  # Assuming 'resume' column contains text
        y = data['label']   # Assuming 'label' column contains labels
        return X, y

    def train(self, serialize=False, model_name="Naive Bayes"):
        # Load and preprocess data
        data = self.load_data("C:\\Users\\hp\\Documents\\ML-Internship-Home-Assignment\\data\\raw\\resume.csv")
        if data is None or data.empty:  # Check if the data is None or empty
            st.error("Data is empty or could not be loaded.")
            return None, None  # Exit early if data couldn't be loaded or is empty
        
        X, y = self.preprocess_data(data)
        
        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Vectorize text data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train the model (Naive Bayes as an example)
        self.model = MultinomialNB()
        self.model.fit(X_train_vec, y_train)

        # Predict and evaluate
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        self.f1 = f1_score(y_test, y_pred, average='weighted')  # Store the F1 score

        # Optionally serialize the model
        if isinstance(serialize, bool) and serialize:
            self.serialize_model(model_name)

        # Save and plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return accuracy, self.f1  # Return both accuracy and F1 score


    def plot_confusion_matrix(self, y_test, y_pred):
        # Créer la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)

        # Obtenir les étiquettes des classes
        class_labels = sorted(list(set(y_test)))  # Récupère les classes à partir de y_test

        # Tracer la matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Sauvegarder et afficher l'image
        os.makedirs(os.path.dirname(self.cm_plot_path), exist_ok=True)
        plt.savefig(self.cm_plot_path)
        plt.close()

    def render_confusion_matrix(self):
        # Display the confusion matrix plot in the Streamlit interface
        if os.path.exists(self.cm_plot_path):
            st.image(self.cm_plot_path, use_column_width=True)
        else:
            st.error("Confusion matrix image not found!")

    def serialize_model(self, model_name):
        # Serialize and save the model
        model_filepath = os.path.join(self.model_save_path, f"{model_name}_model.pkl")
        os.makedirs(self.model_save_path, exist_ok=True)
        joblib.dump(self.model, model_filepath)
        st.success(f"Model serialized and saved at {model_filepath}")
    
    def get_model_performance(self):
        # Return the model performance metrics (accuracy and F1 score)
        return self.model, self.f1  # Return the f1 score stored during training



class SVMPipeline:
    def __init__(self):
        self.model = None
        self.vectorizer = CountVectorizer()  # Utilisation du CountVectorizer pour la vectorisation
        self.cm_plot_path = "static/confusion_matrix_svm.png"  # Emplacement de l'image de la matrice de confusion
        self.model_save_path = "C:/Users/hp/Documents/ML-Internship-Home-Assignment/models"  # Répertoire de sauvegarde du modèle
        self.f1 = None  # Ajouter l'attribut f1 pour stocker le score F1

    def load_data(self, data_path):
        try:
            data = pd.read_csv(data_path)
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    def preprocess_data(self, data):
        X = data['resume']  # Colonne 'resume' qui contient le texte
        y = data['label']   # Colonne 'label' qui contient les étiquettes
        return X, y

    def train(self, serialize=False, model_name="SVM"):
        # Charger et prétraiter les données
        data = self.load_data("C:\\Users\\hp\\Documents\\ML-Internship-Home-Assignment\\data\\raw\\resume.csv")
        if data is None or data.empty:
            st.error("Data is empty or could not be loaded.")
            return None, None  # Quitter si les données sont vides ou introuvables

        X, y = self.preprocess_data(data)

        # Diviser les données en jeu d'entraînement et jeu de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Vectoriser les données textuelles
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Entraîner le modèle SVM
        self.model = SVC(kernel='linear')  # Utilisation du noyau linéaire pour un modèle SVM simple
        self.model.fit(X_train_vec, y_train)

        # Faire des prédictions et évaluer le modèle
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        self.f1 = f1_score(y_test, y_pred, average='weighted')  # Calcul du score F1

        # Sauvegarder le modèle si nécessaire
        if isinstance(serialize, bool) and serialize:
            self.serialize_model(model_name)

        # Sauvegarder et afficher la matrice de confusion
        self.plot_confusion_matrix(y_test, y_pred)

        return accuracy, self.f1  # Retourner la précision et le score F1

    def plot_confusion_matrix(self, y_test, y_pred):
        # Créer la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)

        # Obtenir les étiquettes des classes
        class_labels = sorted(list(set(y_test)))  # Récupère les classes à partir de y_test

        # Tracer la matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Sauvegarder l'image de la matrice de confusion
        os.makedirs(os.path.dirname(self.cm_plot_path), exist_ok=True)
        plt.savefig(self.cm_plot_path)
        plt.close()

    def render_confusion_matrix(self):
        # Afficher la matrice de confusion dans l'interface Streamlit
        if os.path.exists(self.cm_plot_path):
            st.image(self.cm_plot_path, use_column_width=True)
        else:
            st.error("Confusion matrix image not found!")

    def serialize_model(self, model_name):
        # Sérialiser et sauvegarder le modèle
        model_filepath = os.path.join(self.model_save_path, f"{model_name}_model.pkl")
        os.makedirs(self.model_save_path, exist_ok=True)
        joblib.dump(self.model, model_filepath)
        st.success(f"Model serialized and saved at {model_filepath}")
    
    def get_model_performance(self):
        # Retourner les métriques de performance du modèle
        return self.model, self.f1

