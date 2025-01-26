#! Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from imblearn.combine import SMOTETomek
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#! Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

#! Text preprocessing function
def preprocess_text(text):
    """
    Preprocess text by:
    - Lowercasing
    - Removing digits
    - Removing punctuation
    - Removing stopwords
    - Lemmatization
    """
    # Lowercase
    text = text.lower()
    # Remove digits
    text = ''.join(char for char in text if not char.isdigit())
    # Remove punctuation
    text = ''.join(char for char in text if char not in string.punctuation)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

#! Main training function
def render_training_section(data_path):
    """
    Handle model training, evaluation, and visualization.
    """

    #! 1) Initialize session state
    if 'trained_model' not in st.session_state:
        st.session_state['trained_model'] = None
    if 'trained_vectorizer' not in st.session_state:
        st.session_state['trained_vectorizer'] = None
    if 'best_model_name' not in st.session_state:
        st.session_state['best_model_name'] = None
    if 'best_score' not in st.session_state:
        st.session_state['best_score'] = -1.0
    if 'scaler' not in st.session_state:
        st.session_state['scaler'] = None
    if 'svd' not in st.session_state:
        st.session_state['svd'] = None

    #! 2) Training button
    if st.button("Train Model"):
        # Load data
        data = pd.read_csv(data_path)
        X = data['resume'].fillna("").apply(preprocess_text)
        y = data['label']

        st.subheader("Class Distribution Before Balancing")
        class_counts = y.value_counts().sort_index()
        st.bar_chart(class_counts)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
            max_features=3000,
            stop_words='english'
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Dimensionality reduction with TruncatedSVD
        svd = TruncatedSVD(n_components=100, random_state=42)
        X_train_vec_svd = svd.fit_transform(X_train_vec)
        X_test_vec_svd = svd.transform(X_test_vec)

        # Normalize data
        scaler = StandardScaler()
        X_train_vec_scaled = scaler.fit_transform(X_train_vec_svd)
        X_test_vec_scaled = scaler.transform(X_test_vec_svd)

        # SMOTETomek for balancing
        smote_tomek = SMOTETomek(random_state=42)
        X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_vec_scaled, y_train)

        st.subheader("Class Distribution After Balancing")
        balanced_class_counts = pd.Series(y_train_resampled).value_counts().sort_index()
        st.bar_chart(balanced_class_counts)

        # Define models and parameters
        models_and_params = {
            "Naive Bayes": (
                MultinomialNB(),
                {
                    "alpha": [0.1, 0.5, 5.0]
                }
            ),
            "Logistic Regression": (
                LogisticRegression(
                    class_weight='balanced',
                    max_iter=400,
                    random_state=42,
                    solver='sag'
                ),
                {
                    "C": [0.1, 1, 10],
                    "penalty": ['l2']
                }
            ),
            "Random Forest": (
                RandomForestClassifier(
                    class_weight='balanced',
                    random_state=42
                ),
                {
                    "n_estimators": [100, 400],
                    "max_depth": [10, None],
                    "min_samples_split": [2, 8],
                    "min_samples_leaf": [1, 4],
                    "bootstrap": [True],
                    "max_features": ["sqrt"]
                }
            ),
            "Support Vector Machine": (
                SVC(
                    class_weight='balanced',
                    probability=True,
                    random_state=42
                ),
                {
                    "C": [0.1, 1, 10],
                    "kernel": ['linear', 'rbf']
                }
            )
        }

        # Separate data for Naive Bayes
        X_train_nb, y_train_nb = smote_tomek.fit_resample(X_train_vec, y_train)

        best_model = None
        best_score = -1.0
        best_model_name = None
        results = []

        #! 3) Training with GridSearch
        for model_name, (model, param_grid) in models_and_params.items():
            st.write(f"**Training {model_name}...**")
            if model_name == "Naive Bayes":
                # Use non-scaled data for NB
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=StratifiedKFold(n_splits=5),
                    scoring='f1_macro',
                    n_jobs=-1
                )
                grid_search.fit(X_train_nb, y_train_nb)
            else:
                # Use scaled data for others
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=StratifiedKFold(n_splits=5),
                    scoring='f1_macro',
                    n_jobs=-1
                )
                grid_search.fit(X_train_resampled, y_train_resampled)

            mean_cv_score = grid_search.best_score_
            results.append((model_name, grid_search.best_params_, mean_cv_score))

            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_model = grid_search.best_estimator_
                best_model_name = model_name

        #! 4) Evaluation on test set
        st.subheader("Model Comparison")
        for model_name, params, score in results:
            st.write(f"- {model_name}: best_params={params}, CV f1_macro={score:.4f}")

        st.subheader("Best Model Evaluation")
        st.write(f"**Selected model**: {best_model_name} (CV f1={best_score:.4f})")

        # Predict
        if best_model_name == "Naive Bayes":
            y_pred = best_model.predict(X_test_vec)
        else:
            y_pred = best_model.predict(X_test_vec_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1_val = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.write(f"**Test Accuracy**: {acc:.4f}")
        st.write(f"**Test F1 Score (weighted)**: {f1_val:.4f}")
        st.text("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        plt.figure(figsize=(6, 4))
        sorted_labels = sorted(y.unique())
        sns.heatmap(
            conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted_labels, yticklabels=sorted_labels
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(plt)

        #! 5) Store trained model and objects in session state
        st.session_state['trained_model'] = best_model
        st.session_state['trained_vectorizer'] = vectorizer
        st.session_state['best_model_name'] = best_model_name
        st.session_state['best_score'] = best_score
        st.session_state['scaler'] = scaler
        st.session_state['svd'] = svd

        st.success(
            f"Training complete. Best model: {best_model_name} with CV F1={best_score:.4f}"
        )

    #! 6) Display "Save" button if a model is trained
    if st.session_state.get('trained_model') is not None:
        st.subheader("Save Best Model and Vectorizer")

        pipeline_name = st.text_input("Enter Pipeline Name", value="best_pipeline")

        if st.button("Save"):
            if pipeline_name.strip() == "":
                st.error("Please enter a valid pipeline name.")
            else:
                valid_name = re.sub(r'[^a-zA-Z0-9_-]', '_', pipeline_name.strip())
                models_dir = "models"
                os.makedirs(models_dir, exist_ok=True)

                vectorizer_path = os.path.join(models_dir, f"{valid_name}_vectorizer.pkl")
                model_path = os.path.join(models_dir, f"{valid_name}_model.pkl")
                scaler_path = os.path.join(models_dir, f"{valid_name}_scaler.pkl")
                svd_path = os.path.join(models_dir, f"{valid_name}_svd.pkl")

                # Save vectorizer
                with open(vectorizer_path, "wb") as vec_file:
                    pickle.dump(st.session_state['trained_vectorizer'], vec_file)

                # Save model
                with open(model_path, "wb") as model_file:
                    pickle.dump(st.session_state['trained_model'], model_file)

                # Save scaler
                with open(scaler_path, "wb") as scaler_file:
                    pickle.dump(st.session_state['scaler'], scaler_file)

                # Save SVD
                with open(svd_path, "wb") as svd_file:
                    pickle.dump(st.session_state['svd'], svd_file)

                st.success(
                    f"Best model saved as '{model_path}', vectorizer '{vectorizer_path}', "
                    f"scaler '{scaler_path}' and SVD '{svd_path}'."
                )

    else:
        st.info("No trained model available yet. Please train the model first.")

#! Main
if __name__ == "__main__":
    st.title("Demo - Resume Classification Training")
    data_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if data_file:
        render_training_section(data_file)
    else:
        st.warning("Please upload a CSV file to proceed.")
