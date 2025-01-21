# **ML Internship Home Assignment**

This repository contains my submission for the ML Internship Home Assignment. The goal of this assignment was to build a resume classification system using machine learning techniques. Below is an overview of the project, including the steps I took, the improvements I made.

---

## **Project Overview**

The project involves building a machine learning pipeline to classify resumes into one of 13 predefined categories (e.g., Java Developer, Business Analyst, etc.). The pipeline includes:

1. **Exploratory Data Analysis (EDA)**: Analyzing the dataset to derive insights and prepare the data for modeling.
2. **Training Pipeline**: Building and training a machine learning model to classify resumes.
3. **Inference**: Using the trained model to predict the category of new resumes.
4. **Streamlit Dashboard**: A user-friendly interface to interact with the EDA, training, and inference components.

---

## **Improvements Made**

### **Baseline Model**
- The baseline model used a **Naive Bayes classifier** with **CountVectorizer**.
- Achieved an **F1 score of 0.867**.

### **Improved Model**
- Replaced the baseline model with a **Logistic Regression classifier** and **TfidfVectorizer**.
- Added **hyperparameter tuning** using `GridSearchCV` to optimize the model.
- Improved the **F1 score to 0.9337**, a significant increase from the baseline.

### **Key Changes**
1. **Feature Engineering**:
   - Used **TF-IDF vectorization** instead of CountVectorizer to better capture the importance of words.
   - Added **n-grams (bi-grams)** to capture word combinations (e.g., "machine learning").
2. **Model Selection**:
   - Switched to **Logistic Regression**, which performed better than Naive Bayes for this dataset.
3. **Hyperparameter Tuning**:
   - Tuned the `C` parameter (regularization strength) and `penalty` (L2 regularization) using `GridSearchCV`.
4. **Error Analysis**:
   - Analyzed the confusion matrix to identify misclassified labels and improve the model's performance.

---

## **Code Refactoring**
- **Modularization**: The Streamlit dashboard was refactored into separate components (`eda.py`, `training.py`, and `inference.py`) for better readability and maintainability.
- **Clean Code**: The code follows PEP 8 guidelines, with proper indentation, variable naming, and docstrings.

---

## **Exploratory Data Analysis (EDA)**
- **Interactive Components**: Added interactive widgets (e.g., sliders, dropdowns) to the EDA section for better user experience.
- **Visualizations**: Enhanced visualizations with bar charts, histograms, box plots, and word clouds.
- **Data Preprocessing**: Added optional text preprocessing (e.g., removing punctuation, converting to lowercase) for better analysis.

---

## **Training Pipeline**
- **Improved Model**: Replaced the baseline `NaiveBayes` model with `LogisticRegression` and `TfidfVectorizer` for better performance.
- **Hyperparameter Tuning**: Added `GridSearchCV` for hyperparameter tuning to optimize model performance.
- **Serialization**: Implemented model serialization using `joblib` to save and load trained pipelines.

---

## **Inference**
- **SQLite Integration**: Added a SQLite database to store prediction results, with a feature to display prediction history.
- **Error Handling**: Added error handling for edge cases (e.g., invalid input, API server down).

---

## **Unit Testing**
- **Test Coverage**: Added unit tests for the `TrainingPipeline` class, covering initialization, training, performance evaluation, and confusion matrix rendering.
- **Mocking**: Used `unittest.mock` to mock external dependencies (e.g., `GridSearchCV`, `LogisticRegression`) for testing.

---

