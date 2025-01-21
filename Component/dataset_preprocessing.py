import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#This function is used inside the eda_component
def Dataset_preprocessingHelperF(dataset , LABELS_MAP):
    """
    Preprocesses the dataset and calculates keyword frequencies based on LABELS_MAP.

    Args:
        dataset (pd.DataFrame): The input dataset containing a 'resume' column.

    Returns:
        pd.DataFrame: Cleaned dataset with a 'cleaned_resume' column.
        dict: A dictionary containing frequencies of keywords from LABELS_MAP.
    """
    # NLTK resource downloads
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Define LABELS_MAP with job titles to track

    keywords = [label.lower() for label in LABELS_MAP.values()]  # Lowercase labels for consistency

    # Initialize tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Function to clean and process individual resumes
    def clean_text(resume):
        tokens = word_tokenize(resume)  # Tokenize
        tokens = [word for word in tokens if word.isalnum()]  # Remove non-alphanumeric tokens
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
        return ' '.join(tokens)

    # Apply cleaning to the 'resume' column
    dataset['resume'] = dataset['resume'].str.replace('\n', ' ').str.strip().str.lower()
    dataset['resume'] = dataset['resume'].fillna('')  # Replace NaN with empty strings
    dataset['cleaned_resume'] = dataset['resume'].apply(clean_text)  # Apply cleaning function

    # Drop duplicates and missing values
    dataset = dataset.drop_duplicates()
    dataset = dataset.dropna()

    # Calculate keyword frequencies
    label_counts_key = {label: 0 for label in LABELS_MAP.values()}
    for resume in dataset['cleaned_resume']:
        for label in LABELS_MAP.values():
            if label.lower() in resume:
                label_counts_key[label] += 1

    return dataset, label_counts_key
###