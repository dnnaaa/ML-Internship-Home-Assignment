import re


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from data_ml_assignment.models.base_model import BaseModel


# class NaiveBayesModel(BaseModel):
#     def __init__(self, **kwargs):
#         super().__init__(
#             model=Pipeline(
#                 [("countv", CountVectorizer()), ("nbc", MultinomialNB(**kwargs))]
#             )
#         )


class NaiveBayesModel(BaseModel):
    def __init__(self, **kwargs):
        def preprocess_text(text):
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r'[^a-z\s]', ' ', text)
            text = ' '.join(text.split())
            return text
            
        super().__init__(
            model=Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=27000,  # Increased features for better representation
                    min_df=5, 
                    max_df=0.90, 
                    ngram_range=(1, 3),  
                    sublinear_tf=True,  
                    stop_words='english',  
                    strip_accents='unicode',
                    norm='l2',  
                )),
                ('nb', MultinomialNB(
                    alpha=0.001,  
                    fit_prior=True,  
                    **kwargs
                ))
            ])
        )

