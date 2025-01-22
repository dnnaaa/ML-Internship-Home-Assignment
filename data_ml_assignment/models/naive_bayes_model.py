from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from .base_model import BaseModel

class NaiveBayesModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(
            model=Pipeline([
                ("vectorizer", CountVectorizer()),
                ("classifier", MultinomialNB(**kwargs))
            ])
        )