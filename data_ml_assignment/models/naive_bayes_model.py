from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from data_ml_assignment.models.base_model import BaseModel


class NaiveBayesModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(
            model=Pipeline(
                [('tfidf', TfidfVectorizer()), ("nbc", MultinomialNB(**kwargs))]
            )
        )
