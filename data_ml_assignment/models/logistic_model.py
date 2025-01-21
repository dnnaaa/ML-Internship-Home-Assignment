from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from data_ml_assignment.models.base_model import BaseModel


class LogisticModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(
            model=Pipeline(
                [
                    ("tfidf", TfidfVectorizer(stop_words='english', max_features=5000)),
                    ("logreg", LogisticRegression(**kwargs))
                ]
            )
        )