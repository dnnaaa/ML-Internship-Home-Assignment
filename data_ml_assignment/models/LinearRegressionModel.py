from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from data_ml_assignment.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(
            model=Pipeline(
                [("countv", CountVectorizer()), ("logreg", LogisticRegression(**kwargs))]
            )
        )