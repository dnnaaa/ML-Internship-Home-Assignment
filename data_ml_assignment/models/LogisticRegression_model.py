from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
import re
from data_ml_assignment.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        def preprocess_text(text):
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = ' '.join(text.split())
            return text
            
        super().__init__(
            model=Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,  
                    min_df=5,
                    max_df=0.95,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents='unicode',
                    norm='l2',
                    stop_words='english'
                )),
                
                ('logistic_regression', LogisticRegression(
                    penalty='l2',
                    C=1.0,  # Regularization strength
                    solver='lbfgs',
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=42,
                    **kwargs
                ))
            ])
        )
