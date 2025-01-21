

from xgboost import XGBClassifier  # Import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
import re
from data_ml_assignment.models.base_model import BaseModel


# class XGBCModel(BaseModel):
#     def __init__(self, **kwargs):
#         super().__init__(model=Pipeline([("xgbc", xgb.XGBClassifier(**kwargs))]))


class XGBCModel(BaseModel):  
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
                
                ('xgbc', XGBClassifier(  # Replace LogisticRegression with XGBClassifier
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    **kwargs
                ))
            ])
        )