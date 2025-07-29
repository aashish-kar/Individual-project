# src/models/gradient_boosting.py

from xgboost import XGBClassifier
from src.models.base_model import BaseModel

class GradientBoostingModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = XGBClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 6),
            learning_rate=config.get('learning_rate', 0.1),
            random_state=config.get('random_state', 42)
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
