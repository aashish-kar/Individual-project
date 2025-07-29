# src/models/random_forest.py

from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 10),
            random_state=config.get('random_state', 42)
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)


    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
