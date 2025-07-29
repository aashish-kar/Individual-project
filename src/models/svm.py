# src/models/svm.py

from sklearn.svm import SVC
from src.models.base_model import BaseModel

class SVMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = SVC(
            C=config.get('C', 1.0),
            kernel=config.get('kernel', 'rbf'),
            gamma=config.get('gamma', 'scale'),
            probability=True,
            random_state=config.get('random_state', 42)
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
