# src/models/neural_network.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.models.base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Sequential([
            Dense(config.get('units', 64), activation='relu', input_shape=(config.get('input_dim', 30),)),
            Dropout(config.get('dropout_rate', 0.5)),
            Dense(config.get('units', 64), activation='relu'),
            Dropout(config.get('dropout_rate', 0.5)),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(
            optimizer=Adam(learning_rate=config.get('learning_rate', 0.001)),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        epochs = self.config.get('epochs', 10)
        batch_size = self.config.get('batch_size', 32)

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
        else:
            self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        return self.model.predict(X)
