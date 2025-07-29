# src/models/optimizer.py

from sklearn.model_selection import GridSearchCV
from src.models.model_factory import create_model

class HyperparameterOptimizer:
    def __init__(self, config_path=None):
        # You can load YAML config here if needed
        pass

    def two_stage_optimization(self, X, y, model_type):
        model = create_model(model_type)

        if model_type == "RandomForest":
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5]
            }

        elif model_type == "GradientBoosting":
            param_grid = {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }

        elif model_type == "SVM":
            param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }

        elif model_type == "NeuralNetwork":
            param_grid = {
                "hidden_layer_sizes": [(64,), (64, 64)],
                "activation": ["relu", "tanh"],
                "solver": ["adam"],
                "learning_rate_init": [0.001, 0.01]
            }

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        search = GridSearchCV(model.model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        search.fit(X, y)

        return search.best_params_, search.best_score_
