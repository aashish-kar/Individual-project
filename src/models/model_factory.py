# src/models/model_factory.py

from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import GradientBoostingModel
from src.models.svm import SVMModel
from src.models.NeuralNetwork import NeuralNetworkModel

def create_model(model_type, config):
    if model_type == 'RandomForest':
        return RandomForestModel(config)
    elif model_type == 'GradientBoosting':
        return GradientBoostingModel(config)
    elif model_type == 'SVM':
        return SVMModel(config)
    elif model_type == 'NeuralNetwork':
        return NeuralNetworkModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
