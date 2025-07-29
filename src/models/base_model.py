"""
Base model class for all football performance prediction models.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

class BaseModel:
    """Base class for all football performance prediction models."""
    
    def __init__(self, model_name, config_path=None):
        """
        Initialize the model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        config_path : str, optional
            Path to configuration file
        """
        self.model_name = model_name
        self.model = None
        self.hyperparams = {}
        self.logger = logging.getLogger(f"model.{model_name}")
        
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                if 'models' in self.config and model_name.lower() in self.config['models']:
                    self.hyperparams = self.config['models'][model_name.lower()].get('hyperparameters', {})
        else:
            self.config = {}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model on training data.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training feature matrix
        y_train : pandas.Series
            Training target vector
        X_val : pandas.DataFrame, optional
            Validation feature matrix
        y_val : pandas.Series, optional
            Validation target vector
            
        Returns:
        --------
        self
            Trained model instance
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
            
        Returns:
        --------
        numpy.ndarray
            Predicted classes
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Make probability predictions on new data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
            
        Returns:
        --------
        numpy.ndarray
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, create a simple approximation
            y_pred = self.predict(X)
            probas = np.zeros((len(y_pred), 2))
            probas[np.arange(len(y_pred)), y_pred] = 1.0
            return probas
    
    def evaluate(self, X, y):
        """
        Evaluate model performance on test data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Test feature matrix
        y : pandas.Series
            Test target vector
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X)
        
        # Handle binary or single-class probability shape
        y_pred_proba_all = self.predict_proba(X)
        y_pred_proba = y_pred_proba_all[:, 1] if y_pred_proba_all.shape[1] > 1 else y_pred_proba_all[:, 0]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        # Log detailed classification report
        class_report = classification_report(y, y_pred)
        self.logger.info(f"Classification report:\n{class_report}")
        
        # Log confusion matrix
        cm = confusion_matrix(y, y_pred)
        self.logger.info(f"Confusion matrix:\n{cm}")
        
        return metrics

    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
            
        Returns:
        --------
        str
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load(self, filepath):
        """
        Load a pre-trained model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        self
            Model instance with loaded model
        """
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance if the model supports it.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature names and importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            self.logger.warning(f"{self.model_name} does not provide feature importances")
            return None
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        return feature_importance