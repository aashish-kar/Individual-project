"""
Feature selection module for football player performance prediction.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

class FeatureSelector:
    """Implements various feature selection techniques."""
    
    def __init__(self, config_path=None):
        """Initialize the feature selector."""
        self.logger = logging.getLogger(__name__)
        
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
    
    def select_k_best(self, X, y, k=30, classification=True):
        """
        Select top k features using univariate statistical tests.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target vector
        k : int
            Number of features to select
        classification : bool
            Whether this is a classification task
            
        Returns:
        --------
        tuple
            Selected features matrix and feature names
        """
        self.logger.info(f"Selecting top {k} features using univariate tests")
        
        # Choose scoring function based on problem type
        score_func = f_classif if classification else f_regression
        
        # Create selector and fit
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        # Get feature importance scores
        scores = selector.scores_
        feature_scores = [(feature, score) for feature, score in zip(X.columns, scores)]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Selected {len(selected_features)} features using SelectKBest")
        return X.iloc[:, selected_indices], feature_scores[:k]
    
    def recursive_feature_elimination(self, X, y, n_features=30):
        """
        Select features using Recursive Feature Elimination.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target vector
        n_features : int
            Number of features to select
            
        Returns:
        --------
        tuple
            Selected features matrix and feature names
        """
        self.logger.info(f"Selecting {n_features} features using RFE")
        
        # Create estimator for RFE
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create RFE selector
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector.fit(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        # Get feature importance
        feature_ranks = selector.ranking_
        feature_scores = [(feature, -rank) for feature, rank in zip(X.columns, feature_ranks)]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Selected {len(selected_features)} features using RFE")
        return X.iloc[:, selected_indices], feature_scores[:n_features]
    
    def pca_selection(self, X, n_components=30):
        """
        Select features using Principal Component Analysis.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        n_components : int
            Number of components to select
            
        Returns:
        --------
        tuple
            Transformed feature matrix and component names
        """
        self.logger.info(f"Selecting {n_components} components using PCA")
        
        # Standardize the data
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create component names
        component_names = [f"PC{i+1}" for i in range(n_components)]
        
        # Create DataFrame with component names
        X_pca_df = pd.DataFrame(X_pca, columns=component_names, index=X.index)
        
        # Get variance explained by each component
        explained_variance = pca.explained_variance_ratio_
        component_scores = [(name, var) for name, var in zip(component_names, explained_variance)]
        
        self.logger.info(f"Selected {n_components} components using PCA, explaining {sum(explained_variance)*100:.2f}% of variance")
        return X_pca_df, component_scores
    
    def cross_validated_feature_selection(self, X, y, method='SelectKBest', n_features=30):
        """
        Select features using cross-validation to find optimal feature set.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target vector
        method : str
            Feature selection method ('SelectKBest', 'RFE', or 'PCA')
        n_features : int
            Number of features to select
            
        Returns:
        --------
        tuple
            Selected features matrix and feature information
        """
        self.logger.info(f"Performing {method} feature selection with cross-validation")
        
        if method == 'SelectKBest':
            X_selected, feature_info = self.select_k_best(X, y, k=n_features)
        
        elif method == 'RFE':
            X_selected, feature_info = self.recursive_feature_elimination(X, y, n_features=n_features)
        
        elif method == 'PCA':
            X_selected, feature_info = self.pca_selection(X, n_components=n_features)
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Evaluate feature set with cross-validation
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(estimator, X_selected, y, cv=5, scoring='f1')
        
        self.logger.info(f"Cross-validated F1 score with selected features: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return X_selected, feature_info