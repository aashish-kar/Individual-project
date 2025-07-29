"""
Data preprocessing module for football player performance prediction.
"""

import pandas as pd
import numpy as np
import yaml
import logging

class DataPreprocessor:
    """Handles preprocessing of the FPL dataset."""
    
    def __init__(self, config_path="config/system_config.yaml"):
        """Initialize the preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.logger = logging.getLogger(__name__)
    
    def preprocess_data(self, df, target_column='FPL_points', classification_threshold=5):
        """
        Preprocess the data for machine learning.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The raw dataset
        target_column : str
            Column name of the target variable
        classification_threshold : float
            Threshold for binary classification of player performance
            
        Returns:
        --------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("Starting data preprocessing")
        
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # 1. Filter for players with playing time
        data = data[data['Min'] > 0]
        self.logger.info(f"Filtered rows with playing time > 0: {data.shape[0]} rows remaining")
        
        # 2. Handle missing values
        data = self._handle_missing_values(data)
        
        # 3. Create target variable
        if target_column in data.columns:
            self.logger.info(f"Creating binary target variable with threshold {classification_threshold}")
            y = (data[target_column] >= classification_threshold).astype(int)
            self.logger.info(f"Target variable distribution: {y.value_counts().to_dict()}")
        else:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # 4. Drop non-feature columns
        X = data.drop(columns=self._get_columns_to_drop(data, target_column), errors='ignore')
        
        # 5. Transform features
        X = self._transform_features(X)
        
        # 6. Split the data temporally
        X_train, X_val, X_test, y_train, y_val, y_test = self._temporal_split(X, y)
        
        self.logger.info("Data preprocessing completed")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataset with missing values
            
        Returns:
        --------
        pandas.DataFrame
            The dataset with handled missing values
        """
        self.logger.info("Handling missing values")
        
        # For numerical columns: fill missing values with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                self.logger.debug(f"Filled {col} missing values with median")
        
        # For categorical columns: fill missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
                self.logger.debug(f"Filled {col} missing values with mode")
        
        # Special handling for expected metrics
        for col in ['xG', 'xA']:
            if col in df.columns and df[col].isnull().sum() > 0:
                # For xG, use a relationship with shots if available
                if col == 'xG' and 'Sh' in df.columns:
                    df[col] = df[col].fillna(df['Sh'] * 0.1)  # Rough approximation
                    self.logger.debug(f"Filled {col} using shots relationship")
                else:
                    df[col] = df[col].fillna(0)
                    self.logger.debug(f"Filled {col} with 0")
        
        self.logger.info(f"Missing values handled: {df.isnull().sum().sum()} remaining")
        return df
    
    def _get_columns_to_drop(self, df, target_column):
        """
        Get list of columns to drop from the feature set.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataset
        target_column : str
            Name of the target column
            
        Returns:
        --------
        list
            List of columns to drop
        """
        # Default columns to drop
        cols_to_drop = [
            target_column,  # Target column
            'Date', 'Day', 'Kickoff_time',  # Date-related columns
            'FPL_name', 'Clean_name', 'Name', 'Name_original',  # Name columns
            'Element', 'Fixture'  # Identifiers
        ]
        
        # Only include columns that exist in the dataframe
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        self.logger.info(f"Dropping {len(cols_to_drop)} non-feature columns")
        return cols_to_drop
    
    def _transform_features(self, X):
        """
        Transform features for machine learning.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
            
        Returns:
        --------
        pandas.DataFrame
            Transformed features
        """
        self.logger.info("Transforming features")
        
        # Handle categorical variables
        X = self._encode_categorical_variables(X)
        
        # Handle outliers
        X = self._handle_outliers(X)
        
        # Log transformation for highly skewed features
        skewed_features = ['Gls', 'Ast', 'xG', 'xA']
        for feature in skewed_features:
            if feature in X.columns:
                # Add 1 to handle zeros before log transform
                X[f'log_{feature}'] = np.log1p(X[feature])
                self.logger.debug(f"Log-transformed {feature}")
        
        self.logger.info(f"Feature transformation complete: {X.shape[1]} features")
        return X
    
    def _encode_categorical_variables(self, X):
        """
        Encode categorical variables for machine learning.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with encoded categorical variables
        """
        # Identify categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_columns:
            return X
        
        self.logger.info(f"Encoding {len(categorical_columns)} categorical variables")
        
        # For categorical columns with low cardinality, use one-hot encoding
        # For high cardinality columns, use label encoding
        for col in categorical_columns:
            if X[col].nunique() < 10:  # Low cardinality
                # One-hot encode
                n_categories = X[col].nunique()
                one_hot = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(col, axis=1), one_hot], axis=1)
                self.logger.debug(f"One-hot encoded {col} with {n_categories} categories")

            else:  # High cardinality
                # Label encode
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.logger.debug(f"Label encoded {col} with {X[col].nunique()} categories")
        
        return X
    
    def _handle_outliers(self, X):
        """
        Handle outliers in numerical features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with handled outliers
        """
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Calculate Z-scores
            z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
            
            # Identify outliers (Z-score > 3)
            outliers = z_scores > 3
            
            if outliers.sum() > 0:
                # Apply winsorization: cap values at 3 standard deviations
                upper_bound = X[col].mean() + 3 * X[col].std()
                lower_bound = X[col].mean() - 3 * X[col].std()
                
                X.loc[X[col] > upper_bound, col] = upper_bound
                X.loc[X[col] < lower_bound, col] = lower_bound
                
                self.logger.debug(f"Winsorized {outliers.sum()} outliers in {col}")
        
        return X
    
    def _temporal_split(self, X, y):
        """
        Split the data temporally for time-series validation.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        y : pandas.Series
            The target variable
            
        Returns:
        --------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # If Season column exists, use it for temporal split
        if 'Season' in X.columns:
            # Sort by Season and GW if available
            if 'GW' in X.columns:
                sorted_indices = X.sort_values(['Season', 'GW']).index
            else:
                sorted_indices = X.sort_values(['Season']).index
            
            X = X.loc[sorted_indices]
            y = y.loc[sorted_indices]
            
            # Get unique seasons
            seasons = X['Season'].unique()
            
            if len(seasons) >= 3:
                # Use earlier seasons for training, second-to-last for validation,
                # and the last season for testing
                train_seasons = seasons[:-2]
                val_season = seasons[-2]
                test_season = seasons[-1]
                
                train_mask = X['Season'].isin(train_seasons)
                val_mask = X['Season'] == val_season
                test_mask = X['Season'] == test_season
                
                X_train, y_train = X[train_mask], y[train_mask]
                X_val, y_val = X[val_mask], y[val_mask]
                X_test, y_test = X[test_mask], y[test_mask]
                
                self.logger.info(f"Temporal split: {len(train_seasons)} seasons for training, {val_season} for validation, {test_season} for testing")
                self.logger.info(f"Split sizes: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
                
                return X_train, X_val, X_test, y_train, y_val, y_test
        
        # Default: split 80% train, 10% validation, 10% test
        n = len(X)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        
        X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
        X_val, y_val = X.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
        X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]
        
        self.logger.info(f"Default temporal split (80/10/10): Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test