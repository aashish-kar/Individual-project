"""
Data loading module for football player performance prediction.
"""

import pandas as pd
import numpy as np
import yaml
import logging

class DataLoader:
    """Handles loading and initial validation of the FPL dataset."""
    
    def __init__(self, config_path="config/system_config.yaml"):
        """Initialize the data loader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path=None):
        """
        Load the FPL dataset from the specified path.
        
        Parameters:
        -----------
        file_path : str, optional
            Path to the dataset file. If None, uses the path from config.
            
        Returns:
        --------
        pandas.DataFrame
            The loaded dataset
        """
        if file_path is None:
            file_path = self.config['data_paths']['raw_data']
        
        self.logger.info(f"Loading data from {file_path}")
        try:
            # Load the dataset
            df = pd.read_csv(file_path)
            
            # Perform basic validation
            self._validate_data(df)
            
            # Log summary statistics
            self.logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            self.logger.info(f"Columns: {df.columns.tolist()[:10]}... (and {len(df.columns)-10} more)")
            self.logger.info(f"Data types: {df.dtypes.value_counts().to_dict()}")
            self.logger.info(f"Missing values: {df.isnull().sum().sum()} total missing values")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_data(self, df):
        """
        Validate that the loaded data meets basic requirements.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataset to validate
        """
        # Check that required columns are present
        required_cols = self.config.get('required_columns', ['Min', 'FPL_points'])
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
        # Check that dataset is not empty
        if df.empty:
            self.logger.error("Dataset is empty")
            raise ValueError("Dataset is empty")
        
        # Check for basic data integrity issues
        if df['Min'].isnull().sum() > df.shape[0] * 0.5:
            self.logger.warning("More than 50% of 'Min' values are missing")
        
        self.logger.info("Data validation passed")