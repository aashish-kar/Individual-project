"""
Football Player Performance Prediction Using Machine Learning
===================================================
Main execution script for training and evaluating models.

Author: [Your Name]
Date: April 2025
"""

import argparse
import logging
import yaml
import os
from datetime import datetime

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.selector import FeatureSelector
from src.models.model_factory import create_model
from src.models.optimizer import HyperparameterOptimizer
from src.evaluation.visualizer import Visualizer
from src.utils.logger import setup_logging

def main(args):
    """Main execution function for football performance prediction"""
    # Set up logging
    log_filename = f"football_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_filename)
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Data Loading
        logger.info("1. Starting data loading")
        data_loader = DataLoader(config_path=args.config)
        df = data_loader.load_data(args.data_path)
        
        # 2. Data Preprocessing
        logger.info("2. Starting data preprocessing")
        preprocessor = DataPreprocessor(config_path=args.config)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(
            df, 
            target_column=args.target_column,
            classification_threshold=args.threshold
        )
        
        # 3. Feature Selection
        if not args.skip_feature_selection:
            logger.info("3. Starting feature selection")
            feature_selector = FeatureSelector(config_path=args.config)
            X_train, feature_info = feature_selector.cross_validated_feature_selection(
                X_train, y_train, 
                method=args.feature_selection,
                n_features=args.n_features
            )
            
            # Apply same feature selection to validation and test sets
            selected_columns = X_train.columns.tolist()
            X_val = X_val[selected_columns]
            X_test = X_test[selected_columns]


            
            # Save selected features for the prediction UI
            features_path = os.path.join(args.output_dir, "selected_features.joblib")
            import joblib
            joblib.dump(selected_columns, features_path)
            logger.info(f"Saved selected features to {features_path} for prediction UI")
        
        # 4. Model Training and Evaluation
        logger.info("4. Starting model training and evaluation")
        metrics_dict = {}
        model_types = ['RandomForest', 'SVM', 'NeuralNetwork', 'GradientBoosting']
        
        if args.skip_models:
            for model_name in args.skip_models:
                if model_name in model_types:
                    model_types.remove(model_name)
                    logger.info(f"Skipping {model_name} model as requested")
        
        model_key_map = {
            'RandomForest': 'random_forest',
            'GradientBoosting': 'gradient_boosting',
            'SVM': 'svm',
            'NeuralNetwork': 'neural_network'
        }

        for model_type in model_types:
            logger.info(f"Training and evaluating {model_type} model")

            # Load config and get the correct model config
            with open(args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
                model_key = model_key_map[model_type]
                model = create_model(model_type, config_dict['models'][model_key]['hyperparameters'])

            # Optimize hyperparameters if requested
            if args.optimize:
                logger.info(f"Optimizing hyperparameters for {model_type}")
                optimizer = HyperparameterOptimizer(config_path=args.config)
                best_params, _ = optimizer.two_stage_optimization(
                    X_train, y_train, model_type
                )
                model.hyperparams = best_params

            # Train model
            model.fit(X_train, y_train, X_val, y_val)

            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            metrics_dict[model_type] = metrics
            # Save trained model to disk
            model_path = os.path.join(args.output_dir, f"{model_type.lower()}_model.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_type} model to {model_path}")

            logger.info(f"{model_type} metrics: {metrics}")

            # Save model and feature importance
            if hasattr(model.model, 'feature_importances_') and X_train.shape[1] > 0:
                feature_importance = model.get_feature_importance(X_train.columns)
                importance_path = os.path.join(args.output_dir, f"{model_type.lower()}_feature_importance.csv")
                feature_importance.to_csv(importance_path, index=False)
                logger.info(f"Saved feature importance to {importance_path}")

        # 5. Visualizations
        logger.info("5. Creating visualizations")
        visualizer = Visualizer(config_path=args.config, output_dir=os.path.join(args.output_dir, "visualizations"))
        visualizer.plot_model_comparison(metrics_dict)
        visualizer.plot_roc_curves(model_types, metrics_dict)
        
        # For each model, plot confusion matrix and feature importance if available
        for model_type in model_types:
            if model_type in metrics_dict:
                model = create_model(model_type, config_dict['models'][model_key_map[model_type]]['hyperparameters'])
                model = model.load(os.path.join(args.output_dir, f"{model_type.lower()}_model.joblib"))
                
                # Plot confusion matrix
                y_pred = model.predict(X_test)
                visualizer.plot_confusion_matrix(y_test, y_pred, model_type)
                
                # Plot feature importance for tree-based models
                if model_type in ['RandomForest', 'GradientBoosting']:
                    visualizer.plot_feature_importance(model, X_train.columns, model_type)
        
        logger.info("Pipeline completed successfully")
        return metrics_dict
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Football Player Performance Prediction')
    
    parser.add_argument('--data_path', type=str, default='data/raw/FPL_logs.csv',
                        help='Path to the FPL_logs.csv dataset')
    
    parser.add_argument('--config', type=str, default='config/system_config.yaml',
                        help='Path to system configuration file')
    
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save models and results')
    
    parser.add_argument('--target_column', type=str, default='FPL_points',
                        help='Column name of the target variable')
    
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='Threshold for binary classification of player performance')
    
    parser.add_argument('--feature_selection', type=str, default='SelectKBest',
                        choices=['SelectKBest', 'RFE', 'PCA'],
                        help='Feature selection method')
    
    parser.add_argument('--n_features', type=int, default=30,
                        help='Number of features to select')
    
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization')
    
    parser.add_argument('--skip_feature_selection', action='store_true',
                        help='Skip feature selection step')
    
    parser.add_argument('--skip_models', type=str, nargs='+',
                        choices=['RandomForest', 'SVM', 'NeuralNetwork', 'GradientBoosting'],
                        help='Models to skip training')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    
    # Run main function
    main(args)