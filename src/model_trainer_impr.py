"""
Model training utilities for fraud detection.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path

from src.logger import setup_logger
from src.config import model_config

logger = setup_logger(__name__, "model_trainer.log")


class ModelTrainer:
    """
    Handles training of various fraud detection models.
    Supports class weights for handling imbalance without SMOTE.
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        use_class_weights: bool = False,
        custom_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model to train
                       ('logistic_regression', 'random_forest', 'xgboost')
            use_class_weights: Whether to use class weights for imbalance
            custom_params: Custom parameters (overrides defaults, e.g., from Optuna)
        """
        self.model_type = model_type
        self.use_class_weights = use_class_weights
        self.custom_params = custom_params
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on type with optional class weights."""
        if self.model_type == "logistic_regression":
            params = self.custom_params if self.custom_params else model_config.lr_params.copy()
            if self.use_class_weights:
                params['class_weight'] = 'balanced'
            self.model = LogisticRegression(**params)
            logger.info(f"Initialized Logistic Regression model (class_weights={self.use_class_weights})")
            
        elif self.model_type == "random_forest":
            params = self.custom_params if self.custom_params else model_config.rf_params.copy()
            if self.use_class_weights:
                params['class_weight'] = 'balanced'
            self.model = RandomForestClassifier(**params)
            logger.info(f"Initialized Random Forest model (class_weights={self.use_class_weights})")
            
        elif self.model_type == "xgboost":
            params = self.custom_params if self.custom_params else model_config.xgb_params.copy()
            # XGBoost uses scale_pos_weight instead of class_weight
            # This will be set during training when we know the class distribution
            self.model = xgb.XGBClassifier(**params)
            logger.info(f"Initialized XGBoost model (class_weights={self.use_class_weights})")
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        validate: bool = True
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            validate: Whether to perform cross-validation
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # For XGBoost with class weights, set scale_pos_weight
        if self.model_type == "xgboost" and self.use_class_weights:
            n_positive = (y_train == 1).sum()
            n_negative = (y_train == 0).sum()
            scale_pos_weight = n_negative / n_positive
            self.model.set_params(scale_pos_weight=scale_pos_weight)
            logger.info(f"Set XGBoost scale_pos_weight to {scale_pos_weight:.2f}")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info(f"Model training complete")
        
        # Get training score
        train_score = self.model.score(X_train, y_train)
        metrics = {"train_accuracy": train_score}
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        # Perform cross-validation if requested
        if validate:
            cv_scores = self._cross_validate(X_train, y_train)
            metrics.update(cv_scores)
        
        return metrics
    
    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            Dictionary with CV metrics
        """
        logger.info(f"Performing {model_config.cv_folds}-fold cross-validation")
        
        cv_scores = cross_val_score(
            self.model,
            X, y,
            cv=model_config.cv_folds,
            scoring='accuracy',
            n_jobs=-1
        )
        
        metrics = {
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),
            "cv_scores": cv_scores.tolist()
        }
        
        logger.info(
            f"CV Accuracy: {metrics['cv_mean_accuracy']:.4f} "
            f"(+/- {metrics['cv_std_accuracy']:.4f})"
        )
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
        
        Returns:
            Array of predictions
        
        Raises:
            RuntimeError: If model not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict
        
        Returns:
            Array of prediction probabilities
        
        Raises:
            RuntimeError: If model not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importance or None if not available
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        if self.model_type == "logistic_regression":
            # Use coefficients for logistic regression
            importance = np.abs(self.model.coef_[0])
        elif hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            logger.warning(f"Feature importance not available for {self.model_type}")
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save_model(
        self,
        filepath: Optional[Path] = None,
        dataset_name: str = "model"
    ) -> Path:
        """
        Save trained model to disk.
        
        Args:
            filepath: Custom save path (uses config default if None)
            dataset_name: Name identifier for the model
        
        Returns:
            Path where model was saved
        
        Raises:
            RuntimeError: If model not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        if filepath is None:
            filename = f"{self.model_type}_{dataset_name}_model.joblib"
            filepath = model_config.model_save_dir / filename
        
        logger.info(f"Saving model to {filepath}")
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved successfully")
        
        return filepath
    
    def load_model(self, filepath: Path) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model
        
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        logger.info(f"Loading model from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded successfully: {self.model_type}")


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dataset_name: str,
    model_types: Optional[list] = None
) -> Dict[str, ModelTrainer]:
    """
    Train multiple models on the same dataset.
    
    Args:
        X_train: Training features
        y_train: Training target
        dataset_name: Name identifier for saving models
        model_types: List of model types to train (uses config default if None)
    
    Returns:
        Dictionary mapping model type to trained ModelTrainer
    """
    if model_types is None:
        model_types = model_config.model_types
    
    trained_models = {}
    
    for model_type in model_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type} for {dataset_name}")
        logger.info(f"{'='*60}\n")
        
        try:
            trainer = ModelTrainer(model_type=model_type)
            metrics = trainer.train(X_train, y_train)
            
            # Save model
            trainer.save_model(dataset_name=dataset_name)
            
            trained_models[model_type] = trainer
            
            logger.info(f"✓ {model_type} trained successfully")
            
        except Exception as e:
            logger.error(f"✗ Error training {model_type}: {str(e)}")
            continue
    
    return trained_models
