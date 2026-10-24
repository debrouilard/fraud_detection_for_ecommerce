"""
Data preprocessing utilities for fraud detection.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.logger import setup_logger
from src.config import model_config

logger = setup_logger(__name__, "preprocessor.log")


class DataPreprocessor:
    """Handles data preprocessing including scaling and balancing."""
    
    def __init__(self, use_smote: bool = None):
        """
        Initialize preprocessor.
        
        Args:
            use_smote: Whether to use SMOTE for class balancing
        """
        self.use_smote = use_smote if use_smote is not None else model_config.use_smote
        self.scaler = StandardScaler()
        self.smote = None
        self.fitted = False
        
        if self.use_smote:
            self.smote = SMOTE(
                sampling_strategy=model_config.smote_sampling_strategy,
                random_state=model_config.smote_random_state
            )
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit preprocessor and transform training data.
        
        Args:
            X: Training features
            y: Training target
        
        Returns:
            Tuple of (transformed X, transformed y)
        """
        logger.info("Fitting and transforming training data")
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Scale features
        X_scaled = self._scale_features(X, fit=True)
        
        # Apply SMOTE if enabled
        if self.use_smote:
            X_balanced, y_balanced = self._apply_smote(X_scaled, y)
            self.fitted = True
            return X_balanced, y_balanced
        
        self.fitted = True
        return X_scaled, y
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test/new data using fitted preprocessor.
        
        Args:
            X: Features to transform
        
        Returns:
            Transformed features
        
        Raises:
            RuntimeError: If preprocessor not fitted
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        logger.info("Transforming data")
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Scale features
        X_scaled = self._scale_features(X, fit=False)
        
        return X_scaled
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in features.
        
        Args:
            X: Input features
        
        Returns:
            DataFrame with missing values handled
        """
        X = X.copy()
        
        missing_counts = X.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Found missing values in columns: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Fill numeric columns with median
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X[col].isnull().any():
                    X[col].fillna(X[col].median(), inplace=True)
            
            # Fill remaining with mode or 0
            X.fillna(0, inplace=True)
            
            logger.info("Missing values handled")
        
        return X
    
    def _scale_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            X: Input features
            fit: Whether to fit the scaler
        
        Returns:
            Scaled features
        """
        X = X.copy()
        
        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            logger.info("Fitted and scaled features")
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            logger.info("Scaled features using fitted scaler")
        
        return X_scaled
    
    def _apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance classes.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            Tuple of (balanced X, balanced y)
        """
        original_counts = y.value_counts().to_dict()
        logger.info(f"Original class distribution: {original_counts}")
        
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        new_counts = pd.Series(y_resampled).value_counts().to_dict()
        logger.info(f"Resampled class distribution: {new_counts}")
        
        # Convert back to DataFrame/Series with proper index
        X_balanced = pd.DataFrame(X_resampled, columns=X.columns)
        y_balanced = pd.Series(y_resampled, name=y.name)
        
        return X_balanced, y_balanced
    
    def get_feature_names(self) -> list:
        """
        Get feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted first")
        
        return list(self.scaler.feature_names_in_)


def preprocess_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    use_smote: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Complete preprocessing pipeline for train and test data.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        use_smote: Whether to use SMOTE
    
    Returns:
        Tuple of (X_train_processed, y_train_processed, X_test_processed)
    """
    preprocessor = DataPreprocessor(use_smote=use_smote)
    
    # Fit and transform training data
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    logger.info(
        f"Preprocessing complete: train={X_train_processed.shape}, "
        f"test={X_test_processed.shape}"
    )
    
    return X_train_processed, y_train_processed, X_test_processed
