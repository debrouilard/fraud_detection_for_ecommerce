"""
Unit tests for preprocessor module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.preprocessor import DataPreprocessor, preprocess_pipeline


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    # Create imbalanced target
    y = pd.Series([0] * 90 + [1] * 10, name='class')
    return X, y


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values."""
    X = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [5, np.nan, 7, 8, 9],
        'feature3': [10, 11, 12, np.nan, 14]
    })
    y = pd.Series([0, 1, 0, 1, 0], name='class')
    return X, y


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.use_smote is True
        assert preprocessor.fitted is False
        assert preprocessor.smote is not None
    
    def test_initialization_no_smote(self):
        """Test initialization without SMOTE."""
        preprocessor = DataPreprocessor(use_smote=False)
        assert preprocessor.use_smote is False
    
    def test_fit_transform_without_smote(self, sample_data):
        """Test fit_transform without SMOTE."""
        X, y = sample_data
        preprocessor = DataPreprocessor(use_smote=False)
        
        X_transformed, y_transformed = preprocessor.fit_transform(X, y)
        
        assert preprocessor.fitted is True
        assert len(X_transformed) == len(X)
        assert len(y_transformed) == len(y)
        assert isinstance(X_transformed, pd.DataFrame)
    
    def test_fit_transform_with_smote(self, sample_data):
        """Test fit_transform with SMOTE."""
        X, y = sample_data
        preprocessor = DataPreprocessor(use_smote=True)
        
        original_fraud_count = (y == 1).sum()
        X_transformed, y_transformed = preprocessor.fit_transform(X, y)
        new_fraud_count = (y_transformed == 1).sum()
        
        assert preprocessor.fitted is True
        assert new_fraud_count > original_fraud_count  # SMOTE increased minority class
        assert len(X_transformed) > len(X)  # Total samples increased
    
    def test_transform_not_fitted(self, sample_data):
        """Test transform without fitting first."""
        X, _ = sample_data
        preprocessor = DataPreprocessor(use_smote=False)
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            preprocessor.transform(X)
    
    def test_transform_after_fit(self, sample_data):
        """Test transform after fitting."""
        X, y = sample_data
        preprocessor = DataPreprocessor(use_smote=False)
        
        # Fit first
        preprocessor.fit_transform(X, y)
        
        # Then transform new data
        X_test = X.iloc[:10].copy()
        X_transformed = preprocessor.transform(X_test)
        
        assert len(X_transformed) == len(X_test)
        assert list(X_transformed.columns) == list(X.columns)
    
    def test_handle_missing_values(self, sample_data_with_missing):
        """Test missing value handling."""
        X, y = sample_data_with_missing
        preprocessor = DataPreprocessor(use_smote=False)
        
        X_transformed, _ = preprocessor.fit_transform(X, y)
        
        # Check no missing values remain
        assert X_transformed.isnull().sum().sum() == 0
    
    def test_scale_features(self, sample_data):
        """Test feature scaling."""
        X, y = sample_data
        preprocessor = DataPreprocessor(use_smote=False)
        
        X_transformed, _ = preprocessor.fit_transform(X, y)
        
        # Check that features are scaled (mean ~0, std ~1)
        for col in X_transformed.columns:
            assert abs(X_transformed[col].mean()) < 0.1
            assert abs(X_transformed[col].std() - 1.0) < 0.1
    
    def test_get_feature_names(self, sample_data):
        """Test getting feature names."""
        X, y = sample_data
        preprocessor = DataPreprocessor(use_smote=False)
        
        preprocessor.fit_transform(X, y)
        feature_names = preprocessor.get_feature_names()
        
        assert feature_names == list(X.columns)
    
    def test_get_feature_names_not_fitted(self):
        """Test getting feature names before fitting."""
        preprocessor = DataPreprocessor()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            preprocessor.get_feature_names()


class TestPreprocessPipeline:
    """Tests for preprocess_pipeline function."""
    
    def test_preprocess_pipeline_with_smote(self, sample_data):
        """Test complete preprocessing pipeline with SMOTE."""
        X, y = sample_data
        X_train = X.iloc[:80]
        y_train = y.iloc[:80]
        X_test = X.iloc[80:]
        
        X_train_proc, y_train_proc, X_test_proc = preprocess_pipeline(
            X_train, y_train, X_test, use_smote=True
        )
        
        # Check shapes
        assert len(X_train_proc) > len(X_train)  # SMOTE increased size
        assert len(X_test_proc) == len(X_test)
        assert len(y_train_proc) == len(X_train_proc)
        
        # Check columns match
        assert list(X_train_proc.columns) == list(X_test_proc.columns)
    
    def test_preprocess_pipeline_without_smote(self, sample_data):
        """Test complete preprocessing pipeline without SMOTE."""
        X, y = sample_data
        X_train = X.iloc[:80]
        y_train = y.iloc[:80]
        X_test = X.iloc[80:]
        
        X_train_proc, y_train_proc, X_test_proc = preprocess_pipeline(
            X_train, y_train, X_test, use_smote=False
        )
        
        # Check shapes
        assert len(X_train_proc) == len(X_train)  # No SMOTE
        assert len(X_test_proc) == len(X_test)
        assert len(y_train_proc) == len(y_train)
    
    def test_preprocess_pipeline_preserves_columns(self, sample_data):
        """Test that pipeline preserves column names."""
        X, y = sample_data
        X_train = X.iloc[:80]
        y_train = y.iloc[:80]
        X_test = X.iloc[80:]
        
        X_train_proc, _, X_test_proc = preprocess_pipeline(
            X_train, y_train, X_test, use_smote=False
        )
        
        assert list(X_train_proc.columns) == list(X.columns)
        assert list(X_test_proc.columns) == list(X.columns)
