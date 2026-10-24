"""
Unit tests for data loader module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.data_loader import DataLoader, load_train_test_split


@pytest.fixture
def sample_fraud_data():
    """Create sample fraud data for testing."""
    return pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'signup_time': pd.date_range('2024-01-01', periods=5),
        'purchase_time': pd.date_range('2024-01-02', periods=5),
        'purchase_value': [100, 200, 50, 150, 300],
        'age': [25, 30, 35, 40, 45],
        'class': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_creditcard_data():
    """Create sample credit card data for testing."""
    data = {
        'Time': [0, 1, 2, 3, 4],
        'Amount': [100, 200, 50, 150, 300],
        'class': [0, 1, 0, 1, 0]
    }
    # Add V1-V5 features
    for i in range(1, 6):
        data[f'V{i}'] = np.random.randn(5)
    return pd.DataFrame(data)


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.config is not None
    
    @patch('pandas.read_csv')
    def test_load_fraud_data_success(self, mock_read_csv, sample_fraud_data):
        """Test successful fraud data loading."""
        mock_read_csv.return_value = sample_fraud_data
        
        loader = DataLoader()
        df = loader.load_fraud_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'class' in df.columns
        mock_read_csv.assert_called_once()
    
    @patch('pandas.read_csv')
    def test_load_fraud_data_file_not_found(self, mock_read_csv):
        """Test fraud data loading with missing file."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_fraud_data()
    
    @patch('pandas.read_csv')
    def test_load_creditcard_data_success(self, mock_read_csv, sample_creditcard_data):
        """Test successful credit card data loading."""
        mock_read_csv.return_value = sample_creditcard_data
        
        loader = DataLoader()
        df = loader.load_creditcard_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'class' in df.columns
    
    @patch('pandas.read_csv')
    def test_load_ip_to_country(self, mock_read_csv):
        """Test IP to country loading."""
        mock_data = pd.DataFrame({
            'lower_bound_ip': [0, 1000],
            'upper_bound_ip': [999, 2000],
            'country': ['US', 'UK']
        })
        mock_read_csv.return_value = mock_data
        
        loader = DataLoader()
        df = loader.load_ip_to_country()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    
    def test_validate_empty_dataframe(self):
        """Test validation with empty dataframe."""
        loader = DataLoader()
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="dataset is empty"):
            loader._validate_dataframe(empty_df, "test")
    
    def test_validate_class_imbalance_warning(self, sample_fraud_data):
        """Test class imbalance warning."""
        # Create severely imbalanced data
        imbalanced_data = sample_fraud_data.copy()
        imbalanced_data.loc[:98, 'class'] = 0
        imbalanced_data.loc[99:, 'class'] = 1
        
        loader = DataLoader()
        # Should not raise, but log warning
        loader._validate_dataframe(imbalanced_data, "test")
    
    @patch('pandas.DataFrame.to_csv')
    def test_save_processed_data(self, mock_to_csv, sample_fraud_data):
        """Test saving processed data."""
        loader = DataLoader()
        loader.save_processed_data(sample_fraud_data, "fraud")
        
        mock_to_csv.assert_called_once()
        call_args = mock_to_csv.call_args
        assert call_args[1]['index'] is False
    
    def test_save_processed_data_invalid_type(self, sample_fraud_data):
        """Test saving with invalid dataset type."""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="Unknown dataset type"):
            loader.save_processed_data(sample_fraud_data, "invalid")


class TestLoadTrainTestSplit:
    """Tests for load_train_test_split function."""
    
    @patch('src.data_loader.DataLoader.load_processed_data')
    def test_load_train_test_split(self, mock_load_data, sample_fraud_data):
        """Test train/test split loading."""
        mock_load_data.return_value = sample_fraud_data
        
        X_train, X_test, y_train, y_test = load_train_test_split('fraud')
        
        # Check shapes
        assert len(X_train) + len(X_test) == len(sample_fraud_data)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Check that target column is not in features
        assert 'class' not in X_train.columns
        assert 'class' not in X_test.columns
