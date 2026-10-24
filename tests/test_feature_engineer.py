"""
Unit tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
from src.feature_engineer import (
    FraudDataFeatureEngineer,
    CreditCardFeatureEngineer,
    prepare_features
)


@pytest.fixture
def fraud_data_sample():
    """Create sample fraud data."""
    return pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'signup_time': ['2024-01-01 10:00:00', '2024-01-01 11:00:00',
                       '2024-01-01 12:00:00', '2024-01-01 13:00:00',
                       '2024-01-01 14:00:00'],
        'purchase_time': ['2024-01-01 12:00:00', '2024-01-01 11:30:00',
                         '2024-01-01 15:00:00', '2024-01-01 14:00:00',
                         '2024-01-01 16:00:00'],
        'purchase_value': [100, 200, 50, 150, 300],
        'age': [25, 30, 35, 40, 45],
        'ip_address': ['192.168.1.1', '10.0.0.1', '172.16.0.1', '8.8.8.8', '1.1.1.1'],
        'browser': ['Chrome', 'Firefox', 'Safari', 'Chrome', 'Firefox'],
        'source': ['Ads', 'SEO', 'Direct', 'Ads', 'SEO'],
        'sex': ['M', 'F', 'M', 'F', 'M'],
        'class': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def creditcard_data_sample():
    """Create sample credit card data."""
    data = {
        'Time': [0, 100, 200, 300, 400],
        'Amount': [100, 200, 50, 150, 300],
        'class': [0, 1, 0, 1, 0]
    }
    # Add V1-V10 features
    for i in range(1, 11):
        data[f'V{i}'] = np.random.randn(5)
    return pd.DataFrame(data)


class TestFraudDataFeatureEngineer:
    """Tests for FraudDataFeatureEngineer."""
    
    def test_initialization(self):
        """Test initialization."""
        engineer = FraudDataFeatureEngineer()
        assert engineer.scaler is not None
        assert engineer.fitted is False
    
    def test_engineer_features(self, fraud_data_sample):
        """Test feature engineering."""
        engineer = FraudDataFeatureEngineer()
        result = engineer.engineer_features(fraud_data_sample)
        
        # Check that new features are created
        assert 'hour_of_day' in result.columns
        assert 'day_of_week' in result.columns
        assert 'time_since_signup_hours' in result.columns
        assert 'quick_purchase' in result.columns
    
    def test_temporal_features(self, fraud_data_sample):
        """Test temporal feature creation."""
        engineer = FraudDataFeatureEngineer()
        result = engineer.engineer_features(fraud_data_sample)
        
        assert 'hour_of_day' in result.columns
        assert 'day_of_week' in result.columns
        assert 'is_weekend' in result.columns
        assert 'day_of_month' in result.columns
        assert 'month' in result.columns
    
    def test_ip_to_int_conversion(self):
        """Test IP address to integer conversion."""
        engineer = FraudDataFeatureEngineer()
        
        # Test valid IP
        ip_int = engineer._ip_to_int('192.168.1.1')
        assert ip_int > 0
        assert isinstance(ip_int, int)
        
        # Test invalid IP
        ip_int = engineer._ip_to_int('invalid')
        assert ip_int == 0
    
    def test_purchase_value_features(self, fraud_data_sample):
        """Test purchase value feature engineering."""
        engineer = FraudDataFeatureEngineer()
        result = engineer.engineer_features(fraud_data_sample)
        
        assert 'log_purchase_value' in result.columns
        assert 'purchase_value_rounded' in result.columns
    
    def test_categorical_encoding(self, fraud_data_sample):
        """Test one-hot encoding of categorical variables."""
        engineer = FraudDataFeatureEngineer()
        result = engineer.engineer_features(fraud_data_sample)
        
        # Check that categorical columns are encoded
        assert any('browser_' in col for col in result.columns)
        assert any('source_' in col for col in result.columns)


class TestCreditCardFeatureEngineer:
    """Tests for CreditCardFeatureEngineer."""
    
    def test_initialization(self):
        """Test initialization."""
        engineer = CreditCardFeatureEngineer()
        assert engineer.amount_scaler is not None
        assert engineer.time_scaler is not None
        assert engineer.fitted is False
    
    def test_engineer_features_fit(self, creditcard_data_sample):
        """Test feature engineering with fit=True."""
        engineer = CreditCardFeatureEngineer()
        result = engineer.engineer_features(creditcard_data_sample, fit=True)
        
        assert 'scaled_time' in result.columns
        assert 'scaled_amount' in result.columns
        assert 'hour' in result.columns
        assert 'day' in result.columns
        assert engineer.fitted is True
    
    def test_engineer_features_transform(self, creditcard_data_sample):
        """Test feature engineering with fit=False."""
        engineer = CreditCardFeatureEngineer()
        
        # First fit
        _ = engineer.engineer_features(creditcard_data_sample, fit=True)
        
        # Then transform
        result = engineer.engineer_features(creditcard_data_sample, fit=False)
        assert 'scaled_amount' in result.columns
    
    def test_amount_features(self, creditcard_data_sample):
        """Test amount-based feature creation."""
        engineer = CreditCardFeatureEngineer()
        result = engineer.engineer_features(creditcard_data_sample, fit=True)
        
        assert 'log_amount' in result.columns
        assert 'amount_category' in result.columns
    
    def test_interaction_features(self, creditcard_data_sample):
        """Test interaction features creation."""
        engineer = CreditCardFeatureEngineer()
        result = engineer.engineer_features(creditcard_data_sample, fit=True)
        
        # Check for interaction features
        interaction_cols = [col for col in result.columns if 'interaction' in col]
        assert len(interaction_cols) > 0


class TestPrepareFeatures:
    """Tests for prepare_features function."""
    
    def test_prepare_fraud_features(self, fraud_data_sample):
        """Test preparing fraud features."""
        result = prepare_features(
            fraud_data_sample,
            dataset_type='fraud',
            drop_columns=['user_id', 'class'],
            fit=True
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'user_id' not in result.columns
        assert 'class' not in result.columns
        
        # All columns should be numeric
        assert result.select_dtypes(include=['number', 'bool']).shape[1] == result.shape[1]
    
    def test_prepare_creditcard_features(self, creditcard_data_sample):
        """Test preparing credit card features."""
        result = prepare_features(
            creditcard_data_sample,
            dataset_type='creditcard',
            drop_columns=['class'],
            fit=True
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'class' not in result.columns
        
        # All columns should be numeric
        assert result.select_dtypes(include=['number', 'bool']).shape[1] == result.shape[1]
    
    def test_prepare_features_invalid_type(self, fraud_data_sample):
        """Test with invalid dataset type."""
        with pytest.raises(ValueError, match="Unknown dataset type"):
            prepare_features(fraud_data_sample, dataset_type='invalid')
    
    def test_prepare_features_no_drop(self, fraud_data_sample):
        """Test without dropping columns."""
        result = prepare_features(
            fraud_data_sample,
            dataset_type='fraud',
            drop_columns=None,
            fit=True
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(fraud_data_sample)
