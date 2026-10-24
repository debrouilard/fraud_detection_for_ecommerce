"""
Pytest configuration and fixtures.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_fraud_dataframe():
    """Create a sample fraud detection dataframe."""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'user_id': range(n_samples),
        'signup_time': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'purchase_time': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'purchase_value': np.random.uniform(10, 500, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'ip_address': [f'192.168.{i%256}.{(i*7)%256}' for i in range(n_samples)],
        'browser': np.random.choice(['Chrome', 'Firefox', 'Safari'], n_samples),
        'source': np.random.choice(['Ads', 'SEO', 'Direct'], n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'class': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })


@pytest.fixture
def sample_creditcard_dataframe():
    """Create a sample credit card dataframe."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Time': np.random.uniform(0, 172800, n_samples),
        'Amount': np.random.uniform(0, 1000, n_samples),
        'class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    }
    
    # Add V1-V28 features (PCA components)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)
