"""
Unit tests for configuration module.
"""
import pytest
from pathlib import Path
from src.config import (
    DataConfig, ModelConfig, LoggingConfig, APIConfig,
    data_config, model_config, logging_config, api_config,
    BASE_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR
)


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.target_column == "class"
        assert config.test_size == 0.2
        assert config.random_state == 42
    
    def test_fraud_drop_columns(self):
        """Test fraud drop columns list."""
        config = DataConfig()
        expected_cols = ["user_id", "signup_time", "purchase_time", "device_id", "country"]
        assert config.fraud_drop_columns == expected_cols
    
    def test_file_paths_exist(self):
        """Test that config paths are Path objects."""
        config = DataConfig()
        assert isinstance(config.fraud_data_path, Path)
        assert isinstance(config.creditcard_data_path, Path)


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_model_types(self):
        """Test available model types."""
        config = ModelConfig()
        expected_types = ["logistic_regression", "random_forest", "xgboost"]
        assert config.model_types == expected_types
    
    def test_rf_params(self):
        """Test Random Forest parameters."""
        config = ModelConfig()
        assert "n_estimators" in config.rf_params
        assert "random_state" in config.rf_params
        assert config.rf_params["random_state"] == 42
    
    def test_smote_config(self):
        """Test SMOTE configuration."""
        config = ModelConfig()
        assert config.use_smote is True
        assert 0 < config.smote_sampling_strategy <= 1


class TestLoggingConfig:
    """Tests for LoggingConfig."""
    
    def test_log_level(self):
        """Test default log level."""
        config = LoggingConfig()
        assert config.log_level == "INFO"
    
    def test_log_format(self):
        """Test log format contains required fields."""
        config = LoggingConfig()
        assert "%(asctime)s" in config.log_format
        assert "%(levelname)s" in config.log_format


class TestAPIConfig:
    """Tests for APIConfig."""
    
    def test_default_host_port(self):
        """Test default host and port."""
        config = APIConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
    
    def test_api_metadata(self):
        """Test API metadata."""
        config = APIConfig()
        assert config.title == "Fraud Detection API"
        assert config.version == "1.0.0"


class TestDirectoryCreation:
    """Tests for directory creation."""
    
    def test_base_directories_exist(self):
        """Test that base directories are created."""
        assert BASE_DIR.exists()
        assert DATA_DIR.exists()
        assert MODELS_DIR.exists()
        assert LOGS_DIR.exists()


class TestConfigInstances:
    """Tests for config instances."""
    
    def test_data_config_instance(self):
        """Test data_config instance."""
        assert isinstance(data_config, DataConfig)
    
    def test_model_config_instance(self):
        """Test model_config instance."""
        assert isinstance(model_config, ModelConfig)
    
    def test_logging_config_instance(self):
        """Test logging_config instance."""
        assert isinstance(logging_config, LoggingConfig)
    
    def test_api_config_instance(self):
        """Test api_config instance."""
        assert isinstance(api_config, APIConfig)
