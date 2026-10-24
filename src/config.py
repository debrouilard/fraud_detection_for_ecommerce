"""
Configuration management for fraud detection system.
"""
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import os


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # File paths
    fraud_data_path: Path = RAW_DATA_DIR / "Fraud_Data.csv"
    creditcard_data_path: Path = RAW_DATA_DIR / "creditcard.csv"
    ip_to_country_path: Path = RAW_DATA_DIR / "IpAddress_to_Country.csv"
    
    # Processed data paths
    fraud_processed_path: Path = PROCESSED_DATA_DIR / "fraud_data_processed.csv"
    creditcard_processed_path: Path = PROCESSED_DATA_DIR / "creditcard_processed.csv"
    
    # Feature columns to drop
    # These must match what the notebook pipeline drops to ensure model compatibility
    fraud_drop_columns: List[str] = field(default_factory=lambda: [
        "user_id", "signup_time", "purchase_time", "device_id", "country",
        "ip_address", "browser_source", "ip_country"  # Added for notebook compatibility
    ])
    creditcard_drop_columns: List[str] = field(default_factory=lambda: [
        "Time", "Amount"
    ])
    
    # Target column
    target_column: str = "class"
    
    # Test split ratio
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    # Model types
    model_types: List[str] = field(default_factory=lambda: [
        "logistic_regression", "random_forest", "xgboost"
    ])
    
    # Random Forest parameters
    rf_params: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced"
    })
    
    # XGBoost parameters
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss"
    })
    
    # Logistic Regression parameters
    lr_params: Dict = field(default_factory=lambda: {
        "max_iter": 1000,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced"
    })
    
    # SMOTE parameters
    use_smote: bool = True
    smote_sampling_strategy: float = 0.5
    smote_random_state: int = 42
    
    # Model save paths
    model_save_dir: Path = MODELS_DIR
    
    # Cross-validation
    cv_folds: int = 5
    
    # Threshold optimization
    optimize_threshold: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    log_dir: Path = LOGS_DIR
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_to_console: bool = True


@dataclass
class APIConfig:
    """Configuration for API service."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    title: str = "Fraud Detection API"
    description: str = "API for detecting fraudulent transactions in e-commerce and banking"
    version: str = "1.0.0"


# Create config instances
data_config = DataConfig()
model_config = ModelConfig()
logging_config = LoggingConfig()
api_config = APIConfig()
