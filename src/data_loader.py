"""
Data loading utilities for fraud detection system.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from src.logger import setup_logger
from src.config import data_config

logger = setup_logger(__name__, "data_loader.log")


class DataLoader:
    """Handles loading and basic validation of datasets."""
    
    def __init__(self, config=None):
        """
        Initialize DataLoader.
        
        Args:
            config: DataConfig instance (uses default if None)
        """
        self.config = config or data_config
        
    def load_fraud_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load fraud detection dataset.
        
        Args:
            file_path: Path to fraud data file (uses config default if None)
        
        Returns:
            DataFrame containing fraud data
        
        Raises:
            FileNotFoundError: If data file doesn't exist
            pd.errors.EmptyDataError: If file is empty
        """
        path = file_path or self.config.fraud_data_path
        
        try:
            logger.info(f"Loading fraud data from {path}")
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded {len(df)} fraud records")
            
            # Basic validation
            self._validate_dataframe(df, "fraud")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Fraud data file not found: {path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Fraud data file is empty: {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading fraud data: {str(e)}")
            raise
    
    def load_creditcard_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load credit card transaction dataset.
        
        Args:
            file_path: Path to credit card data file (uses config default if None)
        
        Returns:
            DataFrame containing credit card data
        
        Raises:
            FileNotFoundError: If data file doesn't exist
            pd.errors.EmptyDataError: If file is empty
        """
        path = file_path or self.config.creditcard_data_path
        
        try:
            logger.info(f"Loading credit card data from {path}")
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded {len(df)} credit card records")
            
            # Basic validation
            self._validate_dataframe(df, "creditcard")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Credit card data file not found: {path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Credit card data file is empty: {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading credit card data: {str(e)}")
            raise
    
    def load_ip_to_country(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load IP to country mapping dataset.
        
        Args:
            file_path: Path to IP mapping file (uses config default if None)
        
        Returns:
            DataFrame containing IP to country mappings
        
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        path = file_path or self.config.ip_to_country_path
        
        try:
            logger.info(f"Loading IP to country mapping from {path}")
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded {len(df)} IP mappings")
            return df
            
        except FileNotFoundError:
            logger.warning(f"IP to country file not found: {path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading IP to country data: {str(e)}")
            raise
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        dataset_type: str,
        file_path: Optional[Path] = None
    ) -> None:
        """
        Save processed dataframe to CSV.
        
        Args:
            df: DataFrame to save
            dataset_type: Type of dataset ('fraud' or 'creditcard')
            file_path: Custom save path (uses config default if None)
        """
        if dataset_type == "fraud":
            path = file_path or self.config.fraud_processed_path
        elif dataset_type == "creditcard":
            path = file_path or self.config.creditcard_processed_path
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        try:
            logger.info(f"Saving processed {dataset_type} data to {path}")
            df.to_csv(path, index=False)
            logger.info(f"Successfully saved {len(df)} records")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def load_processed_data(self, dataset_type: str) -> pd.DataFrame:
        """
        Load processed dataset.
        
        Args:
            dataset_type: Type of dataset ('fraud' or 'creditcard')
        
        Returns:
            Processed DataFrame
        """
        if dataset_type == "fraud":
            path = self.config.fraud_processed_path
        elif dataset_type == "creditcard":
            path = self.config.creditcard_processed_path
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        try:
            logger.info(f"Loading processed {dataset_type} data from {path}")
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded {len(df)} processed records")
            return df
            
        except FileNotFoundError:
            logger.error(f"Processed data file not found: {path}")
            raise
    
    def _validate_dataframe(self, df: pd.DataFrame, dataset_type: str) -> None:
        """
        Validate basic properties of loaded dataframe.
        
        Args:
            df: DataFrame to validate
            dataset_type: Type of dataset for validation
        
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError(f"{dataset_type} dataset is empty")
        
        # Check for target column
        if self.config.target_column not in df.columns:
            logger.warning(
                f"Target column '{self.config.target_column}' not found in {dataset_type} data"
            )
        else:
            # Log class distribution
            class_counts = df[self.config.target_column].value_counts()
            logger.info(f"{dataset_type} class distribution:\n{class_counts}")
            
            # Check for severe imbalance
            if len(class_counts) == 2:
                imbalance_ratio = class_counts.min() / class_counts.max()
                if imbalance_ratio < 0.01:
                    logger.warning(
                        f"Severe class imbalance detected in {dataset_type}: "
                        f"{imbalance_ratio:.4f}"
                    )
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values in {dataset_type}:\n{missing[missing > 0]}")


def load_train_test_split(
    dataset_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load processed data and split into train/test sets.
    
    Args:
        dataset_type: Type of dataset ('fraud' or 'creditcard')
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    loader = DataLoader()
    df = loader.load_processed_data(dataset_type)
    
    # Separate features and target
    y = df[data_config.target_column]
    X = df.drop(columns=[data_config.target_column])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_config.test_size,
        random_state=data_config.random_state,
        stratify=y
    )
    
    logger.info(f"Split {dataset_type} data: train={len(X_train)}, test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test
