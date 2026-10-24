"""
Unit tests for model trainer module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from src.model_trainer import ModelTrainer, train_all_models


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    y_train = pd.Series(np.random.randint(0, 2, 100), name='class')
    return X_train, y_train


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    def test_initialization_random_forest(self):
        """Test initialization with Random Forest."""
        trainer = ModelTrainer(model_type='random_forest')
        assert trainer.model is not None
        assert trainer.model_type == 'random_forest'
        assert trainer.is_trained is False
    
    def test_initialization_logistic_regression(self):
        """Test initialization with Logistic Regression."""
        trainer = ModelTrainer(model_type='logistic_regression')
        assert trainer.model_type == 'logistic_regression'
    
    def test_initialization_xgboost(self):
        """Test initialization with XGBoost."""
        trainer = ModelTrainer(model_type='xgboost')
        assert trainer.model_type == 'xgboost'
    
    def test_initialization_invalid_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelTrainer(model_type='invalid_model')
    
    def test_train(self, sample_training_data):
        """Test model training."""
        X_train, y_train = sample_training_data
        trainer = ModelTrainer(model_type='logistic_regression')
        
        metrics = trainer.train(X_train, y_train, validate=False)
        
        assert trainer.is_trained is True
        assert 'train_accuracy' in metrics
        assert 0 <= metrics['train_accuracy'] <= 1
        assert trainer.feature_names == list(X_train.columns)
    
    def test_train_with_validation(self, sample_training_data):
        """Test training with cross-validation."""
        X_train, y_train = sample_training_data
        trainer = ModelTrainer(model_type='logistic_regression')
        
        metrics = trainer.train(X_train, y_train, validate=True)
        
        assert 'cv_mean_accuracy' in metrics
        assert 'cv_std_accuracy' in metrics
        assert 'cv_scores' in metrics
    
    def test_predict_not_trained(self, sample_training_data):
        """Test prediction without training."""
        X_train, _ = sample_training_data
        trainer = ModelTrainer(model_type='logistic_regression')
        
        with pytest.raises(RuntimeError, match="must be trained"):
            trainer.predict(X_train)
    
    def test_predict_after_training(self, sample_training_data):
        """Test prediction after training."""
        X_train, y_train = sample_training_data
        trainer = ModelTrainer(model_type='logistic_regression')
        
        trainer.train(X_train, y_train, validate=False)
        predictions = trainer.predict(X_train)
        
        assert len(predictions) == len(X_train)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, sample_training_data):
        """Test probability prediction."""
        X_train, y_train = sample_training_data
        trainer = ModelTrainer(model_type='logistic_regression')
        
        trainer.train(X_train, y_train, validate=False)
        probas = trainer.predict_proba(X_train)
        
        assert probas.shape == (len(X_train), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_get_feature_importance_random_forest(self, sample_training_data):
        """Test feature importance for Random Forest."""
        X_train, y_train = sample_training_data
        trainer = ModelTrainer(model_type='random_forest')
        
        trainer.train(X_train, y_train, validate=False)
        importance_df = trainer.get_feature_importance()
        
        assert importance_df is not None
        assert len(importance_df) == len(X_train.columns)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_get_feature_importance_not_trained(self):
        """Test feature importance before training."""
        trainer = ModelTrainer(model_type='random_forest')
        
        with pytest.raises(RuntimeError, match="must be trained"):
            trainer.get_feature_importance()
    
    @patch('joblib.dump')
    def test_save_model(self, mock_dump, sample_training_data):
        """Test saving model."""
        X_train, y_train = sample_training_data
        trainer = ModelTrainer(model_type='logistic_regression')
        trainer.train(X_train, y_train, validate=False)
        
        save_path = trainer.save_model(dataset_name='test')
        
        assert mock_dump.called
        assert isinstance(save_path, Path)
    
    def test_save_model_not_trained(self):
        """Test saving untrained model."""
        trainer = ModelTrainer(model_type='logistic_regression')
        
        with pytest.raises(RuntimeError, match="must be trained"):
            trainer.save_model(dataset_name='test')
    
    @patch('joblib.load')
    def test_load_model(self, mock_load, tmp_path):
        """Test loading model."""
        # Create mock model data
        mock_model_data = {
            'model': Mock(),
            'model_type': 'logistic_regression',
            'feature_names': ['f1', 'f2', 'f3'],
            'is_trained': True
        }
        mock_load.return_value = mock_model_data
        
        model_path = tmp_path / "test_model.joblib"
        model_path.touch()  # Create file
        
        trainer = ModelTrainer(model_type='logistic_regression')
        trainer.load_model(model_path)
        
        assert trainer.model_type == 'logistic_regression'
        assert trainer.is_trained is True
        assert trainer.feature_names == ['f1', 'f2', 'f3']
    
    def test_load_model_file_not_found(self, tmp_path):
        """Test loading non-existent model."""
        trainer = ModelTrainer(model_type='logistic_regression')
        non_existent_path = tmp_path / "non_existent.joblib"
        
        with pytest.raises(FileNotFoundError):
            trainer.load_model(non_existent_path)


class TestTrainAllModels:
    """Tests for train_all_models function."""
    
    @patch('src.model_trainer.ModelTrainer.save_model')
    @patch('src.model_trainer.ModelTrainer.train')
    def test_train_all_models(self, mock_train, mock_save, sample_training_data):
        """Test training all model types."""
        X_train, y_train = sample_training_data
        mock_train.return_value = {'train_accuracy': 0.85}
        mock_save.return_value = Path('/fake/path')
        
        model_types = ['logistic_regression', 'random_forest']
        trained_models = train_all_models(
            X_train, y_train,
            dataset_name='test',
            model_types=model_types
        )
        
        assert len(trained_models) == len(model_types)
        for model_type in model_types:
            assert model_type in trained_models
