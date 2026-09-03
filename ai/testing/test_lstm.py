"""Test suite for LSTM model training and inference pipeline."""
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from ai.models.deep_learning.lstm_model import LSTMWrapper, LSTMModel, LSTMConfig
from ai.training.data_preparation import prepare_dataset
from ai.datasets.horizon_configs import HORIZON_CONFIGS
from ai.features.technical import create_all_features


@pytest.fixture
def sample_data():
    """Generate synthetic OHLCV data for testing."""
    n_samples = 5000
    dates = pd.date_range(start='2026-01-01', periods=n_samples, freq='1min')
    
    # Generate realistic price series
    base_price = 1250.0
    returns = np.random.randn(n_samples) * 0.0005
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n_samples) * 0.0001)),
        'low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.0001)),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n_samples)
    })
    
    return df


@pytest.fixture
def prepared_dataset(sample_data):
    """Prepare dataset with features and labels."""
    df = create_all_features(sample_data)
    df = prepare_dataset(df, "60s")
    return df


class TestLSTMModel:
    """Test suite for LSTM model training and inference."""

    def test_model_initialization(self):
        """Test that LSTM model can be initialized."""
        model = LSTMWrapper({
            "hidden_size": 32,
            "num_layers": 1,
            "epochs": 5,
            "sequence_length": 20
        })
        
        assert model is not None
        assert model.lstm_config.hidden_size == 32
        assert model.lstm_config.sequence_length == 20
        assert model.is_trained is False

    def test_sequence_creation(self):
        """Test that sequences are created correctly."""
        model = LSTMWrapper({"sequence_length": 10})
        
        X = np.random.randn(100, 5)
        sequences = model._create_sequences(X)
        
        assert sequences.shape[0] == 90  # 100 - 10 + 1
        assert sequences.shape[1] == 10  # sequence_length
        assert sequences.shape[2] == 5   # features

    def test_training_cycle(self, prepared_dataset):
        """Test complete training cycle."""
        data = prepared_dataset
        model = LSTMWrapper({
            "hidden_size": 32,
            "num_layers": 1,
            "epochs": 3,
            "batch_size": 64,
            "sequence_length": 20
        })
        
        metrics = model.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val']
        )
        
        assert model.is_trained is True
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' in metrics
        assert 0 <= metrics['train_accuracy'] <= 1
        assert 0 <= metrics['val_accuracy'] <= 1

    def test_inference(self, prepared_dataset):
        """Test model inference."""
        data = prepared_dataset
        model = LSTMWrapper({
            "hidden_size": 32,
            "num_layers": 1,
            "epochs": 2,
            "sequence_length": 20
        })
        
        # Train on small subset for speed
        model.train(
            data['X_train'][:500], data['y_train'][:500],
            data['X_val'][:100], data['y_val'][:100]
        )
        
        # Test prediction
        probs = model.predict_proba(data['X_test'][:100])
        preds = model.predict(data['X_test'][:100])
        
        assert probs.shape[0] == 100
        assert probs.shape[1] == 2  # Two classes
        assert preds.shape[0] == 100
        assert set(preds).issubset({0, 1})

    def test_save_load(self, prepared_dataset, tmp_path):
        """Test model serialization."""
        data = prepared_dataset
        model = LSTMWrapper({
            "hidden_size": 32,
            "num_layers": 1,
            "epochs": 2,
            "sequence_length": 20
        })
        
        model.train(
            data['X_train'][:500], data['y_train'][:500],
            data['X_val'][:100], data['y_val'][:100]
        )
        
        # Save model
        save_path = tmp_path / "test_lstm.pkl"
        model.save(str(save_path))
        assert save_path.exists()
        
        # Load model
        loaded_model = LSTMWrapper({})
        loaded_model.load(str(save_path))
        
        assert loaded_model.is_trained is True
        assert loaded_model.lstm_config.hidden_size == model.lstm_config.hidden_size

    def test_device_selection(self):
        """Test that device selection works correctly."""
        model = LSTMWrapper({})
        assert model.device in ['cuda', 'cpu']


class TestLSTMConfig:
    """Test LSTM configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LSTMConfig()
        assert config.hidden_size == 64
        assert config.num_layers == 2
        assert config.dropout == 0.2
        assert config.learning_rate == 0.001
        assert config.batch_size == 64
        assert config.epochs == 50
        assert config.sequence_length == 60

    def test_custom_config(self):
        """Test custom configuration."""
        config = LSTMConfig(
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
            learning_rate=0.0005
        )
        assert config.hidden_size == 128
        assert config.num_layers == 3
        assert config.dropout == 0.3
        assert config.learning_rate == 0.0005


class TestLSTMIntegration:
    """Test LSTM integration with existing pipeline."""

    def test_predictor_integration(self, prepared_dataset):
        """Test LSTM with predictor system."""
        from ai.inference.predictor import AIPredictor
        
        # Train a quick model
        data = prepared_dataset
        model = LSTMWrapper({
            "hidden_size": 32,
            "num_layers": 1,
            "epochs": 2,
            "sequence_length": 20
        })
        
        model.train(
            data['X_train'][:500], data['y_train'][:500],
            data['X_val'][:100], data['y_val'][:100]
        )
        
        # Save and load through predictor
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model.save(f.name)
            predictor = AIPredictor()
            predictor.models['60s'] = {'lstm': model}
            
            # Test prediction
            df = create_all_features(pd.read_csv("sample_historical.csv"))
            result = predictor.predict(df, "60s")
            
            assert 'horizon' in result
            assert 'ensemble' in result
            assert 'signal' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])