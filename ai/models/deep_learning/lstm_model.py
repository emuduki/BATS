"""
LSTM model implementation for Phase 3 price-direction prediction.
Handles temporal sequences for capturing time-series patterns in market data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Tuple
from ai.models.base import BaseModel, ModelMetrics, ModelConfig


class LSTMModel(nn.Module):
    """
    LSTM network for temporal sequence processing.
    Designed for financial time-series prediction with dropout and batch norm.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            hidden_size * self.num_directions,
            2  # Binary classification: DOWN, UP
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            out = torch.cat((lstm_out[:, -1, :self.hidden_size], 
                           lstm_out[:, -1, self.hidden_size:]), dim=1)
        else:
            out = lstm_out[:, -1, :]
        
        out = self.dropout(out)
        return self.fc(out)


class LSTMWrapper(BaseModel):
    """
    Wrapper for LSTM model that handles sequence creation and training.
    Provides consistent interface with other models in the system.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = config.feature_count if hasattr(config, 'sequence_length') else 60
        self.hidden_size = getattr(config, 'hidden_size', 64)
        self.num_layers = getattr(config, 'num_layers', 2)
        self.dropout = getattr(config, 'dropout', 0.2)
        self.batch_size = getattr(config, 'batch_size', 64)
        self.epochs = getattr(config, 'epochs', 50)
        self.learning_rate = getattr(config, 'learning_rate', 0.001)
        
        # Will be set during training
        self.model = None
    
    def _create_sequences(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert flat features into temporal sequences.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            X_seq: Sequence tensor of shape (n_samples - seq_len, seq_len, n_features)
            y_seq: Labels for sequences
        """
        n_samples = X.shape[0]
        if n_samples < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} samples, got {n_samples}"
            )
        
        sequences = []
        labels = []
        
        for i in range(self.sequence_length, n_samples):
            seq = X[i - self.sequence_length:i]
            sequences.append(seq)
            labels.append(X[i, -1] if X.shape[1] > self.sequence_length else 0)  # Last feature as label proxy
        
        X_seq = np.array(sequences)
        y_seq = np.array(labels)
        
        return X_seq, y_seq
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> ModelMetrics:
        """Train LSTM model with early stopping."""
        # Create sequences
        X_seq_train, y_seq_train = self._create_sequences(X_train)
        X_seq_val, y_seq_val = None, None
        
        if X_val is not None and y_val is not None:
            X_seq_val, y_seq_val = self._create_sequences(X_val)
        
        # Initialize model
        input_size = X_train.shape[1]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=False
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_seq_train).to(self.device),
            torch.LongTensor(y_seq_train).to(self.device)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = None
        if X_seq_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_seq_val).to(self.device),
                torch.LongTensor(y_seq_val).to(self.device)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size
            )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            self.model.train()
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                val_loss = val_loss / len(val_loader)
                self.model.train()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_state_dict = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= 5:  # Early stopping patience
                break
        
        # Load best model
        if 'best_state_dict' in locals():
            self.model.load_state_dict(best_state_dict)
        
        self.is_trained = True
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            train_probs = self.predict_proba(X_train)
            train_preds = self.predict(X_train)
        
        self.metrics = ModelMetrics(
            accuracy=accuracy_score(y_seq_train, train_preds),
            precision=precision_score(y_seq_train, train_preds, zero_division=0),
            recall=recall_score(y_seq_train, train_preds, zero_division=0),
            f1_score=f1_score(y_seq_train, train_preds, zero_division=0),
            roc_auc=roc_auc_score(y_seq_train, train_probs[:, 1]) if len(np.unique(y_seq_train)) > 1 else 0.5
        )
        
        # Validation metrics
        if X_seq_val is not None and y_seq_val is not None:
            with torch.no_grad():
                val_probs = self.predict_proba(X_val)
                val_preds = self.predict(X_val)
            
            self.metrics.update(ModelMetrics(
                accuracy=accuracy_score(y_seq_val, val_preds),
                precision=precision_score(y_seq_val, val_preds, zero_division=0),
                recall=recall_score(y_seq_val, val_preds, zero_division=0),
                f1_score=f1_score(y_seq_val, val_preds, zero_division=0),
                roc_auc=roc_auc_score(y_seq_val, val_probs[:, 1]) if len(np.unique(y_seq_val)) > 1 else 0.5
            ))
        
        return self.metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability predictions for sequences."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained")
        
        X_seq, _ = self._create_sequences(X)
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X_seq).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Pad beginning with neutral probabilities
        padding = np.full((self.sequence_length, 2), 0.5)
        return np.vstack([padding, probs])
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return (probs[:, 1] > threshold).astype(int)
    
    def get_feature_importance(self) -> np.ndarray:
        """LSTM doesn't provide direct feature importance - return zeros."""
        return np.zeros(self.config.feature_count)


def create_lstm_config(horizon: str) -> ModelConfig:
    """Create appropriate config for LSTM model based on horizon."""
    horizon_map = {
        "30s": {"sequence_length": 30, "hidden_size": 32},
        "60s": {"sequence_length": 60, "hidden_size": 64},
        "120s": {"sequence_length": 90, "hidden_size": 96},
        "300s": {"sequence_length": 120, "hidden_size": 128}
    }
    
    params = horizon_map.get(horizon, {"sequence_length": 60, "hidden_size": 64})
    
    return ModelConfig(
        model_name=f"LSTM_{horizon}",
        horizon=horizon,
        feature_count=35,
        sequence_length=params["sequence_length"],
        hidden_size=params["hidden_size"],
        threshold=0.55,
        version="1.0"
    )