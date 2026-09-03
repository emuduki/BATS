"""
Base model classes for Phase 3 AI models.
Defines common interface for all prediction models.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import joblib
from pathlib import Path


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    log_loss: float = 0.0
    calibration_error: float = 0.0
    
    # Trading metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ModelMetrics':
        return cls(**data)


@dataclass
class ModelConfig:
    """Base configuration for models."""
    model_name: str
    horizon: str
    feature_count: int
    threshold: float = 0.5
    calibration_method: str = "isotonic"
    created_at: str = ""
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class BaseModel(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.metrics = ModelMetrics()
        self.calibrator = None
        self.feature_importance_: Optional[np.ndarray] = None
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> ModelMetrics:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability predictions for both classes."""
        pass
    
    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict class labels."""
        threshold = threshold or self.config.threshold
        probs = self.predict_proba(X)
        return (probs[:, 1] > threshold).astype(int)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with confidence scores."""
        probs = self.predict_proba(X)
        preds = self.predict(X)
        confidence = np.max(probs, axis=1)
        return preds, confidence
    
    def calibrate(
        self,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        method: str = "isotonic"
    ) -> None:
        """Calibrate probability outputs."""
        from sklearn.calibration import CalibratedClassifierCV
        
        if self.model is None:
            raise RuntimeError("Model must be trained before calibration")
        
        self.calibrator = CalibratedClassifierCV(
            estimator=self.model,
            method=method,
            cv=3
        )
        self.calibrator.fit(X_calib, y_calib)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'config': self.config,
            'metrics': self.metrics,
            'calibrator': self.calibrator,
            'feature_importance': self.feature_importance_
        }, path)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.config = data['config']
        self.metrics = data['metrics']
        self.calibrator = data.get('calibrator')
        self.feature_importance_ = data.get('feature_importance')
        self.is_trained = True
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        return self.feature_importance_
    
    def evaluate_trading_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        price_data: np.ndarray,
        payout_rate: float = 0.85
    ) -> ModelMetrics:
        """Evaluate model using trading simulation metrics."""
        probs = self.predict_proba(X)
        preds = self.predict(X)
        
        # Simulate trades
        returns = price_data * (2 * preds - 1)  # UP = +1, DOWN = -1
        payouts = np.where(preds == y, returns * payout_rate, -returns)
        
        # Trading metrics
        n_trades = len(payouts)
        n_wins = np.sum(preds == y)
        win_rate = n_wins / n_trades if n_trades > 0 else 0
        
        total_profit = np.sum(payouts)
        max_dd = np.minimum.accumulate(np.cumsum(payouts))
        max_drawdown = np.min(max_dd) / np.max(np.cumsum(payouts) + 1e-10)
        
        avg_return = np.mean(payouts)
        std_return = np.std(payouts)
        sharpe = avg_return / (std_return + 1e-10) if std_return > 0 else 0
        
        profit_factor = (
            np.sum(payouts[payouts > 0]) / 
            np.abs(np.sum(payouts[payouts < 0]) + 1e-10)
        )
        
        return ModelMetrics(
            accuracy=self.metrics.accuracy,
            precision=self.metrics.precision,
            recall=self.metrics.recall,
            f1_score=self.metrics.f1_score,
            roc_auc=self.metrics.roc_auc,
            log_loss=self.metrics.log_loss,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=abs(max_drawdown),
            sharpe_ratio=sharpe
        )


class EnsembleModel(BaseModel):
    """Ensemble of multiple models with weighted voting."""
    
    def __init__(self, config: ModelConfig, models: Dict[str, BaseModel], weights: Optional[Dict[str, float]] = None):
        super().__init__(config)
        self.models = models
        self.weights = weights or {k: 1.0 for k in models.keys()}
        self.weights = {k: v / sum(self.weights.values()) for k, v in self.weights.items()}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of model probabilities."""
        probs = np.zeros((X.shape[0], 2))
        
        for name, model in self.models.items():
            if model.is_trained:
                model_probs = model.predict_proba(X)
                probs += self.weights[name] * model_probs
        
        return probs
    
    def train(self, *args, **kwargs) -> ModelMetrics:
        """Ensemble doesn't need separate training."""
        self.is_trained = all(m.is_trained for m in self.models.values())
        return self.metrics