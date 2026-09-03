"""
XGBoost model implementation for Phase 3 price-direction prediction.
Provides XGBoost with proper calibration and evaluation for binary classification.
"""

import xgboost as xgb
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ai.models.base import BaseModel, ModelMetrics, ModelConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss


class XGBoostModel(BaseModel):
    """
    XGBoost classifier optimized for binary options price-direction prediction.
    Supports probabilistic outputs with calibration.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        n_est = (config.feature_count * 10) if config.feature_count else 200
        self.model = xgb.XGBClassifier(
            n_estimators=n_est,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> ModelMetrics:
        """Train XGBoost model with validation."""
        if X_val is not None and y_val is not None:
            try:
                self.model.set_params(early_stopping_rounds=20)
            except Exception:
                pass
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)
        
        self.is_trained = True
        
        # Calculate training metrics
        train_probs = self.predict_proba(X_train)
        train_preds = self.predict(X_train)
        
        self.metrics = ModelMetrics(
            accuracy=accuracy_score(y_train, train_preds),
            precision=precision_score(y_train, train_preds, zero_division=0),
            recall=recall_score(y_train, train_preds, zero_division=0),
            f1_score=f1_score(y_train, train_preds, zero_division=0),
            roc_auc=roc_auc_score(y_train, train_probs[:, 1]),
            log_loss=log_loss(y_train, train_probs)
        )
        
        # Calculate validation metrics if available
        if X_val is not None and y_val is not None:
            val_probs = self.predict_proba(X_val)
            val_preds = self.predict(X_val)
            
            self.metrics.update(ModelMetrics(
                accuracy=accuracy_score(y_val, val_preds),
                precision=precision_score(y_val, val_preds, zero_division=0),
                recall=recall_score(y_val, val_preds, zero_division=0),
                f1_score=f1_score(y_val, val_preds, zero_division=0),
                roc_auc=roc_auc_score(y_val, val_probs[:, 1]),
                log_loss=log_loss(y_val, val_probs)
            ))
        
        return self.metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability predictions [P(DOWN), P(UP)]."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        raw_probs = self.model.predict_proba(X)
        # XGBoost returns [P(class0), P(class1)] where class1 is '1' label
        # Our labels: 0 = DOWN, 1 = UP
        return raw_probs  # Already in correct format
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels with specified threshold."""
        probs = self.predict_proba(X)
        return (probs[:, 1] > threshold).astype(int)
    
    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance scores."""
        if self.is_trained:
            return self.model.feature_importances_
        return np.zeros(self.config.feature_count)
    
    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray, method: str = "isotonic") -> None:
        """Calibrate probabilities using isotonic regression."""
        from sklearn.calibration import CalibratedClassifierCV
        
        self.calibrator = CalibratedClassifierCV(
            estimator=self.model,
            method=method,
            cv=3
        )
        self.calibrator.fit(X_calib, y_calib)
        self.is_trained = True
    
    def predict_calibrated(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with calibrated probabilities."""
        if self.calibrator is None:
            return self.predict(X, threshold)
        
        raw_probs = self.model.predict_proba(X)
        calibrated_probs = self.calibrator.predict_proba(X)
        return (calibrated_probs[:, 1] > threshold).astype(int), calibrated_probs


class LogisticRegressionModel(BaseModel):
    """Placeholder Logistic Regression model class."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val=None, y_val=None) -> ModelMetrics:
        from sklearn.linear_model import LogisticRegression
        
        self.model = LogisticRegression(
            random_state=42,
            solver='lbfgs',
            max_iter=1000
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_preds = self.predict(X_train)
        val_preds = self.predict(X_val) if X_val is not None else None
        
        self.metrics = ModelMetrics(
            accuracy=accuracy_score(y_train, train_preds)
        )
        
        if val_preds is not None:
            self.metrics.update(ModelMetrics(
                accuracy=accuracy_score(y_val, val_preds)
            ))
        
        return self.metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] > threshold).astype(int)


def create_xgboost_config(horizon: str) -> ModelConfig:
    """Create appropriate config for XGBoost model based on horizon."""
    return ModelConfig(
        model_name=f"XGBoost_{horizon}",
        horizon=horizon,
        feature_count=35,
        threshold=0.55,  # Default trading threshold
        version="1.0"
    )


def create_logistic_regression_config(horizon: str) -> ModelConfig:
    """Create config for Logistic Regression model."""
    return ModelConfig(
        model_name=f"Logistic_{horizon}",
        horizon=horizon,
        feature_count=35,
        threshold=0.55,
        version="1.0"
    )