"""
Model inference engine for Phase 3 price-direction prediction.
Handles real-time predictions and integration with trading system.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import joblib
from ai.models.base import BaseModel, ModelMetrics, ModelConfig, ModelMetrics
from ai.models.classical.xgboost_model import XGBoostModel, create_xgboost_config
from ai.models.deep_learning.lstm_model import LSTMWrapper, create_lstm_config, LSTMModel


class AIPredictor:
    """
    Main AI prediction engine that integrates multiple models
    across different horizons for binary options trading.
    """
    
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self.models: Dict[str, Dict[str, BaseModel]] = {}
        self.calibrators: Dict[str, Any] = {}
        self.feature_columns: Dict[str, List[str]] = {}
        self.active_horizon: Optional[str] = None
        self.prediction_history: List[Dict] = []
        
    def load_models(self, horizons: List[str] = None) -> None:
        """
        Load all trained models from disk.
        """
        if horizons is None:
            horizons = list(HORIZON_CONFIGS.keys())
        
        for horizon in horizons:
            self.models[horizon] = {}
            
            # Load XGBoost model
            xgb_path = self.model_dir / f"{horizon}_xgboost.pkl"
            if xgb_path.exists():
                model = XGBoostModel(create_xgboost_config(horizon))
                model.load(str(xgb_path))
                self.models[horizon]['xgboost'] = model
                self.feature_columns[horizon] = model.feature_columns if hasattr(model, 'feature_columns') else []
            
            # Load LSTM model
            lstm_path = self.model_dir / f"{horizon}_lstm.pkl"
            if lstm_path.exists():
                model = LSTMWrapper(create_lstm_config(horizon))
                model.load(str(lstm_path))
                self.models[horizon]['lstm'] = model
    
    def predict(
        self,
        features: np.ndarray,
        horizon: str = "60s",
        model_type: str = "xgboost",
        apply_calibration: bool = True
    ) -> Dict[str, any]:
        """
        Make prediction for given horizon and features.
        
        Args:
            features: Feature vector
            horizon: Prediction horizon
            model_type: Which model to use
            apply_calibration: Whether to apply probability calibration
            
        Returns:
            Dictionary with prediction results
        """
        if horizon not in self.models or model_type not in self.models[horizon]:
            return {"error": f"No {model_type} model loaded for {horizon}"}
        
        model = self.models[horizon][model_type]
        
        if not model.is_trained:
            return {"error": f"{model_type} model not trained"}
        
        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        if model_type == "xgboost":
            raw_probs = model.predict_proba(features)
            confidence = float(np.max(raw_probs))
            pred_class = int(model.predict(features, threshold=model.config.threshold)[0])
            
            result = {
                'horizon': horizon,
                'prediction': 'UP' if pred_class == 1 else 'DOWN',
                'probability_up': float(raw_probs[0, 1]),
                'probability_down': float(raw_probs[0, 0]),
                'confidence': confidence,
                'threshold': model.config.threshold
            }
            
            # Apply calibration if enabled and available
            if apply_calibration and model.calibrator is not None:
                calibrated_probs = model.calibrator.predict_proba(features)[0]
                result['probability_up'] = float(calibrated_probs[1])
                result['probability_down'] = float(calibrated_probs[0])
                
        elif model_type == "lstm":
            # Handle sequence input for LSTM
            if features.ndim == 1:
                # Need to pad or create sequence from last features
                seq_len = model.sequence_length if hasattr(model, 'sequence_length') else 60
                features = features.reshape(1, -1) if len(features) > 0 else np.zeros((1,))
                
                # Create simplified prediction
                raw_probs = model.predict_proba(features)
                confidence = float(np.max(raw_probs))
                pred_class = int(model.predict(features, threshold=0.5)[0])
                
                result = {
                    'horizon': horizon,
                    'prediction': 'UP' if pred_class == 1 else 'DOWN',
                    'probability_up': float(raw_probs[0, 1]),
                    'probability_down': float(raw_probs[0, 0]),
                    'confidence': confidence,
                    'threshold': 0.5
                }
        
        # Record prediction in history
        self.prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'horizon': horizon,
            'prediction': result['prediction'],
            'probability_up': result['probability_up'],
            'probability_down': result['probability_down'],
            'confidence': result['confidence'],
            'features_count': len(features[0]) if len(features) > 0 else 0
        })
        
        # Keep history manageable
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        return result
    
    def get_prediction_summary(self, horizon: Optional[str] = None) -> Dict[str, any]:
        """Get prediction statistics and history."""
        if horizon:
            history = [p for p in self.prediction_history if p['horizon'] == horizon]
        else:
            history = self.prediction_history
        
        if not history:
            return {"message": "No prediction history available"}
        
        # Calculate statistics
        total = len(history)
        up_preds = sum(1 for p in history if p['prediction'] == 'UP')
        down_preds = sum(1 for p in history if p['prediction'] == 'DOWN')
        avg_confidence = np.mean([p['confidence'] for p in history])
        
        # Recent predictions
        recent = history[-20:] if len(history) >= 20 else history
        recent_accuracy = 0
        
        # Count actual vs predicted outcomes if available
        if 'actual_outcome' in recent[0]:
            correct = sum(1 for p in recent if p.get('actual_outcome') == p['prediction'])
            recent_accuracy = correct / len(recent) if recent else 0
        
        return {
            'total_predictions': total,
            'up_predictions': up_preds,
            'down_predictions': down_preds,
            'up_percentage': up_preds / total * 100 if total > 0 else 0,
            'avg_confidence': avg_confidence,
            'recent_accuracy': recent_accuracy,
            'recent_predictions': len(recent)
        }
    
    def predict_live_signal(
        self,
        current_price: float,
        rsi: float,
        ema_9: float,
        ema_21: float,
        ema_50: float,
        macd: float,
        macd_signal: float,
        macd_hist: float,
        volatility: float,
        return_1: float,
        return_5: float,
        return_30: float,
        candle_body: float,
        upper_wick: float,
        lower_wick: float,
        horizon: str = "60s"
    ) -> Dict[str, any]:
        """
        Create prediction from live market data inputs.
        
        This is the main interface for integration with Phase 4 decision system.
        """
        # Create feature vector
        feature_vector = np.array([
            current_price,
            rsi,
            ema_9,
            ema_21,
            ema_50,
            macd,
            macd_signal,
            macd_hist,
            volatility,
            return_1,
            return_5,
            return_30,
            candle_body,
            upper_wick,
            lower_wick
        ]).reshape(1, -1)
        
        # Make prediction for 60s horizon
        result = self.predict(feature_vector, horizon="60s", model_type="xgboost")
        
        # Enhance with live signal metadata
        signal = {
            'horizon': result.get('horizon', '60s'),
            'direction': result.get('prediction', 'NEUTRAL'),
            'confidence': result.get('confidence', 0.5),
            'probability_up': result.get('probability_up', 0.5),
            'probability_down': result.get('probability_down', 0.5),
            
            # Input metadata for Phase 4
            'input_features': {
                'price': current_price,
                'rsi': rsi,
                'ema_trend': 'bullish' if ema_9 > ema_21 else 'bearish',
                'macd_hist': macd_hist,
                'volatility': volatility,
                'recent_returns': [return_1, return_5, return_30],
                'candle_pattern': 'bullish' if candle_body > 0 else 'bearish'
            },
            
            # Trading metadata
            'recommended_stake_pct': min(result['confidence'] * 2, 5),  # Scale confidence to position size
            'suggested_duration': horizon,
            'signal_metadata': result.get('threshold', 0.55)
        }
        
        return signal


# Global predictor instance
ai_predictor = AIPredictor()