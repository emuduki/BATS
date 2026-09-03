"""
Model training pipeline for Phase 3.
Handles training, validation, and model saving for all horizons.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from ai.datasets.horizon_configs import HORIZON_CONFIGS, HorizonConfig
from ai.training.data_preparation import DataPreparationPipeline, HistoricalDataLoader
from ai.models.base import BaseModel, ModelMetrics, ModelConfig
from ai.models.classical.xgboost_model import XGBoostModel, create_xgboost_config
from ai.models.deep_learning.lstm_model import LSTMWrapper, create_lstm_config


class ModelTrainer:
    """
    Complete training pipeline for Phase 3 models.
    Handles multiple horizons and model types with proper validation.
    """
    
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_results: Dict[str, Dict[str, ModelMetrics]] = {}
        self.models: Dict[str, Dict[str, BaseModel]] = {}
        self.data_loader = HistoricalDataLoader()
        self.pipeline = DataPreparationPipeline(self.data_loader)
    
    def train_horizon(
        self,
        df: pd.DataFrame,
        horizon: str,
        model_type: str = "xgboost",
        save_model: bool = True
    ) -> Tuple[BaseModel, ModelMetrics, Dict[str, any]]:
        """
        Train a model for a specific horizon.
        
        Args:
            df: DataFrame with OHLCV data
            horizon: Horizon name (e.g., "60s")
            model_type: "xgboost", "logistic", "lstm"
            save_model: Whether to save the trained model
            
        Returns:
            (model, metrics, data_info)
        """
        print(f"\n{'='*60}")
        print(f"Training {model_type} model for {horizon}")
        print(f"{'='*60}")
        
        # Prepare data
        data_info = self.pipeline.prepare_training_data(df, horizon)
        config = data_info['horizon_config']
        
        # Create model
        if model_type == "xgboost":
            model = XGBoostModel(create_xgboost_config(horizon))
        elif model_type == "lstm":
            model = LSTMWrapper(create_lstm_config(horizon))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        print(f"Training on {len(data_info['X_train'])} samples...")
        metrics = model.train(
            data_info['X_train'], data_info['y_train'],
            data_info['X_val'], data_info['y_val']
        )
        
        # Evaluate on test set
        test_metrics = self._evaluate_on_test(model, data_info)
        
        # Save model
        if save_model:
            model_path = self.model_dir / f"{horizon}_{model_type}.pkl"
            model.save(str(model_path))
            print(f"Model saved to: {model_path}")
        
        # Store results
        if horizon not in self.training_results:
            self.training_results[horizon] = {}
        self.training_results[horizon][model_type] = test_metrics
        
        if horizon not in self.models:
            self.models[horizon] = {}
        self.models[horizon][model_type] = model
        
        print(f"\nTest Results for {horizon} - {model_type}:")
        print(f"  Accuracy: {test_metrics.accuracy:.4f}")
        print(f"  ROC-AUC: {test_metrics.roc_auc:.4f}")
        print(f"  Win Rate: {test_metrics.win_rate:.4f}")
        
        return model, test_metrics, data_info
    
    def train_all_horizons(
        self,
        df: pd.DataFrame,
        model_types: List[str] = None,
        save_models: bool = True
    ) -> Dict[str, Dict[str, ModelMetrics]]:
        """
        Train models for all configured horizons.
        
        Args:
            df: DataFrame with OHLCV data
            model_types: List of model types to train (default: ["xgboost"])
            save_models: Whether to save trained models
            
        Returns:
            Dictionary of metrics for each horizon and model type
        """
        if model_types is None:
            model_types = ["xgboost"]
        
        results = {}
        
        for horizon in HORIZON_CONFIGS:
            try:
                horizon_results = {}
                for model_type in model_types:
                    model, metrics, _ = self.train_horizon(
                        df, horizon, model_type, save_models
                    )
                    horizon_results[model_type] = metrics
                
                results[horizon] = horizon_results
            except Exception as e:
                print(f"Error training {horizon}: {e}")
                results[horizon] = {"error": str(e)}
        
        self.training_results = results
        return results
    
    def train_walk_forward(
        self,
        df: pd.DataFrame,
        horizon: str,
        model_type: str = "xgboost",
        window_size: int = 1000,
        step_size: int = 200
    ) -> List[Dict[str, any]]:
        """
        Train using walk-forward validation.
        
        Returns list of training results for each window.
        """
        print(f"\nWalk-forward training for {horizon} - {model_type}")
        
        prepared_splits = self.pipeline.prepare_walk_forward_data(
            df, horizon, window_size, step_size
        )
        
        results = []
        for i, split in enumerate(prepared_splits):
            print(f"\nWindow {i+1}/{len(prepared_splits)}")
            
            # Create model
            if model_type == "xgboost":
                model = XGBoostModel(create_xgboost_config(horizon))
            elif model_type == "lstm":
                model = LSTMWrapper(create_lstm_config(horizon))
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train
            metrics = model.train(
                split['X_train'], split['y_train'],
                split['X_val'], split['y_val']
            )
            
            results.append({
                'window': i,
                'train_metrics': metrics,
                'model': model,
                'split': split
            })
        
        return results
    
    def _evaluate_on_test(
        self,
        model: BaseModel,
        data_info: Dict[str, any]
    ) -> ModelMetrics:
        """Evaluate model on test set."""
        X_test = data_info['X_test']
        y_test = data_info['y_test']
        
        test_probs = model.predict_proba(X_test)
        test_preds = model.predict(X_test)
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, log_loss
        )
        
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, test_preds),
            precision=precision_score(y_test, test_preds, zero_division=0),
            recall=recall_score(y_test, test_preds, zero_division=0),
            f1_score=f1_score(y_test, test_preds, zero_division=0),
            roc_auc=roc_auc_score(y_test, test_probs[:, 1]),
            log_loss=log_loss(y_test, test_probs)
        )
        
        # Calculate trading metrics
        if 'close' in data_info['test_df'].columns:
            price_data = data_info['test_df']['close'].values
            trading_metrics = model.evaluate_trading_metrics(
                X_test, y_test, price_data
            )
            metrics.win_rate = trading_metrics.win_rate
            metrics.max_drawdown = trading_metrics.max_drawdown
            metrics.sharpe_ratio = trading_metrics.sharpe_ratio
        
        return metrics
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of all training results."""
        rows = []
        
        for horizon, models in self.training_results.items():
            for model_type, metrics in models.items():
                if isinstance(metrics, ModelMetrics):
                    rows.append({
                        'Horizon': horizon,
                        'Model': model_type,
                        'Accuracy': metrics.accuracy,
                        'ROC-AUC': metrics.roc_auc,
                        'Win Rate': metrics.win_rate,
                        'Max DD': metrics.max_drawdown,
                        'Sharpe': metrics.sharpe_ratio
                    })
        
        return pd.DataFrame(rows)
    
    def save_training_results(self, path: str = "training_results.json") -> None:
        """Save training results to JSON file."""
        results = {}
        
        for horizon, models in self.training_results.items():
            results[horizon] = {}
            for model_type, metrics in models.items():
                if isinstance(metrics, ModelMetrics):
                    results[horizon][model_type] = metrics.to_dict()
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Training results saved to: {path}")


def train_models_from_file(
    data_file: str,
    horizon: str = "60s",
    model_type: str = "xgboost"
) -> Tuple[BaseModel, ModelMetrics]:
    """
    Convenience function to train models from a data file.
    
    Args:
        data_file: Path to CSV file with OHLCV data
        horizon: Horizon to train for
        model_type: Type of model to train
        
    Returns:
        (trained_model, metrics)
    """
    loader = HistoricalDataLoader()
    df = loader.load_csv_data(data_file)
    
    trainer = ModelTrainer()
    model, metrics, _ = trainer.train_horizon(df, horizon, model_type)
    
    return model, metrics


def train_all_models(
    data_file: str,
    horizons: List[str] = None,
    model_types: List[str] = None
) -> Dict[str, Dict[str, ModelMetrics]]:
    """
    Train all models for all horizons from a data file.
    
    Args:
        data_file: Path to CSV file with OHLCV data
        horizons: List of horizons to train (default: all)
        model_types: List of model types (default: ["xgboost"])
        
    Returns:
        Dictionary of training results
    """
    if horizons is None:
        horizons = list(HORIZON_CONFIGS.keys())
    
    if model_types is None:
        model_types = ["xgboost"]
    
    loader = HistoricalDataLoader()
    df = loader.load_csv_data(data_file)
    
    trainer = ModelTrainer()
    results = trainer.train_all_horizons(df, model_types)
    
    return results