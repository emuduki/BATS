# -*- coding: utf-8 -*-
"""
Train XGBoost model using yfinance or synthetic data.
Usage: python train_simple.py [--symbol SYMBOL] [--period PERIOD] [--interval INTERVAL]
"""

import sys
import os
import argparse

sys.path.insert(0, '.')

from ai.training.data_preparation import HistoricalDataLoader
from ai.models.classical.xgboost_model import XGBoostModel, create_xgboost_config
from ai.training.data_preparation import DataPreparationPipeline


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model for BATS')
    parser.add_argument('--symbol', default='^GDAXI', help='Yahoo Finance symbol (e.g., ^GDAXI, BTC-USD, AAPL)')
    parser.add_argument('--period', default='2y', help='Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--interval', default='1m', help='Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)')
    parser.add_argument('--horizon', default='30s', help='Trading horizon (15s, 30s, 60s, 120s, 300s)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data instead of yfinance')
    args = parser.parse_args()

    print("=" * 60)
    print("BATS MODEL TRAINING")
    print("=" * 60)

    loader = HistoricalDataLoader()

    if args.synthetic:
        print("Generating synthetic data...")
        df = loader.generate_synthetic_data(n_samples=200000)
    else:
        try:
            df = loader.load_yfinance_data(symbol=args.symbol, period=args.period, interval=args.interval)
        except ImportError:
            print("yfinance not installed. Install with: pip install yfinance")
            print("Falling back to synthetic data...")
            df = loader.generate_synthetic_data(n_samples=200000)
        except Exception as e:
            print(f"Error loading yfinance data: {e}")
            print("Falling back to synthetic data...")
            df = loader.generate_synthetic_data(n_samples=200000)

    print(f"Data loaded: {len(df)} rows")

    config = create_xgboost_config(args.horizon)
    model = XGBoostModel(config)

    pipeline = DataPreparationPipeline(loader)
    try:
        data_info = pipeline.prepare_training_data(df, args.horizon)
        print(f"Training samples: {len(data_info['X_train'])}")
    except ValueError as e:
        print(f"Error preparing data: {e}")
        print("Try using synthetic data with --synthetic flag")
        return

    metrics = model.train(data_info['X_train'], data_info['y_train'],
                         data_info['X_val'], data_info['y_val'])

    print(f"\nAccuracy:  {metrics.accuracy:.4f}")
    print(f"ROC-AUC:   {metrics.roc_auc:.4f}")
    print(f"Win Rate:  {metrics.win_rate:.4f}")

    os.makedirs('ai/models', exist_ok=True)
    model_path = f'ai/models/{args.horizon}_xgboost.pkl'
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()