"""
Train AI models for BATS trading system.
Usage: python train_models.py
"""

import sys
sys.path.insert(0, '.')

from ai.training.data_preparation import HistoricalDataLoader
from ai.training.trainer import ModelTrainer

def main():
    print("=" * 60)
    print("BATS MODEL TRAINING")
    print("=" * 60)
    
    # Load or generate training data
    loader = HistoricalDataLoader(data_dir="data/historical")
    
    try:
        # Try to load existing data
        df = loader.load_csv_data("market_data.csv")
        print(f"Loaded {len(df)} rows from data/historical/market_data.csv")
    except FileNotFoundError:
        # Generate synthetic data for training
        print("Generating synthetic training data...")
        df = loader.generate_synthetic_data(n_samples=50000)
        print(f"Generated {len(df)} synthetic samples")
    
    # Initialize trainer
    trainer = ModelTrainer(model_dir="ai/models")
    
    # Train XGBoost models for key horizons
    horizons_to_train = ["15s", "30s", "60s"]
    
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    for horizon in horizons_to_train:
        try:
            print(f"\n>>> Training for {horizon} horizon...")
            model, metrics, _ = trainer.train_horizon(
                df, 
                horizon=horizon, 
                model_type="xgboost",
                save_model=True
            )
            print(f"    Accuracy: {metrics.accuracy:.2%}")
            print(f"    ROC-AUC:  {metrics.roc_auc:.4f}")
            print(f"    Win Rate: {metrics.win_rate:.2%}")
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Save training summary
    summary = trainer.get_training_summary()
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    
    # Save results
    trainer.save_training_results("training_results.json")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Models saved to: ai/models/")

if __name__ == "__main__":
    main()
