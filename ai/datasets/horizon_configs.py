"""
Configuration for multi-horizon price-direction prediction models.
Each horizon defines specific requirements for label creation and validation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json

@dataclass
class HorizonConfig:
    name: str
    horizon_seconds: int
    horizon_description: str
    label_threshold_pct: float = 0.05  # Minimum meaningful price movement
    min_feature_lookback: int = 60     # Number of candles for features
    min_training_samples: int = 5000   # Minimum valid samples required
    prediction_threshold: float = 0.65  # Minimum model confidence to trade
    early_stopping_patience: int = 10
    validation_frequency: int = 5  # Validate every N epochs

    @property
    def label_columns(self) -> list:
        return [f'label_{self.name}', f'future_price_{self.name}', f'future_return_{self.name}']

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'horizon_seconds': self.horizon_seconds,
            'horizon_description': self.horizon_description,
            'label_threshold_pct': self.label_threshold_pct,
            'min_feature_lookback': self.min_feature_lookback,
            'min_training_samples': self.min_training_samples,
            'prediction_threshold': self.prediction_threshold,
            'early_stopping_patience': self.early_stopping_patience,
            'validation_frequency': self.validation_frequency,
        }

HORIZON_CONFIGS: Dict[str, HorizonConfig] = {
    "30s": HorizonConfig(
        name="30s",
        horizon_seconds=30,
        horizon_description="30-second expiry contracts",
        label_threshold_pct=0.05,
        min_feature_lookback=50,
        min_training_samples=10000,
        prediction_threshold=0.70
    ),
    "60s": HorizonConfig(
        name="60s",
        horizon_seconds=60,
        horizon_description="60-second expiry contracts - most liquid",
        label_threshold_pct=0.10,
        min_feature_lookback=60,
        min_training_samples=5000,
        prediction_threshold=0.70
    ),
    "120s": HorizonConfig(
        name="120s",
        horizon_seconds=120,
        horizon_description="120-second expiry contracts",
        label_threshold_pct=0.15,
        min_feature_lookback=100,
        min_training_samples=25000,
        prediction_threshold=0.75
    ),
    "300s": HorizonConfig(
        name="300s",
        horizon_seconds=300,
        horizon_description="300-second (5-minute) expiry contracts",
        label_threshold_pct=0.25,
        min_feature_lookback=200,
        min_training_samples=10000,
        prediction_threshold=0.75
    ),
}

def validate_config(config: HorizonConfig) -> bool:
    """Validate horizon configuration parameters"""
    if config.horizon_seconds < 30:
        return False
    if not 0 <= config.label_threshold_pct <= 0.1:
        return False
    if config.min_training_samples < 1000:
        return False
    if not 0 <= config.prediction_threshold <= 1.0:
        return False
    return True

def load_configs_from_file(path: str) -> Dict[str, HorizonConfig]:
    """Load horizon configs from JSON file"""
    with open(path, 'r') as f:
        data = json.load(f)
    
    configs = {}
    for name, cfg in data.items():
        configs[name] = HorizonConfig(**cfg)
        if not validate_config(configs[name]):
            raise ValueError(f"Invalid horizon configuration for {name}")
    
    return configs

def get_active_horizons() -> list:
    """Return list of valid horizon names"""
    return list(HORIZON_CONFIGS.keys())

def get_horizon_config(name: str) -> Optional[HorizonConfig]:
    """Get configuration for specific horizon"""
    return HORIZON_CONFIGS.get(name)