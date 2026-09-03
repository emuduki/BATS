# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, '.')

from ai.training.data_preparation import HistoricalDataLoader

loader = HistoricalDataLoader()
df = loader.generate_synthetic_data(n_samples=100000)

from ai.models.classical.xgboost_model import XGBoostModel, create_xgboost_config
from ai.training.data_preparation import DataPreparationPipeline

horizon = "30s"
config = create_xgboost_config(horizon)
model = XGBoostModel(config)

pipeline = DataPreparationPipeline(loader)
data_info = pipeline.prepare_training_data(df, horizon)

print('Training on {} samples...'.format(len(data_info['X_train'])))
metrics = model.train(data_info['X_train'], data_info['y_train'], 
                     data_info['X_val'], data_info['y_val'])

print('Accuracy: {:.4f}'.format(metrics.accuracy))
print('ROC-AUC: {:.4f}'.format(metrics.roc_auc))
print('Win Rate: {:.4f}'.format(metrics.win_rate))

import os
os.makedirs('ai/models', exist_ok=True)
model.save('ai/models/60s_xgboost.pkl')
print('Model saved to ai/models/60s_xgboost.pkl')
