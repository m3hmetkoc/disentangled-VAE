"""
Training module for Video VAE models.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ExperimentConfig,
    get_baseline_config,
    get_lstm_config,
    get_disentangled_config,
    get_ablation_configs
)

from training.trainer import Trainer, create_trainer

__all__ = [
    'DataConfig',
    'ModelConfig',
    'TrainingConfig',
    'ExperimentConfig',
    'get_baseline_config',
    'get_lstm_config',
    'get_disentangled_config',
    'get_ablation_configs',
    'Trainer',
    'create_trainer'
]

