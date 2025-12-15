from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import os


@dataclass
class DataConfig:
    """Data configuration."""
    root: str = './data'
    train_size: int = 10000
    val_size: int = 1000
    test_size: int = 1000
    sequence_length: int = 20
    frame_size: int = 64
    num_digits: int = 2
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = 'disentangled'  # 'baseline', 'lstm', 'disentangled'
    
    # Latent dimensions
    latent_dim: int = 256  # For baseline and LSTM
    content_dim: int = 128  # For disentangled
    motion_dim: int = 128  # For disentangled
    
    # CNN dimensions
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    motion_channels: List[int] = field(default_factory=lambda: [16, 32])
    feature_dim: int = 512
    
    # LSTM dimensions
    lstm_hidden: int = 256
    lstm_layers: int = 1
    bidirectional: bool = True
    
    # Other
    in_channels: int = 1
    sequence_length: int = 20
    content_aggregation: str = 'mean'  # 'mean' or 'middle'


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimizer
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-5
    
    # Loss weights
    beta_vae: float = 4.0  # β for KL divergence
    beta_content: Optional[float] = None  # Separate β for content KL
    beta_motion: Optional[float] = None  # Separate β for motion KL
    temporal_weight: float = 0.0
    independence_weight: float = 0.0
    recon_loss_type: str = 'mse'  # 'mse' or 'bce'
    
    # KL annealing and collapse prevention
    use_kl_annealing: bool = True
    warmup_epochs: int = 10
    annealing_type: str = 'linear'  # 'linear' or 'cyclical'
    kl_start_epoch: int = 0  # Epoch to start including KL loss (delayed KL)
    use_free_bits: bool = True  # Enable free bits to prevent KL collapse
    free_bits: float = 0.5  # Minimum KL per dimension (prevents collapse)
    
    # Training schedule
    num_epochs: int = 100
    gradient_clip: float = 1.0
    
    # Learning rate schedule
    lr_scheduler: str = 'exponential'  # 'exponential', 'cosine', 'none'
    lr_decay: float = 0.98
    lr_min: float = 1e-6
    
    # Device
    device: str = 'mps'  # 'mps', 'cuda', 'cpu'
    
    # Saving
    save_every: int = 10
    log_every: int = 100
    visualize_every: int = 5
    checkpoint_dir: str = './checkpoints'
    results_dir: str = './results'


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = 'experiment'
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'data': self.data.__dict__.copy(),
            'model': self.model.__dict__.copy(),
            'training': self.training.__dict__.copy()
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls(name=data['name'])
        config.data = DataConfig(**data['data'])
        config.model = ModelConfig(**data['model'])
        config.training = TrainingConfig(**data['training'])
        return config


def get_baseline_config(name: str = 'baseline_vae') -> ExperimentConfig:
    """Get configuration for Baseline VAE."""
    config = ExperimentConfig(name=name)
    config.model.model_type = 'baseline'
    config.model.latent_dim = 256
    return config


def get_lstm_config(name: str = 'lstm_vae') -> ExperimentConfig:
    """Get configuration for LSTM-VAE."""
    config = ExperimentConfig(name=name)
    config.model.model_type = 'lstm'
    config.model.latent_dim = 256
    config.model.lstm_hidden = 256
    config.model.bidirectional = True
    return config


def get_disentangled_config(name: str = 'disentangled_vae') -> ExperimentConfig:
    """Get configuration for Disentangled VAE."""
    config = ExperimentConfig(name=name)
    config.model.model_type = 'disentangled'
    config.model.content_dim = 128
    config.model.motion_dim = 128
    config.training.independence_weight = 0.1
    return config


def get_ablation_configs() -> Dict[str, ExperimentConfig]:
    """Get configurations for ablation studies."""
    configs = {}
    
    # Different β values
    for beta in [1.0, 2.0, 4.0, 8.0, 16.0]:
        config = get_disentangled_config(f'ablation_beta_{beta}')
        config.training.beta_vae = beta
        configs[f'beta_{beta}'] = config
    
    # Different latent dimensions
    for dim in [64, 128, 256]:
        config = get_disentangled_config(f'ablation_dim_{dim}')
        config.model.content_dim = dim
        config.model.motion_dim = dim
        configs[f'dim_{dim}'] = config
    
    # Content aggregation method
    for agg in ['mean', 'middle']:
        config = get_disentangled_config(f'ablation_agg_{agg}')
        config.model.content_aggregation = agg
        configs[f'agg_{agg}'] = config
    
    # LSTM configuration
    for bidir in [True, False]:
        config = get_disentangled_config(f'ablation_bidir_{bidir}')
        config.model.bidirectional = bidir
        configs[f'bidir_{bidir}'] = config
    
    return configs


# Default configurations
BASELINE_CONFIG = get_baseline_config()
LSTM_CONFIG = get_lstm_config()
DISENTANGLED_CONFIG = get_disentangled_config()

if __name__ == '__main__':
    # Test configurations
    print("Testing configurations")
    
    config = get_disentangled_config('test')
    print(f"\nDisentangled config:")
    print(f"  Model type: {config.model.model_type}")
    print(f"  Content dim: {config.model.content_dim}")
    print(f"  Motion dim: {config.model.motion_dim}")
    print(f"  Beta: {config.training.beta_vae}")
    
    # Save and load
    config.save('/tmp/test_config.json')
    loaded = ExperimentConfig.load('/tmp/test_config.json')
    print(f"\nLoaded config name: {loaded.name}")
    
    # Ablation configs
    ablation = get_ablation_configs()
    print(f"\nAblation studies: {list(ablation.keys())}")
    
    print("\nConfiguration test passed!")


