import argparse
import os
import sys
import torch
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import (
    ExperimentConfig,
    get_baseline_config,
    get_lstm_config,
    get_disentangled_config
)
from training.trainer import create_trainer
from data import get_moving_mnist_dataloaders


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # MPS doesn't have a separate seed function


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Video VAE models')
    
    # Config file (overrides other args)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (overrides other arguments)')
    
    # Model selection
    parser.add_argument('--model', type=str, default='disentangled',
                        choices=['baseline', 'lstm', 'disentangled'],
                        help='Model type to train')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    
    # Data
    parser.add_argument('--train-size', type=int, default=10000,
                        help='Number of training videos')
    parser.add_argument('--val-size', type=int, default=1000,
                        help='Number of validation videos')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    
    # Model architecture
    parser.add_argument('--latent-dim', type=int, default=256,
                        help='Latent dimension (for baseline and lstm)')
    parser.add_argument('--content-dim', type=int, default=128,
                        help='Content latent dimension (for disentangled)')
    parser.add_argument('--motion-dim', type=int, default=128,
                        help='Motion latent dimension (for disentangled)')
    parser.add_argument('--lstm-hidden', type=int, default=256,
                        help='LSTM hidden dimension')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=4.0,
                        help='Beta for KL divergence')
    parser.add_argument('--warmup', type=int, default=10,
                        help='KL warmup epochs')
    parser.add_argument('--no-kl-annealing', action='store_true',
                        help='Disable KL annealing')
    
    # Disentangled-specific
    parser.add_argument('--independence-weight', type=float, default=0.1,
                        help='Weight for independence loss')
    parser.add_argument('--content-agg', type=str, default='mean',
                        choices=['mean', 'middle'],
                        help='Content aggregation method')
    
    # Device
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device to train on')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--visualize-every', type=int, default=5,
                        help='Visualize every N epochs')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config from file if provided
    if args.config is not None:
        print(f"Loading config from: {args.config}")
        config = ExperimentConfig.load(args.config)
        # Allow device override from command line
        if args.device:
            config.training.device = args.device
    else:
        # Get base config
        if args.model == 'baseline':
            config = get_baseline_config()
        elif args.model == 'lstm':
            config = get_lstm_config()
        else:
            config = get_disentangled_config()
        
        # Set experiment name
        if args.name is not None:
            config.name = args.name
        else:
            config.name = f"{args.model}_vae"
        
        # Update config with command line arguments
        config.data.train_size = args.train_size
        config.data.val_size = args.val_size
        config.data.batch_size = args.batch_size
        config.data.num_workers = args.num_workers
        config.data.seed = args.seed
        
        config.model.latent_dim = args.latent_dim
        config.model.content_dim = args.content_dim
        config.model.motion_dim = args.motion_dim
        config.model.lstm_hidden = args.lstm_hidden
        config.model.content_aggregation = args.content_agg
        
        config.training.num_epochs = args.epochs
        config.training.learning_rate = args.lr
        config.training.beta_vae = args.beta
        config.training.warmup_epochs = args.warmup
        config.training.use_kl_annealing = not args.no_kl_annealing
        config.training.independence_weight = args.independence_weight
        config.training.device = args.device
        config.training.save_every = args.save_every
        config.training.visualize_every = args.visualize_every
    
    # Set seed
    set_seed(config.data.seed)
    
    # Print configuration
    print(f"Model type: {config.model.model_type}")
    print(f"\nData:")
    print(f"  Train size: {config.data.train_size}")
    print(f"  Val size: {config.data.val_size}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"\nModel:")
    if config.model.model_type == 'disentangled':
        print(f"  Content dim: {config.model.content_dim}")
        print(f"  Motion dim: {config.model.motion_dim}")
    else:
        print(f"  Latent dim: {config.model.latent_dim}")
    print(f"  LSTM hidden: {config.model.lstm_hidden}")
    print(f"\nTraining:")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Beta (KL weight): {config.training.beta_vae}")
    print(f"  KL annealing: {config.training.use_kl_annealing}")
    print(f"  Device: {config.training.device}")
    print("=" * 60)
    
    # Create dataloaders
    print("\nLoading data")
    train_loader, val_loader, test_loader = get_moving_mnist_dataloaders(
        root=config.data.root,
        batch_size=config.data.batch_size,
        train_size=config.data.train_size,
        val_size=config.data.val_size,
        test_size=config.data.test_size,
        sequence_length=config.data.sequence_length,
        frame_size=config.data.frame_size,
        num_digits=config.data.num_digits,
        num_workers=config.data.num_workers,
        seed=config.data.seed
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create trainer
    print("\nCreating model and trainer")
    trainer = create_trainer(config, train_loader, val_loader)
    
    # Train
    print("\nStarting training")
    history = trainer.train(resume=args.resume)
    
    print("\nTraining complete")
    print(f"Results saved to: {trainer.results_dir}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    
    return history


if __name__ == '__main__':
    main()

