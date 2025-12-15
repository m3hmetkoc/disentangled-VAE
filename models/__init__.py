"""
Models module for Video VAE.

Contains three model variants:
1. BaselineVAE: Simple VAE without temporal modeling
2. LSTMVAE: VAE with LSTM for temporal modeling (unified latent space)
3. DisentangledVAE: VAE with explicit content/motion separation (YOUR CONTRIBUTION)
"""

from .encoders import (
    CNNEncoder,
    LightCNNEncoder,
    VideoEncoder,
    LSTMEncoder,
    VAEEncoder,
    ContentEncoder,
    MotionEncoder,
    reparameterize
)

from .decoders import (
    CNNDecoder,
    LSTMDecoder,
    VideoDecoder,
    DisentangledDecoder
)

from .baseline_vae import BaselineVAE
from .baseline_vae import compute_loss as baseline_loss

from .lstm_vae import LSTMVAE
from .lstm_vae import compute_loss as lstm_loss

from .disentangled_vae import DisentangledVAE, DisentangledVAEWrapper
from .disentangled_vae import compute_loss as disentangled_loss


def get_model(model_type: str, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: One of 'baseline', 'lstm', 'disentangled'
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
    """
    if model_type == 'baseline':
        return BaselineVAE(**kwargs)
    elif model_type == 'lstm':
        return LSTMVAE(**kwargs)
    elif model_type == 'disentangled':
        return DisentangledVAE(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_loss_fn(model_type: str):
    """
    Get the appropriate loss function for a model type.
    
    Args:
        model_type: One of 'baseline', 'lstm', 'disentangled'
        
    Returns:
        Loss function
    """
    if model_type == 'baseline':
        return baseline_loss
    elif model_type == 'lstm':
        return lstm_loss
    elif model_type == 'disentangled':
        return disentangled_loss
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    # Encoders
    'CNNEncoder',
    'LightCNNEncoder',
    'VideoEncoder',
    'LSTMEncoder',
    'VAEEncoder',
    'ContentEncoder',
    'MotionEncoder',
    'reparameterize',
    
    # Decoders
    'CNNDecoder',
    'LSTMDecoder',
    'VideoDecoder',
    'DisentangledDecoder',
    
    # Models
    'BaselineVAE',
    'LSTMVAE',
    'DisentangledVAE',
    'DisentangledVAEWrapper',
    
    # Loss functions
    'baseline_loss',
    'lstm_loss',
    'disentangled_loss',
    
    # Factory functions
    'get_model',
    'get_loss_fn'
]

