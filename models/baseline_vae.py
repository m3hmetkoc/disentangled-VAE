"""
Baseline VAE for video processing.
Processes each frame independently without temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List

from .encoders import CNNEncoder, VAEEncoder, reparameterize
from .decoders import CNNDecoder


class BaselineVAE(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        encoder_channels: List[int] = [32, 64, 128, 256],
        decoder_channels: List[int] = [256, 128, 64, 32],
        feature_dim: int = 512,
        sequence_length: int = 20
    ):
        """
        Initialize Baseline VAE.
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            latent_dim: Dimension of latent space
            encoder_channels: CNN encoder channel progression
            decoder_channels: CNN decoder channel progression
            feature_dim: Intermediate feature dimension
            sequence_length: Number of frames per video
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.model_type = 'baseline_vae'
        
        self.cnn_encoder = CNNEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            output_dim=feature_dim
        )
        
        self.vae_head = VAEEncoder(
            input_dim=feature_dim,
            latent_dim=latent_dim
        )
        
        self.cnn_decoder = CNNDecoder(
            input_dim=latent_dim,
            channels=decoder_channels,
            out_channels=in_channels
        )
        
    def encode(
        self, 
        video: torch.Tensor,
        return_all_frames: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode video to latent space.
        
        Args:
            video: Input video [B, T, C, H, W]
            return_all_frames: If True, return per-frame latents
            
        Returns:
            z: Sampled latent [B, latent_dim]
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]
        """
        batch_size, seq_len = video.shape[:2]
        
        video_flat = video.view(batch_size * seq_len, *video.shape[2:])
        frame_features = self.cnn_encoder(video_flat)
        
        frame_mu, frame_logvar = self.vae_head(frame_features)
        
        frame_mu = frame_mu.view(batch_size, seq_len, -1)
        frame_logvar = frame_logvar.view(batch_size, seq_len, -1)
        
        mu = frame_mu.mean(dim=1)
        logvar = frame_logvar.mean(dim=1)
        
        z = reparameterize(mu, logvar)
        
        if return_all_frames:
            return z, mu, logvar, frame_mu, frame_logvar
        return z, mu, logvar
    
    def decode(
        self, 
        z: torch.Tensor,
        sequence_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decode latent to video.
        
        Args:
            z: Latent code [B, latent_dim]
            sequence_length: Number of frames to generate
            
        Returns:
            Reconstructed video [B, T, C, H, W]
        """
        batch_size = z.size(0)
        seq_len = sequence_length or self.sequence_length
        
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        z_flat = z_expanded.reshape(batch_size * seq_len, -1)
        
        frames = self.cnn_decoder(z_flat)
        
        video = frames.view(batch_size, seq_len, *frames.shape[1:])
        
        return video
    
    def forward(
        self, 
        video: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode and decode.
        
        Args:
            video: Input video [B, T, C, H, W]
            
        Returns:
            recon: Reconstructed video [B, T, C, H, W]
            z: Sampled latent [B, latent_dim]
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]
        """
        z, mu, logvar = self.encode(video)
        recon = self.decode(z, sequence_length=video.shape[1])
        
        return recon, z, mu, logvar
    
    def sample(
        self, 
        num_samples: int = 1,
        device: torch.device = None,
        sequence_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample random videos from prior.

        Returns:
            Generated videos [num_samples, T, C, H, W]
        """
        if device is None:
            device = next(self.parameters()).device
            
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        with torch.no_grad():
            videos = self.decode(z, sequence_length)
            
        return videos
    
    def interpolate(
        self,
        video1: torch.Tensor,
        video2: torch.Tensor,
        num_steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two videos in latent space.
            
        Returns:
            Interpolated videos [num_steps, T, C, H, W]
        """
        with torch.no_grad():
            z1, _, _ = self.encode(video1)
            z2, _, _ = self.encode(video2)
            
            alphas = torch.linspace(0, 1, num_steps, device=z1.device)
            z_interp = []
            for alpha in alphas:
                z = (1 - alpha) * z1 + alpha * z2
                z_interp.append(z)
            z_interp = torch.cat(z_interp, dim=0)
            
            videos = self.decode(z_interp, sequence_length=video1.shape[1])
            
        return videos
    
    def get_latent(self, video: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation (mean) for a video.
        
        Args:
            video: Input video [B, T, C, H, W]
            
        Returns:
            Latent mean [B, latent_dim]
        """
        with torch.no_grad():
            _, mu, _ = self.encode(video)
        return mu


def compute_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    recon_loss_type: str = 'mse',
    free_bits: float = 0.0,
    use_free_bits: bool = False
) -> Dict[str, torch.Tensor]:
    batch_size = recon.size(0)
    num_pixels = recon[0].numel()
    latent_dim = mu.size(1)
    
    if recon_loss_type == 'mse':
        recon_loss = F.mse_loss(recon, target, reduction='mean')
    elif recon_loss_type == 'bce':
        recon_loss = F.binary_cross_entropy(recon, target, reduction='mean')
    else:
        raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")
    
    recon_loss_scaled = recon_loss * num_pixels
    
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    if use_free_bits and free_bits > 0:
        kl_per_dim = torch.maximum(kl_per_dim, torch.tensor(free_bits, device=kl_per_dim.device))
    
    kl_per_sample = kl_per_dim.sum(dim=1)
    kl_loss = kl_per_sample.mean()
    
    kl_per_dim_raw = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss_raw = kl_per_dim_raw.sum(dim=1).mean()
    
    total_loss = recon_loss_scaled + beta * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss_scaled,
        'recon_loss_per_pixel': recon_loss,
        'kl_loss': kl_loss,
        'kl_loss_raw': kl_loss_raw,
        'kl_per_dim': kl_per_dim.mean(),
        'beta': torch.tensor(beta)
    }

if __name__ == '__main__':
    print("Testing Baseline VAE.")
    
    model = BaselineVAE(latent_dim=256)
    video = torch.randn(4, 20, 1, 64, 64)
    recon, z, mu, logvar = model(video)
    
    print(f"Input shape: {video.shape}")
    print(f"Reconstructed shape: {recon.shape}")
    print(f"Latent z shape: {z.shape}")
    
    losses = compute_loss(recon, video, mu, logvar, beta=4.0)
    print(f"Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    samples = model.sample(num_samples=2)
    print(f"Sampled shape: {samples.shape}")
    
    interp = model.interpolate(video[:1], video[1:2], num_steps=5)
    print(f"Interpolation shape: {interp.shape}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    print("Baseline VAE test passed!")


