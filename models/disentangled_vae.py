"""
Architecture:
    video [B, 20, 1, 64, 64]
        ↓
    ┌──────────────────────────┬──────────────────────────┐
    │   Content Encoder Path   │    Motion Encoder Path   │
    │   (WHAT is moving?)      │    (HOW is it moving?)   │
    └──────────────────────────┴──────────────────────────┘
            ↓                            ↓
        z_content [B, C_dim]        z_motion [B, M_dim]
            ↓                            ↓
            └────────────┬───────────────┘
                         ↓
                    Combined Decoder
                         ↓
            reconstructed_video [B, 20, 1, 64, 64]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List

from .encoders import ContentEncoder, MotionEncoder, reparameterize
from .decoders import DisentangledDecoder


class DisentangledVAE(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 1,
        content_dim: int = 128,
        motion_dim: int = 128,
        content_channels: List[int] = [32, 64, 128, 256],
        motion_channels: List[int] = [16, 32],
        decoder_channels: List[int] = [256, 128, 64, 32],
        content_feature_dim: int = 512,
        motion_feature_dim: int = 512,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        sequence_length: int = 20,
        content_aggregation: str = 'mean'
    ):
        super().__init__()
        
        self.content_dim = content_dim
        self.motion_dim = motion_dim
        self.latent_dim = content_dim + motion_dim
        self.sequence_length = sequence_length
        self.model_type = 'disentangled_vae'
        
        self.content_encoder = ContentEncoder(
            in_channels=in_channels,
            channels=content_channels,
            feature_dim=content_feature_dim,
            latent_dim=content_dim,
            aggregation=content_aggregation
        )
        
        self.motion_encoder = MotionEncoder(
            in_channels=in_channels,
            channels=motion_channels,
            feature_dim=motion_feature_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            latent_dim=motion_dim
        )
        
        self.decoder = DisentangledDecoder(
            content_dim=content_dim,
            motion_dim=motion_dim,
            lstm_hidden=lstm_hidden,
            frame_feature_dim=content_feature_dim,
            cnn_channels=decoder_channels,
            sequence_length=sequence_length,
            num_lstm_layers=lstm_layers
        )
        
    def encode(
        self, 
        video: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, 
               Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor]]:
        z_content, (mu_c, logvar_c) = self.content_encoder(video)
        z_motion, (mu_m, logvar_m) = self.motion_encoder(video)
        
        return z_content, z_motion, (mu_c, logvar_c), (mu_m, logvar_m)
    
    def decode(
        self,
        z_content: torch.Tensor,
        z_motion: torch.Tensor,
        sequence_length: Optional[int] = None
    ) -> torch.Tensor:
        return self.decoder(z_content, z_motion, sequence_length)
    
    def forward(
        self, 
        video: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        z_content, z_motion, (mu_c, logvar_c), (mu_m, logvar_m) = self.encode(video)
        recon = self.decode(z_content, z_motion, sequence_length=video.shape[1])
        
        return {
            'recon': recon,
            'z_content': z_content,
            'z_motion': z_motion,
            'mu_content': mu_c,
            'logvar_content': logvar_c,
            'mu_motion': mu_m,
            'logvar_motion': logvar_m
        }
    
    def sample(
        self, 
        num_samples: int = 1,
        device: torch.device = None,
        sequence_length: Optional[int] = None,
        z_content: Optional[torch.Tensor] = None,
        z_motion: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
            
        if z_content is None:
            z_content = torch.randn(num_samples, self.content_dim, device=device)
        if z_motion is None:
            z_motion = torch.randn(num_samples, self.motion_dim, device=device)
            
        with torch.no_grad():
            videos = self.decode(z_content, z_motion, sequence_length)
            
        return videos
    
    def swap_content(
        self,
        video_content: torch.Tensor,
        video_motion: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            z_content, _, _, _ = self.encode(video_content)
            _, z_motion, _, _ = self.encode(video_motion)
            new_video = self.decode(z_content, z_motion, video_content.shape[1])
            
        return new_video
    
    def swap_motion(
        self,
        video_content: torch.Tensor,
        video_motion: torch.Tensor
    ) -> torch.Tensor:
        return self.swap_content(video_content, video_motion)
    
    def interpolate_content(
        self,
        video1: torch.Tensor,
        video2: torch.Tensor,
        num_steps: int = 10,
        use_motion_from: str = 'first'
    ) -> torch.Tensor:
        with torch.no_grad():
            z_c1, z_m1, _, _ = self.encode(video1)
            z_c2, z_m2, _, _ = self.encode(video2)
            
            z_motion = z_m1 if use_motion_from == 'first' else z_m2
            z_motion = z_motion.expand(num_steps, -1)
            
            alphas = torch.linspace(0, 1, num_steps, device=z_c1.device)
            z_content = []
            for alpha in alphas:
                z_c = (1 - alpha) * z_c1 + alpha * z_c2
                z_content.append(z_c)
            z_content = torch.cat(z_content, dim=0)
            
            videos = self.decode(z_content, z_motion, video1.shape[1])
            
        return videos
    
    def interpolate_motion(
        self,
        video1: torch.Tensor,
        video2: torch.Tensor,
        num_steps: int = 10,
        use_content_from: str = 'first'
    ) -> torch.Tensor:
        with torch.no_grad():
            z_c1, z_m1, _, _ = self.encode(video1)
            z_c2, z_m2, _, _ = self.encode(video2)
            
            z_content = z_c1 if use_content_from == 'first' else z_c2
            z_content = z_content.expand(num_steps, -1)
            
            alphas = torch.linspace(0, 1, num_steps, device=z_m1.device)
            z_motion = []
            for alpha in alphas:
                z_m = (1 - alpha) * z_m1 + alpha * z_m2
                z_motion.append(z_m)
            z_motion = torch.cat(z_motion, dim=0)
            
            videos = self.decode(z_content, z_motion, video1.shape[1])
            
        return videos
    
    def get_content_latent(self, video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, (mu_c, _) = self.content_encoder(video, return_distribution=True)
        return mu_c
    
    def get_motion_latent(self, video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, (mu_m, _) = self.motion_encoder(video, return_distribution=True)
        return mu_m
    
    def get_latents(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_content_latent(video), self.get_motion_latent(video)


def compute_loss(
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    beta: float = 1.0,
    beta_content: Optional[float] = None,
    beta_motion: Optional[float] = None,
    recon_loss_type: str = 'mse',
    temporal_weight: float = 0.0,
    independence_weight: float = 0.0
) -> Dict[str, torch.Tensor]:
    recon = output['recon']
    mu_c = output['mu_content']
    logvar_c = output['logvar_content']
    mu_m = output['mu_motion']
    logvar_m = output['logvar_motion']
    z_content = output['z_content']
    z_motion = output['z_motion']
    
    batch_size = recon.size(0)
    num_pixels = recon[0].numel()
    
    beta_c = beta_content if beta_content is not None else beta
    beta_m = beta_motion if beta_motion is not None else beta
    
    if recon_loss_type == 'mse':
        recon_loss = F.mse_loss(recon, target, reduction='mean')
    elif recon_loss_type == 'bce':
        recon_loss = F.binary_cross_entropy(recon, target, reduction='mean')
    else:
        raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")
    
    recon_loss_scaled = recon_loss * num_pixels
    
    kl_content_per_sample = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp(), dim=1)
    kl_content = kl_content_per_sample.mean()
    
    kl_motion_per_sample = -0.5 * torch.sum(1 + logvar_m - mu_m.pow(2) - logvar_m.exp(), dim=1)
    kl_motion = kl_motion_per_sample.mean()
    
    kl_loss = beta_c * kl_content + beta_m * kl_motion
    
    temporal_loss = torch.tensor(0.0, device=recon.device)
    if temporal_weight > 0:
        recon_diff = recon[:, 1:] - recon[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        temp_num_pixels = recon_diff[0].numel()
        temporal_loss = F.mse_loss(recon_diff, target_diff, reduction='mean') * temp_num_pixels
    
    independence_loss = torch.tensor(0.0, device=recon.device)
    if independence_weight > 0:
        z_c_norm = (z_content - z_content.mean(dim=0)) / (z_content.std(dim=0) + 1e-8)
        z_m_norm = (z_motion - z_motion.mean(dim=0)) / (z_motion.std(dim=0) + 1e-8)
        
        batch_size = z_content.size(0)
        correlation = torch.mm(z_c_norm.t(), z_m_norm) / batch_size
        
        independence_loss = correlation.pow(2).mean()
    
    total_loss = recon_loss_scaled + kl_loss + temporal_weight * temporal_loss + independence_weight * independence_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss_scaled,
        'recon_loss_per_pixel': recon_loss,
        'kl_loss': kl_loss,
        'kl_content': kl_content,
        'kl_motion': kl_motion,
        'temporal_loss': temporal_loss,
        'independence_loss': independence_loss,
        'beta': torch.tensor(beta)
    }


class DisentangledVAEWrapper(nn.Module):
    
    def __init__(self, model: DisentangledVAE):
        super().__init__()
        self.model = model
        self.latent_dim = model.latent_dim
        self.content_dim = model.content_dim
        self.motion_dim = model.motion_dim
        self.model_type = 'disentangled_vae'
        
    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(video)
        
        z = torch.cat([output['z_content'], output['z_motion']], dim=1)
        mu = torch.cat([output['mu_content'], output['mu_motion']], dim=1)
        logvar = torch.cat([output['logvar_content'], output['logvar_motion']], dim=1)
        
        return output['recon'], z, mu, logvar
    
    def encode(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_c, z_m, (mu_c, logvar_c), (mu_m, logvar_m) = self.model.encode(video)
        z = torch.cat([z_c, z_m], dim=1)
        mu = torch.cat([mu_c, mu_m], dim=1)
        logvar = torch.cat([logvar_c, logvar_m], dim=1)
        return z, mu, logvar
    
    def decode(self, z: torch.Tensor, sequence_length: Optional[int] = None) -> torch.Tensor:
        z_content = z[:, :self.content_dim]
        z_motion = z[:, self.content_dim:]
        return self.model.decode(z_content, z_motion, sequence_length)
    
    def sample(self, num_samples: int = 1, device: torch.device = None, 
               sequence_length: Optional[int] = None) -> torch.Tensor:
        return self.model.sample(num_samples, device, sequence_length)

if __name__ == '__main__':
    print("Testing Disentangled VAE")
    
    model = DisentangledVAE(content_dim=128, motion_dim=128)
    video = torch.randn(4, 20, 1, 64, 64)
    output = model(video)
    
    print(f"Input shape: {video.shape}")
    print(f"Reconstructed shape: {output['recon'].shape}")
    print(f"Content latent shape: {output['z_content'].shape}")
    print(f"Motion latent shape: {output['z_motion'].shape}")
    
    losses = compute_loss(output, video, beta=4.0, independence_weight=0.1)
    print(f"Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    samples = model.sample(num_samples=2)
    print(f"Sampled shape: {samples.shape}")
    
    interp = model.interpolate_content(video[:1], video[1:2], num_steps=5)
    print(f"Content interpolation shape: {interp.shape}")
    
    interp = model.interpolate_motion(video[:1], video[1:2], num_steps=5)
    print(f"Motion interpolation shape: {interp.shape}")
    
    swapped = model.swap_content(video[:2], video[2:4])
    print(f"Swapped shape: {swapped.shape}")
    
    wrapped = DisentangledVAEWrapper(model)
    recon, z, mu, logvar = wrapped(video)
    print(f"Wrapper recon shape: {recon.shape}")
    print(f"Wrapper z shape: {z.shape}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    print("Disentangled VAE test passed!")

