"""
Shared encoder components for Video VAE models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class CNNEncoder(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = [32, 64, 128, 256],
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_dim: int = 512,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        
        layers = []
        in_ch = in_channels
        for out_ch in channels:
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=not use_batch_norm)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch
        
        self.conv = nn.Sequential(*layers)
        
        self.flat_size = channels[-1] * 4 * 4
        
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h = self.conv(x)
        h = h.view(batch_size, -1)
        features = self.fc(h)
        return features


class LightCNNEncoder(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = [16, 32],
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_dim: int = 512,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        layers = []
        in_ch = in_channels
        for out_ch in channels:
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=not use_batch_norm)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch
        
        self.conv = nn.Sequential(*layers)
        
        self.flat_size = channels[-1] * 16 * 16
        
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h = self.conv(x)
        h = h.view(batch_size, -1)
        features = self.fc(h)
        return features


class VideoEncoder(nn.Module):
    
    def __init__(
        self,
        cnn_encoder: nn.Module,
        pool_type: str = 'none'
    ):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.pool_type = pool_type
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        
        x_flat = x.view(batch_size * seq_len, *x.shape[2:])
        
        features = self.cnn_encoder(x_flat)
        
        features = features.view(batch_size, seq_len, -1)
        
        if self.pool_type == 'mean':
            features = features.mean(dim=1)
        elif self.pool_type == 'max':
            features = features.max(dim=1)[0]
            
        return features


class LSTMEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.output_dim = hidden_dim * self.num_directions
        
    def forward(
        self, 
        x: torch.Tensor,
        return_sequence: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output, (h_n, c_n) = self.lstm(x)
        
        if return_sequence:
            return output, (h_n, c_n)
        
        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            final_hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            final_hidden = h_n[-1]
            
        return final_hidden, (h_n, c_n)


class VAEEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 128,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        if hidden_dim is not None:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        else:
            self.fc = nn.Identity()
            self.fc_mu = nn.Linear(input_dim, latent_dim)
            self.fc_logvar = nn.Linear(input_dim, latent_dim)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if not torch.is_grad_enabled():
        return mu
    
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z


class ContentEncoder(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = [32, 64, 128, 256],
        feature_dim: int = 512,
        latent_dim: int = 128,
        aggregation: str = 'mean'
    ):
        super().__init__()
        
        self.aggregation = aggregation
        self.latent_dim = latent_dim
        
        self.cnn_encoder = CNNEncoder(
            in_channels=in_channels,
            channels=channels,
            output_dim=feature_dim
        )
        
        self.vae_head = VAEEncoder(
            input_dim=feature_dim,
            latent_dim=latent_dim
        )
        
    def forward(
        self, 
        video: torch.Tensor,
        return_distribution: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len = video.shape[:2]
        
        if self.aggregation == 'mean':
            frame = video.mean(dim=1)
        elif self.aggregation == 'middle':
            frame = video[:, seq_len // 2]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        features = self.cnn_encoder(frame)
        
        mu, logvar = self.vae_head(features)
        
        z_content = reparameterize(mu, logvar)
        
        if return_distribution:
            return z_content, (mu, logvar)
        return z_content, None


class MotionEncoder(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = [16, 32],
        feature_dim: int = 512,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        latent_dim: int = 128
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.cnn_encoder = LightCNNEncoder(
            in_channels=in_channels,
            channels=channels,
            output_dim=feature_dim
        )
        
        self.lstm = LSTMEncoder(
            input_dim=feature_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional
        )
        
        self.vae_head = VAEEncoder(
            input_dim=self.lstm.output_dim,
            latent_dim=latent_dim
        )
        
    def compute_frame_differences(self, video: torch.Tensor) -> torch.Tensor:
        return video[:, 1:] - video[:, :-1]
    
    def forward(
        self,
        video: torch.Tensor,
        return_distribution: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len = video.shape[:2]
        
        frame_diffs = self.compute_frame_differences(video)
        diff_len = frame_diffs.shape[1]
        
        frame_diffs_flat = frame_diffs.view(batch_size * diff_len, *frame_diffs.shape[2:])
        diff_features = self.cnn_encoder(frame_diffs_flat)
        diff_features = diff_features.view(batch_size, diff_len, -1)
        
        lstm_out, _ = self.lstm(diff_features)
        
        mu, logvar = self.vae_head(lstm_out)
        
        z_motion = reparameterize(mu, logvar)
        
        if return_distribution:
            return z_motion, (mu, logvar)
        return z_motion, None


if __name__ == '__main__':
    print("Testing Encoders...")
    
    cnn = CNNEncoder(in_channels=1, output_dim=512)
    x = torch.randn(4, 1, 64, 64)
    out = cnn(x)
    print(f"CNN Encoder: {x.shape} -> {out.shape}")
    
    light_cnn = LightCNNEncoder(in_channels=1, output_dim=512)
    out = light_cnn(x)
    print(f"Light CNN Encoder: {x.shape} -> {out.shape}")
    
    video_enc = VideoEncoder(cnn, pool_type='none')
    video = torch.randn(4, 20, 1, 64, 64)
    out = video_enc(video)
    print(f"Video Encoder: {video.shape} -> {out.shape}")
    
    lstm_enc = LSTMEncoder(input_dim=512, hidden_dim=256, bidirectional=True)
    seq = torch.randn(4, 20, 512)
    out, _ = lstm_enc(seq)
    print(f"LSTM Encoder: {seq.shape} -> {out.shape}")
    
    content_enc = ContentEncoder(latent_dim=128)
    z_content, (mu, logvar) = content_enc(video)
    print(f"Content Encoder: {video.shape} -> z={z_content.shape}")
    
    motion_enc = MotionEncoder(latent_dim=128)
    z_motion, (mu, logvar) = motion_enc(video)
    print(f"Motion Encoder: {video.shape} -> z={z_motion.shape}")
    
    print("All encoder tests passed!")

