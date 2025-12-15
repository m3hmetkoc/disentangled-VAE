"""
Shared decoder components for Video VAE models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class CNNDecoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 512,
        channels: List[int] = [256, 128, 64, 32],
        out_channels: int = 1,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_batch_norm: bool = True,
        output_activation: str = 'sigmoid'
    ):
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        self.output_activation = output_activation
        
        self.initial_size = 4
        self.fc = nn.Sequential(
            nn.Linear(input_dim, channels[0] * self.initial_size * self.initial_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        layers = []
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, 
                                   bias=not use_batch_norm)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.deconv = nn.Sequential(*layers)
        
        self.final = nn.ConvTranspose2d(channels[-1], out_channels, kernel_size, 
                                        stride, padding, bias=True)
        
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        else:
            self.output_act = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        h = self.fc(x)
        h = h.view(batch_size, -1, self.initial_size, self.initial_size)
        
        h = self.deconv(h)
        
        out = self.final(h)
        out = self.output_act(out)
        
        return out


class LSTMDecoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 1,
        sequence_length: int = 20,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        
        self.fc_hidden = nn.Linear(input_dim, hidden_dim * num_layers)
        self.fc_cell = nn.Linear(input_dim, hidden_dim * num_layers)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.initial_input = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.xavier_uniform_(self.initial_input)
        
    def forward(
        self, 
        z: torch.Tensor,
        sequence_length: Optional[int] = None
    ) -> torch.Tensor:
        batch_size = z.size(0)
        seq_len = sequence_length or self.sequence_length
        
        h = self.fc_hidden(z)
        h = h.view(batch_size, self.num_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()
        
        c = self.fc_cell(z)
        c = c.view(batch_size, self.num_layers, self.hidden_dim)
        c = c.permute(1, 0, 2).contiguous()
        
        outputs = []
        input_t = self.initial_input.expand(batch_size, -1).unsqueeze(1)
        
        for t in range(seq_len):
            output, (h, c) = self.lstm(input_t, (h, c))
            outputs.append(output)
            input_t = output
        
        outputs = torch.cat(outputs, dim=1)
        outputs = self.fc_out(outputs)
        
        return outputs


class VideoDecoder(nn.Module):
    
    def __init__(
        self,
        latent_dim: int = 256,
        lstm_hidden: int = 256,
        frame_feature_dim: int = 512,
        cnn_channels: List[int] = [256, 128, 64, 32],
        sequence_length: int = 20,
        use_content_modulation: bool = False,
        content_dim: int = 128
    ):
        super().__init__()
        
        self.use_content_modulation = use_content_modulation
        self.content_dim = content_dim
        
        self.lstm_decoder = LSTMDecoder(
            input_dim=latent_dim,
            hidden_dim=lstm_hidden,
            output_dim=frame_feature_dim,
            sequence_length=sequence_length
        )
        
        cnn_input_dim = frame_feature_dim + content_dim if use_content_modulation else frame_feature_dim
        
        self.cnn_decoder = CNNDecoder(
            input_dim=cnn_input_dim,
            channels=cnn_channels
        )
        
        self.sequence_length = sequence_length
        
    def forward(
        self,
        z: torch.Tensor,
        z_content: Optional[torch.Tensor] = None,
        sequence_length: Optional[int] = None
    ) -> torch.Tensor:
        batch_size = z.size(0)
        seq_len = sequence_length or self.sequence_length
        
        frame_features = self.lstm_decoder(z, seq_len)
        
        if self.use_content_modulation and z_content is not None:
            z_content_expanded = z_content.unsqueeze(1).expand(-1, seq_len, -1)
            frame_features = torch.cat([frame_features, z_content_expanded], dim=-1)
        
        frame_features_flat = frame_features.view(batch_size * seq_len, -1)
        frames_flat = self.cnn_decoder(frame_features_flat)
        
        video = frames_flat.view(batch_size, seq_len, *frames_flat.shape[1:])
        
        return video


class DisentangledDecoder(nn.Module):
    
    def __init__(
        self,
        content_dim: int = 128,
        motion_dim: int = 128,
        lstm_hidden: int = 256,
        frame_feature_dim: int = 512,
        cnn_channels: List[int] = [256, 128, 64, 32],
        sequence_length: int = 20,
        num_lstm_layers: int = 1
    ):
        super().__init__()
        
        self.content_dim = content_dim
        self.motion_dim = motion_dim
        self.lstm_hidden = lstm_hidden
        self.num_lstm_layers = num_lstm_layers
        self.sequence_length = sequence_length
        
        combined_dim = content_dim + motion_dim
        
        self.fc_hidden = nn.Linear(combined_dim, lstm_hidden * num_lstm_layers)
        self.fc_cell = nn.Linear(combined_dim, lstm_hidden * num_lstm_layers)
        
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        self.fc_frame = nn.Sequential(
            nn.Linear(lstm_hidden, frame_feature_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.initial_input = nn.Parameter(torch.zeros(1, lstm_hidden))
        nn.init.xavier_uniform_(self.initial_input)
        
        cnn_input_dim = frame_feature_dim + content_dim
        self.cnn_decoder = CNNDecoder(
            input_dim=cnn_input_dim,
            channels=cnn_channels
        )
        
    def forward(
        self,
        z_content: torch.Tensor,
        z_motion: torch.Tensor,
        sequence_length: Optional[int] = None
    ) -> torch.Tensor:
        batch_size = z_content.size(0)
        seq_len = sequence_length or self.sequence_length
        
        z_combined = torch.cat([z_content, z_motion], dim=-1)
        
        h = self.fc_hidden(z_combined)
        h = h.view(batch_size, self.num_lstm_layers, self.lstm_hidden)
        h = h.permute(1, 0, 2).contiguous()
        
        c = self.fc_cell(z_combined)
        c = c.view(batch_size, self.num_lstm_layers, self.lstm_hidden)
        c = c.permute(1, 0, 2).contiguous()
        
        outputs = []
        input_t = self.initial_input.expand(batch_size, -1).unsqueeze(1)
        
        for t in range(seq_len):
            output, (h, c) = self.lstm(input_t, (h, c))
            outputs.append(output)
            input_t = output
        
        outputs = torch.cat(outputs, dim=1)
        
        frame_features = self.fc_frame(outputs)
        
        z_content_expanded = z_content.unsqueeze(1).expand(-1, seq_len, -1)
        frame_features = torch.cat([frame_features, z_content_expanded], dim=-1)
        
        frame_features_flat = frame_features.view(batch_size * seq_len, -1)
        frames_flat = self.cnn_decoder(frame_features_flat)
        
        video = frames_flat.view(batch_size, seq_len, *frames_flat.shape[1:])
        
        return video


if __name__ == '__main__':
    print("Testing Decoders")
    
    cnn_dec = CNNDecoder(input_dim=512)
    z = torch.randn(4, 512)
    out = cnn_dec(z)
    print(f"CNN Decoder: {z.shape} -> {out.shape}")
    
    lstm_dec = LSTMDecoder(input_dim=256, output_dim=512, sequence_length=20)
    z = torch.randn(4, 256)
    out = lstm_dec(z)
    print(f"LSTM Decoder: {z.shape} -> {out.shape}")
    
    video_dec = VideoDecoder(latent_dim=256, sequence_length=20)
    z = torch.randn(4, 256)
    out = video_dec(z)
    print(f"Video Decoder: {z.shape} -> {out.shape}")
    
    video_dec_mod = VideoDecoder(
        latent_dim=256, 
        sequence_length=20,
        use_content_modulation=True,
        content_dim=128
    )
    z = torch.randn(4, 256)
    z_content = torch.randn(4, 128)
    out = video_dec_mod(z, z_content)
    print(f"Video Decoder (modulated): z={z.shape}, z_c={z_content.shape} -> {out.shape}")
    
    dis_dec = DisentangledDecoder(content_dim=128, motion_dim=128, sequence_length=20)
    z_content = torch.randn(4, 128)
    z_motion = torch.randn(4, 128)
    out = dis_dec(z_content, z_motion)
    print(f"Disentangled Decoder: z_c={z_content.shape}, z_m={z_motion.shape} -> {out.shape}")
    
    print("All decoder tests passed!")

