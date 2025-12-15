
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm
import json
import time
from datetime import datetime

from training.config import ExperimentConfig
from models import (
    BaselineVAE, LSTMVAE, DisentangledVAE,
    baseline_loss, lstm_loss, disentangled_loss
)


class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if device is None:
            if config.training.device == 'mps' and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif config.training.device == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        print(f"Training on device: {self.device}")
        self.model = self.model.to(self.device)
        
        self.model_type = getattr(model, 'model_type', 'unknown')
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay
        )
        
        self.scheduler = self._setup_scheduler()
        self.loss_fn = self._get_loss_fn()
        
        self.checkpoint_dir = os.path.join(config.training.checkpoint_dir, config.name)
        self.results_dir = os.path.join(config.training.results_dir, config.name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'videos'), exist_ok=True)
        
        config.save(os.path.join(self.checkpoint_dir, 'config.json'))
        
        self.writer = SummaryWriter(os.path.join(self.results_dir, 'logs'))
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon': [],
            'val_recon': [],
            'train_kl': [],
            'val_kl': [],
            'learning_rate': []
        }
        
    def _setup_scheduler(self):
        if self.config.training.lr_scheduler == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.training.lr_decay
            )
        elif self.config.training.lr_scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.lr_min
            )
        else:
            return None
            
    def _get_loss_fn(self):
        if self.model_type == 'baseline_vae':
            return baseline_loss
        elif self.model_type == 'lstm_vae':
            return lstm_loss
        elif self.model_type == 'disentangled_vae':
            return disentangled_loss
        else:
            return baseline_loss
    
    def _get_beta(self, epoch: int) -> float:
        target_beta = self.config.training.beta_vae
        kl_start = self.config.training.kl_start_epoch
        
        if epoch < kl_start:
            return 0.0
        
        if not self.config.training.use_kl_annealing:
            return target_beta
        
        warmup = self.config.training.warmup_epochs
        epoch_since_start = epoch - kl_start
        
        if warmup <= 0:
            return target_beta

        if self.config.training.annealing_type == 'linear':
            if epoch_since_start < warmup:
                return target_beta * (epoch_since_start / warmup)
            return target_beta

        elif self.config.training.annealing_type == 'cosine':
            if epoch_since_start >= warmup:
                return target_beta
            progress = epoch_since_start / warmup
            value = target_beta * (1 - math.cos(progress * math.pi)) / 2
            return value

        elif self.config.training.annealing_type == 'cyclical':
            cycle_length = warmup
            cycle_pos = epoch_since_start % cycle_length
            cosine = 0.5 * (1 - math.cos(cycle_pos / cycle_length * math.pi))
            return target_beta * cosine

        return target_beta
    
    def _compute_loss(
        self,
        video: torch.Tensor,
        beta: float
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if self.model_type == 'disentangled_vae':
            output = self.model(video)
            losses = self.loss_fn(
                output,
                video,
                beta=beta,
                beta_content=self.config.training.beta_content,
                beta_motion=self.config.training.beta_motion,
                recon_loss_type=self.config.training.recon_loss_type,
                temporal_weight=self.config.training.temporal_weight,
                independence_weight=self.config.training.independence_weight
            )
            recon = output['recon']
        else:
            recon, z, mu, logvar = self.model(video)
            losses = self.loss_fn(
                recon, video, mu, logvar,
                beta=beta,
                recon_loss_type=self.config.training.recon_loss_type,
                free_bits=self.config.training.free_bits,
                use_free_bits=self.config.training.use_free_bits
            )
        
        return losses, recon
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {}
        
        beta = self._get_beta(self.current_epoch)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}', 
                    dynamic_ncols=True, leave=True, mininterval=0.5)
        for batch_idx, (video, metadata) in enumerate(pbar):
            video = video.to(self.device)
            
            self.optimizer.zero_grad()
            losses, _ = self._compute_loss(video, beta)
            
            loss = losses['total_loss']
            loss.backward()
            
            if self.config.training.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            self.optimizer.step()
            
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()
            
            postfix = {
                'loss': f'{loss.item():.4f}',
                'recon': f'{losses["recon_loss"].item():.4f}',
                'kl': f'{losses["kl_loss"].item():.4f}',
                'β': f'{beta:.3f}'
            }
            if 'recon_loss_per_pixel' in losses:
                postfix['pix'] = f'{losses["recon_loss_per_pixel"].item():.6f}'
            if 'kl_loss_raw' in losses:
                postfix['kl_raw'] = f'{losses["kl_loss_raw"].item():.4f}'
            pbar.set_postfix(postfix, refresh=False)
            
            if self.global_step % self.config.training.log_every == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v.item(), self.global_step)
                self.writer.add_scalar('train/beta', beta, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], 
                                       self.global_step)
            
            self.global_step += 1
        
        num_batches = len(self.train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        epoch_losses = {}
        
        beta = self.config.training.beta_vae
        
        for video, metadata in self.val_loader:
            video = video.to(self.device)
            
            losses, _ = self._compute_loss(video, beta)
            
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()
        
        num_batches = len(self.val_loader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            
        return epoch_losses
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt'):
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config.to_dict()
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, filename: str = 'checkpoint.pt'):
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
        return True
    
    @torch.no_grad()
    def visualize_reconstructions(self, num_samples: int = 4):
        self.model.eval()
        
        video, metadata = next(iter(self.val_loader))
        video = video[:num_samples].to(self.device)
        
        if self.model_type == 'disentangled_vae':
            output = self.model(video)
            recon = output['recon']
        else:
            recon, _, _, _ = self.model(video)
        
        for i in range(min(num_samples, video.size(0))):
            self.writer.add_image(
                f'recon/original_{i}_first',
                video[i, 0],
                self.current_epoch
            )
            self.writer.add_image(
                f'recon/original_{i}_last',
                video[i, -1],
                self.current_epoch
            )
            self.writer.add_image(
                f'recon/reconstructed_{i}_first',
                recon[i, 0],
                self.current_epoch
            )
            self.writer.add_image(
                f'recon/reconstructed_{i}_last',
                recon[i, -1],
                self.current_epoch
            )
    
    @torch.no_grad()
    def visualize_samples(self, num_samples: int = 4):
        self.model.eval()
        
        samples = self.model.sample(num_samples=num_samples, device=self.device)
        
        for i in range(samples.size(0)):
            self.writer.add_image(
                f'samples/sample_{i}_first',
                samples[i, 0],
                self.current_epoch
            )
            self.writer.add_image(
                f'samples/sample_{i}_last',
                samples[i, -1],
                self.current_epoch
            )
    
    def train(self, resume: bool = False):
        if resume:
            self.load_checkpoint()
        
        start_epoch = self.current_epoch
        num_epochs = self.config.training.num_epochs
        
        print(f"\nStarting training from epoch {start_epoch}")
        print(f"Model type: {self.model_type}")
        print(f"Device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
        
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            train_losses = self.train_epoch()
            val_losses = self.validate()
            
            if self.scheduler is not None:
                if self.optimizer.param_groups[0]['lr'] > self.config.training.lr_min:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_losses['total_loss'])
            self.history['val_loss'].append(val_losses['total_loss'])
            self.history['train_recon'].append(train_losses['recon_loss'])
            self.history['val_recon'].append(val_losses['recon_loss'])
            self.history['train_kl'].append(train_losses['kl_loss'])
            self.history['val_kl'].append(val_losses['kl_loss'])
            self.history['learning_rate'].append(current_lr)
            
            self.writer.add_scalar('epoch/train_loss', train_losses['total_loss'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_losses['total_loss'], epoch)
            self.writer.add_scalar('epoch/learning_rate', current_lr, epoch)
            
            print(f"\nEpoch {epoch}:")
            train_pix = train_losses.get('recon_loss_per_pixel', train_losses['recon_loss'])
            val_pix = val_losses.get('recon_loss_per_pixel', val_losses['recon_loss'])
            print(f"  Train - Total: {train_losses['total_loss']:.2f}, "
                  f"Recon: {train_losses['recon_loss']:.2f} (pix: {train_pix:.6f}), "
                  f"KL: {train_losses['kl_loss']:.2f}")
            print(f"  Val   - Total: {val_losses['total_loss']:.2f}, "
                  f"Recon: {val_losses['recon_loss']:.2f} (pix: {val_pix:.6f}), "
                  f"KL: {val_losses['kl_loss']:.2f}")
            print(f"  LR: {current_lr:.6f}, β: {self._get_beta(epoch):.3f}")
            
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint('best_model.pt')
                print(f"  New best model! Val loss: {self.best_val_loss:.4f}")
            
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            if (epoch + 1) % self.config.training.visualize_every == 0:
                self.visualize_reconstructions()
                self.visualize_samples()
        
        self.save_checkpoint('final_model.pt')
        
        history_path = os.path.join(self.results_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        self.writer.close()
        
        return self.history


def create_trainer(
    config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> Trainer:
    from models import get_model
    
    model_kwargs = {
        'in_channels': config.model.in_channels,
        'sequence_length': config.model.sequence_length
    }
    
    if config.model.model_type == 'baseline':
        model_kwargs.update({
            'latent_dim': config.model.latent_dim,
            'encoder_channels': config.model.encoder_channels,
            'decoder_channels': config.model.decoder_channels,
            'feature_dim': config.model.feature_dim
        })
    elif config.model.model_type == 'lstm':
        model_kwargs.update({
            'latent_dim': config.model.latent_dim,
            'encoder_channels': config.model.encoder_channels,
            'decoder_channels': config.model.decoder_channels,
            'feature_dim': config.model.feature_dim,
            'lstm_hidden': config.model.lstm_hidden,
            'lstm_layers': config.model.lstm_layers,
            'bidirectional': config.model.bidirectional
        })
    elif config.model.model_type == 'disentangled':
        model_kwargs.update({
            'content_dim': config.model.content_dim,
            'motion_dim': config.model.motion_dim,
            'content_channels': config.model.encoder_channels,
            'motion_channels': config.model.motion_channels,
            'decoder_channels': config.model.decoder_channels,
            'content_feature_dim': config.model.feature_dim,
            'motion_feature_dim': config.model.feature_dim,
            'lstm_hidden': config.model.lstm_hidden,
            'lstm_layers': config.model.lstm_layers,
            'bidirectional': config.model.bidirectional,
            'content_aggregation': config.model.content_aggregation
        })
    
    model = get_model(config.model.model_type, **model_kwargs)
    
    return Trainer(model, config, train_loader, val_loader)


if __name__ == '__main__':
    print("Testing Trainer")
    
    from .config import get_disentangled_config
    from ..data import get_moving_mnist_dataloaders
    
    config = get_disentangled_config('test_trainer')
    config.training.num_epochs = 2
    config.data.train_size = 100
    config.data.val_size = 20
    
    train_loader, val_loader, _ = get_moving_mnist_dataloaders(
        batch_size=config.data.batch_size,
        train_size=config.data.train_size,
        val_size=config.data.val_size,
        test_size=10,
        num_workers=0
    )
    
    trainer = create_trainer(config, train_loader, val_loader)
    
    print(f"Model type: {trainer.model_type}")
    print(f"Device: {trainer.device}")
    print(f"Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    print("\nTrainer test passed!")

