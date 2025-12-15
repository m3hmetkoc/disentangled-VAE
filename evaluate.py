import argparse
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_moving_mnist_dataloaders
from models import BaselineVAE, LSTMVAE, DisentangledVAE
from evaluation.metrics import mse, psnr, ssim_video, temporal_consistency


def load_model(checkpoint_path: str, device: str = 'cuda'): 
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', None)
    if config is None:
        raise ValueError("Checkpoint doesn't contain config. Cannot determine model type.")
    
    if isinstance(config, dict):
        from types import SimpleNamespace
        
        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d
        
        config = dict_to_namespace(config)
    
    model_type = config.model.model_type
    print(f"Loading {model_type} model")
    
    if model_type == 'baseline':
        model = BaselineVAE(
            latent_dim=config.model.latent_dim,
            encoder_channels=config.model.encoder_channels,
            decoder_channels=config.model.decoder_channels,
            feature_dim=config.model.feature_dim,
            in_channels=config.model.in_channels,
            sequence_length=config.model.sequence_length
        )
    elif model_type == 'lstm':
        model = LSTMVAE(
            latent_dim=config.model.latent_dim,
            lstm_hidden=config.model.lstm_hidden,
            lstm_layers=config.model.lstm_layers,
            bidirectional=config.model.bidirectional,
            encoder_channels=config.model.encoder_channels,
            decoder_channels=config.model.decoder_channels,
            feature_dim=config.model.feature_dim,
            in_channels=config.model.in_channels,
            sequence_length=config.model.sequence_length
        )
    elif model_type == 'disentangled':
        model = DisentangledVAE(
            content_dim=config.model.content_dim,
            motion_dim=config.model.motion_dim,
            content_channels=config.model.encoder_channels,
            decoder_channels=config.model.decoder_channels,
            motion_channels=config.model.motion_channels,
            content_feature_dim=config.model.feature_dim,
            motion_feature_dim=config.model.feature_dim,
            lstm_hidden=config.model.lstm_hidden,
            lstm_layers=config.model.lstm_layers,
            bidirectional=config.model.bidirectional,
            in_channels=config.model.in_channels,
            sequence_length=config.model.sequence_length,
            content_aggregation=config.model.content_aggregation
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    best_val_loss = checkpoint.get('best_val_loss', None)
    if best_val_loss is not None:
        print(f"Best val loss: {best_val_loss:.4f}")
    else:
        print("Best val loss: unknown")
    
    return model, config


def visualize_reconstructions(model, dataloader, device, num_samples=5, save_path=None):
    model.eval()
    
    videos, metadata = next(iter(dataloader))
    videos = videos[:num_samples].to(device)
    
    with torch.no_grad():
        if hasattr(model, 'model_type') and model.model_type == 'disentangled_vae':
            output = model(videos)
            recon = output['recon']
        else:
            recon, z, mu, logvar = model(videos)
    
    videos_np = videos.cpu().numpy()
    recon_np = recon.cpu().numpy()
    
    num_frames_to_show = min(10, videos.shape[1])
    fig, axes = plt.subplots(num_samples * 2, num_frames_to_show, figsize=(num_frames_to_show * 1.5, num_samples * 3))
    
    for i in range(num_samples):
        for j in range(num_frames_to_show):
            frame_idx = j * (videos.shape[1] // num_frames_to_show)
            
            ax = axes[i * 2, j]
            ax.imshow(videos_np[i, frame_idx, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(f'Original {i+1}', fontsize=10)
            
            ax = axes[i * 2 + 1, j]
            ax.imshow(recon_np[i, frame_idx, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(f'Recon {i+1}', fontsize=10)
    
    plt.suptitle('Original (odd rows) vs Reconstruction (even rows)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reconstruction visualization to {save_path}")
    
    plt.show()


def visualize_samples(model, device, num_samples=5, sequence_length=20, save_path=None):
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(num_samples, device=device, sequence_length=sequence_length)
    
    samples_np = samples.cpu().numpy()
    
    num_frames_to_show = min(10, sequence_length)
    fig, axes = plt.subplots(num_samples, num_frames_to_show, figsize=(num_frames_to_show * 1.5, num_samples * 1.5))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        for j in range(num_frames_to_show):
            frame_idx = j * (sequence_length // num_frames_to_show)
            ax = axes[i, j]
            ax.imshow(samples_np[i, frame_idx, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
    
    plt.suptitle('Random Samples from Model', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved samples visualization to {save_path}")
    
    plt.show()


def compute_metrics(model, dataloader, device, num_batches=10):
    model.eval()
    
    all_mse = []
    all_psnr = []
    all_ssim = []
    all_temporal = []
    
    print(f"\nComputing metrics on {num_batches} batches...")
    
    for i, (videos, metadata) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        videos = videos.to(device)
        
        with torch.no_grad():
            if hasattr(model, 'model_type') and model.model_type == 'disentangled_vae':
                output = model(videos)
                recon = output['recon']
            else:
                recon, z, mu, logvar = model(videos)
        
        all_mse.append(mse(recon, videos))
        all_psnr.append(psnr(recon, videos))
        all_ssim.append(ssim_video(recon, videos))
        all_temporal.append(temporal_consistency(recon))
    
    results = {
        'mse': np.mean(all_mse),
        'mse_std': np.std(all_mse),
        'psnr': np.mean(all_psnr),
        'psnr_std': np.std(all_psnr),
        'ssim': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'temporal_consistency': np.mean(all_temporal),
        'temporal_consistency_std': np.std(all_temporal),
    }
    
    return results


def interpolate_latent(model, dataloader, device, save_path=None):
    model.eval()
    
    videos, _ = next(iter(dataloader))
    v1, v2 = videos[0:1].to(device), videos[1:2].to(device)
    
    with torch.no_grad():
        if hasattr(model, 'model_type') and model.model_type == 'disentangled_vae':
            z1_c, z1_m, _, _ = model.encode(v1)
            z2_c, z2_m, _, _ = model.encode(v2)
            
            alphas = torch.linspace(0, 1, 7).to(device)
            interp_frames = []
            
            for alpha in alphas:
                z_c = (1 - alpha) * z1_c + alpha * z2_c
                z_m = z1_m
                recon = model.decode(z_c, z_m, sequence_length=20)
                interp_frames.append(recon[0, 10, 0].cpu().numpy())
        else:
            z1, mu1, logvar1 = model.encode(v1)
            z2, mu2, logvar2 = model.encode(v2)
            
            alphas = torch.linspace(0, 1, 7).to(device)
            interp_frames = []
            
            for alpha in alphas:
                z = (1 - alpha) * z1 + alpha * z2
                recon = model.decode(z, sequence_length=20)
                interp_frames.append(recon[0, 10, 0].cpu().numpy())
    
    fig, axes = plt.subplots(1, 9, figsize=(18, 2))
    
    axes[0].imshow(v1[0, 10, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Video 1')
    axes[0].axis('off')
    
    for i, frame in enumerate(interp_frames):
        axes[i + 1].imshow(frame, cmap='gray', vmin=0, vmax=1)
        axes[i + 1].set_title(f'α={i/6:.1f}')
        axes[i + 1].axis('off')
    
    axes[8].imshow(v2[0, 10, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[8].set_title('Video 2')
    axes[8].axis('off')
    
    plt.suptitle('Latent Space Interpolation (middle frame)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved interpolation to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained VAE model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to use')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='./eval_results',
                        help='Directory to save results')
    parser.add_argument('--skip_metrics', action='store_true',
                        help='Skip computing metrics')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("LOADING MODEL")
    model, config = load_model(args.checkpoint, args.device)
    print("LOADING TEST DATA")
    _, _, test_loader = get_moving_mnist_dataloaders(
        root=config.data.root,
        train_size=100,  # Don't need train data
        val_size=100,
        test_size=config.data.test_size,
        batch_size=config.data.batch_size,
        sequence_length=config.data.sequence_length,
        frame_size=config.data.frame_size,
        num_digits=config.data.num_digits,
        num_workers=0,
        seed=config.data.seed
    )
    print(f"Test batches: {len(test_loader)}")
    
    model_name = Path(args.checkpoint).parent.name
    
    if not args.skip_metrics:
        print("COMPUTING METRICS")
        metrics = compute_metrics(model, test_loader, args.device)
        
        print(f"\nResults:")
        print(f"  MSE:  {metrics['mse']:.6f} ± {metrics['mse_std']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
        print(f"  Temporal Consistency: {metrics['temporal_consistency']:.6f} ± {metrics['temporal_consistency_std']:.6f}")
        
        metrics_path = os.path.join(args.save_dir, f'{model_name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {metrics_path}")
    

    print("VISUALIZATIONS")
    
    print("\n1. Reconstruction comparison")
    visualize_reconstructions(
        model, test_loader, args.device, 
        num_samples=args.num_samples,
        save_path=os.path.join(args.save_dir, f'{model_name}_reconstructions.png')
    )
    
    print("\n2. Random samples from model")
    visualize_samples(
        model, args.device,
        num_samples=args.num_samples,
        sequence_length=config.data.sequence_length,
        save_path=os.path.join(args.save_dir, f'{model_name}_samples.png')
    )
    
    print("\n3. Latent interpolation")
    interpolate_latent(
        model, test_loader, args.device,
        save_path=os.path.join(args.save_dir, f'{model_name}_interpolation.png')
    )
    

    print("EVALUATION COMPLETE")
    print(f"Results saved to: {args.save_dir}/")


if __name__ == '__main__':
    main()

