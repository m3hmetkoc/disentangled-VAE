"""
Evaluation Metrics for Video VAE Models
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from skimage.metrics import structural_similarity as ssim_sklearn
from tqdm import tqdm
import time


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred, target, reduction='mean').item()


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse_val = F.mse_loss(pred, target, reduction='mean').item()
    if mse_val == 0:
        return float('inf')
    return 10 * np.log10(max_val**2 / mse_val)


def ssim_video(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    batch_size, seq_len = pred_np.shape[:2]
    ssim_values = []
    
    for b in range(batch_size):
        for t in range(seq_len):
            pred_frame = pred_np[b, t, 0]
            target_frame = target_np[b, t, 0]
            
            ssim_val = ssim_sklearn(
                target_frame, 
                pred_frame,
                data_range=1.0,
                win_size=7
            )
            ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


def temporal_consistency(video: torch.Tensor) -> float:
    frame_diffs = video[:, 1:] - video[:, :-1]
    return frame_diffs.abs().mean().item()


def reconstruction_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    model.eval()
    
    all_mse = []
    all_psnr = []
    all_ssim = []
    all_temporal = []
    
    with torch.no_grad():
        for i, (video, metadata) in enumerate(tqdm(dataloader, desc='Evaluating')):
            if max_batches and i >= max_batches:
                break
                
            video = video.to(device)
            
            model_type = getattr(model, 'model_type', 'unknown')
            if model_type == 'disentangled_vae':
                output = model(video)
                recon = output['recon']
            else:
                recon, _, _, _ = model(video)
            
            all_mse.append(mse(recon, video))
            all_psnr.append(psnr(recon, video))
            all_ssim.append(ssim_video(recon, video))
            all_temporal.append(temporal_consistency(recon))
    
    return {
        'mse': np.mean(all_mse),
        'psnr': np.mean(all_psnr),
        'ssim': np.mean(all_ssim),
        'temporal_consistency': np.mean(all_temporal),
        'mse_std': np.std(all_mse),
        'psnr_std': np.std(all_psnr),
        'ssim_std': np.std(all_ssim)
    }


def generation_diversity(
    model: torch.nn.Module,
    num_samples: int = 100,
    device: torch.device = None
) -> Dict[str, float]:
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(num_samples=num_samples, device=device)
    
    sample_variance = samples.var(dim=0).mean().item()
    
    samples_flat = samples.view(num_samples, -1)
    distances = torch.cdist(samples_flat, samples_flat)
    mean_distance = distances[torch.triu(torch.ones_like(distances), diagonal=1).bool()].mean().item()
    
    temporal_var = samples.var(dim=1).mean().item()
    
    return {
        'sample_variance': sample_variance,
        'mean_pairwise_distance': mean_distance,
        'temporal_variance': temporal_var
    }


def computational_efficiency(
    model: torch.nn.Module,
    video: torch.Tensor,
    num_runs: int = 100,
    device: torch.device = None
) -> Dict[str, float]:
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    video = video.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(video)
    
    encode_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            if hasattr(model, 'encode'):
                _ = model.encode(video)
            else:
                _ = model(video)
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            encode_times.append(time.time() - start)
    
    forward_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(video)
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            forward_times.append(time.time() - start)
    
    gen_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model.sample(num_samples=video.size(0), device=device)
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            gen_times.append(time.time() - start)
    
    return {
        'total_parameters': num_params,
        'trainable_parameters': trainable_params,
        'encode_time_ms': np.mean(encode_times) * 1000,
        'encode_time_std_ms': np.std(encode_times) * 1000,
        'forward_time_ms': np.mean(forward_times) * 1000,
        'forward_time_std_ms': np.std(forward_times) * 1000,
        'generation_time_ms': np.mean(gen_times) * 1000,
        'generation_time_std_ms': np.std(gen_times) * 1000
    }


def compare_models(
    models: Dict[str, torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 50
) -> Dict[str, Dict[str, float]]:
    results = {}
    
    sample_video, _ = next(iter(dataloader))
    sample_video = sample_video[:4].to(device)
    
    for name, model in models.items():
        print(f"\nEvaluating {name}")
        model = model.to(device)
        
        recon_metrics = reconstruction_metrics(model, dataloader, device, max_batches)
        div_metrics = generation_diversity(model, num_samples=50, device=device)
        eff_metrics = computational_efficiency(model, sample_video, num_runs=50, device=device)
        
        results[name] = {**recon_metrics, **div_metrics, **eff_metrics}
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    key_metrics = [
        ('mse', 'MSE ↓', '{:.6f}'),
        ('psnr', 'PSNR ↑', '{:.2f}'),
        ('ssim', 'SSIM ↑', '{:.4f}'),
        ('temporal_consistency', 'Temp. Cons.', '{:.6f}'),
        ('sample_variance', 'Diversity', '{:.6f}'),
        ('trainable_parameters', 'Parameters', '{:,}'),
        ('forward_time_ms', 'Forward (ms)', '{:.2f}')
    ]
    
    model_names = list(results.keys())
    header = ['Metric'] + model_names
    
    print("MODEL COMPARISON")
    print(f"{'Metric':<20}", end='')
    for name in model_names:
        print(f"{name:<20}", end='')
    print()
    
    for metric_key, metric_name, fmt in key_metrics:
        print(f"{metric_name:<20}", end='')
        for name in model_names:
            if metric_key in results[name]:
                val = results[name][metric_key]
                print(f"{fmt.format(val):<20}", end='')
            else:
                print(f"{'N/A':<20}", end='')
        print()

if __name__ == '__main__':
    print("Testing metrics")
    
    pred = torch.rand(2, 20, 1, 64, 64)
    target = torch.rand(2, 20, 1, 64, 64)
    
    print(f"MSE: {mse(pred, target):.6f}")
    print(f"PSNR: {psnr(pred, target):.2f} dB")
    print(f"SSIM: {ssim_video(pred, target):.4f}")
    print(f"Temporal consistency: {temporal_consistency(pred):.6f}")
    
    print("\nMetrics test passed!")

