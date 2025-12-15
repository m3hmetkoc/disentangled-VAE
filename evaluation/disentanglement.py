"""
Disentanglement Metrics for Video VAE Models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from tqdm import tqdm


def get_latent_codes(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    model.eval()
    
    z_content_list = []
    z_motion_list = []
    all_metadata = []
    
    sample_count = 0
    
    with torch.no_grad():
        for video, metadata in dataloader:
            if sample_count >= max_samples:
                break
                
            video = video.to(device)
            batch_size = video.size(0)
            
            model_type = getattr(model, 'model_type', 'unknown')
            
            if model_type == 'disentangled_vae':
                z_c, z_m, _, _ = model.encode(video)
                z_content_list.append(z_c.cpu().numpy())
                z_motion_list.append(z_m.cpu().numpy())
            else:
                z, mu, _ = model.encode(video)
                latent_dim = z.size(1)
                z_content_list.append(z[:, :latent_dim//2].cpu().numpy())
                z_motion_list.append(z[:, latent_dim//2:].cpu().numpy())
            
            all_metadata.extend(metadata)
            sample_count += batch_size
    
    z_content = np.concatenate(z_content_list, axis=0)[:max_samples]
    z_motion = np.concatenate(z_motion_list, axis=0)[:max_samples]
    all_metadata = all_metadata[:max_samples]
    
    return z_content, z_motion, all_metadata


def extract_labels(metadata: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    content_labels = []
    motion_labels = []
    
    for m in metadata:
        content_labels.append(m['labels'])
        
        if 'motion_labels' in m:
            directions = [ml['direction'] for ml in m['motion_labels']]
            motion_labels.append(directions)
        else:
            directions = []
            for vel in m['initial_velocities']:
                angle = np.arctan2(vel[1], vel[0])
                direction = int((angle + np.pi) / (2 * np.pi) * 8) % 8
                directions.append(direction)
            motion_labels.append(directions)
    
    return np.array(content_labels), np.array(motion_labels)


def sap_score(
    z_content: np.ndarray,
    z_motion: np.ndarray,
    content_labels: np.ndarray,
    motion_labels: np.ndarray
) -> Dict[str, float]:
    content_y = content_labels[:, 0]
    motion_y = motion_labels[:, 0]
    
    content_content_gap = _compute_sap_gap(z_content, content_y)
    content_motion_gap = _compute_sap_gap(z_content, motion_y)
    motion_motion_gap = _compute_sap_gap(z_motion, motion_y)
    motion_content_gap = _compute_sap_gap(z_motion, content_y)
    
    disentanglement_score = (
        (content_content_gap - content_motion_gap) +
        (motion_motion_gap - motion_content_gap)
    ) / 2
    
    return {
        'sap_content_content': content_content_gap,
        'sap_content_motion': content_motion_gap,
        'sap_motion_motion': motion_motion_gap,
        'sap_motion_content': motion_content_gap,
        'sap_disentanglement': disentanglement_score
    }


def _compute_sap_gap(z: np.ndarray, labels: np.ndarray) -> float:
    num_dims = z.shape[1]
    accuracies = []
    
    for d in range(num_dims):
        X = z[:, d:d+1]
        clf = LogisticRegression(max_iter=1000, random_state=42)
        try:
            clf.fit(X, labels)
            acc = clf.score(X, labels)
        except:
            acc = 1.0 / len(np.unique(labels))
        accuracies.append(acc)
    
    accuracies = np.sort(accuracies)[::-1]
    
    if len(accuracies) >= 2:
        return accuracies[0] - accuracies[1]
    return accuracies[0]


def correlation_metrics(
    z_content: np.ndarray,
    z_motion: np.ndarray
) -> Dict[str, float]:
    z_c = (z_content - z_content.mean(0)) / (z_content.std(0) + 1e-8)
    z_m = (z_motion - z_motion.mean(0)) / (z_motion.std(0) + 1e-8)
    
    correlation = np.abs(np.corrcoef(z_c.T, z_m.T))
    
    d_c = z_content.shape[1]
    d_m = z_motion.shape[1]
    cross_corr = correlation[:d_c, d_c:]
    
    return {
        'mean_abs_correlation': np.mean(np.abs(cross_corr)),
        'max_abs_correlation': np.max(np.abs(cross_corr)),
        'correlation_frobenius': np.linalg.norm(cross_corr, 'fro') / np.sqrt(d_c * d_m)
    }


def prediction_accuracy(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000
) -> Dict[str, float]:
    z_content, z_motion, metadata = get_latent_codes(model, dataloader, device, max_samples)
    content_labels, motion_labels = extract_labels(metadata)
    
    content_y = content_labels[:, 0]
    motion_y = motion_labels[:, 0]
    
    n = len(content_y)
    train_idx = np.random.choice(n, size=int(0.8*n), replace=False)
    test_idx = np.array([i for i in range(n) if i not in train_idx])
    
    scaler_c = StandardScaler()
    scaler_m = StandardScaler()
    
    z_c_train = scaler_c.fit_transform(z_content[train_idx])
    z_c_test = scaler_c.transform(z_content[test_idx])
    z_m_train = scaler_m.fit_transform(z_motion[train_idx])
    z_m_test = scaler_m.transform(z_motion[test_idx])
    
    results = {}
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(z_c_train, content_y[train_idx])
    results['content_predicts_content'] = clf.score(z_c_test, content_y[test_idx])
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(z_c_train, motion_y[train_idx])
    results['content_predicts_motion'] = clf.score(z_c_test, motion_y[test_idx])
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(z_m_train, motion_y[train_idx])
    results['motion_predicts_motion'] = clf.score(z_m_test, motion_y[test_idx])
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(z_m_train, content_y[train_idx])
    results['motion_predicts_content'] = clf.score(z_m_test, content_y[test_idx])
    
    results['content_disentanglement'] = (
        results['content_predicts_content'] - results['content_predicts_motion']
    )
    results['motion_disentanglement'] = (
        results['motion_predicts_motion'] - results['motion_predicts_content']
    )
    results['overall_disentanglement'] = (
        results['content_disentanglement'] + results['motion_disentanglement']
    ) / 2
    
    return results


def swap_success_rate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_pairs: int = 100,
    digit_classifier: Optional[nn.Module] = None
) -> Dict[str, float]:
    model.eval()
    
    videos = []
    metadata_list = []
    
    for video, metadata in dataloader:
        videos.append(video)
        metadata_list.extend(metadata)
        if len(videos) * video.size(0) >= num_pairs * 2:
            break
    
    videos = torch.cat(videos, dim=0)[:num_pairs * 2].to(device)
    metadata_list = metadata_list[:num_pairs * 2]
    
    video1 = videos[::2]
    video2 = videos[1::2]
    metadata1 = metadata_list[::2]
    metadata2 = metadata_list[1::2]
    
    with torch.no_grad():
        swapped = model.swap_content(video1, video2)
    
    avg_frame_original = video1.mean(dim=1)
    avg_frame_swapped = swapped.mean(dim=1)
    
    content_similarity = 1 - torch.mean((avg_frame_original - avg_frame_swapped).abs()).item()
    
    motion_original = (video2[:, 1:] - video2[:, :-1]).abs().mean(dim=(2, 3, 4))
    motion_swapped = (swapped[:, 1:] - swapped[:, :-1]).abs().mean(dim=(2, 3, 4))
    
    motion_correlation = float(np.corrcoef(
        motion_original.cpu().numpy().flatten(),
        motion_swapped.cpu().numpy().flatten()
    )[0, 1])
    
    return {
        'content_preservation': content_similarity,
        'motion_correlation': motion_correlation,
        'swap_quality': (content_similarity + max(0, motion_correlation)) / 2
    }


def compute_all_disentanglement_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000
) -> Dict[str, float]:
    model_type = getattr(model, 'model_type', 'unknown')
    results = {}
    
    print("Extracting latent codes")
    z_content, z_motion, metadata = get_latent_codes(model, dataloader, device, max_samples)
    
    if len(metadata) == 0 or 'labels' not in metadata[0]:
        print("Warning: No labels in metadata, skipping SAP and prediction metrics")
    else:
        content_labels, motion_labels = extract_labels(metadata)
        
        print("Computing SAP scores")
        sap = sap_score(z_content, z_motion, content_labels, motion_labels)
        results.update(sap)
        
        print("Computing prediction accuracy")
        pred_acc = prediction_accuracy(model, dataloader, device, max_samples)
        results.update(pred_acc)
    
    print("Computing correlation metrics")
    corr = correlation_metrics(z_content, z_motion)
    results.update(corr)
    
    if model_type == 'disentangled_vae':
        print("Computing swap success rate")
        swap = swap_success_rate(model, dataloader, device)
        results.update(swap)
    
    return results


def print_disentanglement_report(metrics: Dict[str, float]):
    sections = [
        ("SAP Scores", ['sap_content_content', 'sap_content_motion', 
                       'sap_motion_motion', 'sap_motion_content', 
                       'sap_disentanglement']),
        ("Prediction Accuracy", ['content_predicts_content', 'content_predicts_motion',
                                'motion_predicts_motion', 'motion_predicts_content',
                                'content_disentanglement', 'motion_disentanglement',
                                'overall_disentanglement']),
        ("Correlation", ['mean_abs_correlation', 'max_abs_correlation', 
                        'correlation_frobenius']),
        ("Swap Success", ['content_preservation', 'motion_correlation', 'swap_quality'])
    ]
    
    for section_name, keys in sections:
        section_metrics = {k: metrics[k] for k in keys if k in metrics}
        if section_metrics:
            for k, v in section_metrics.items():
                print(f"{k}: {v:.4f}")
    

if __name__ == '__main__':
    print("Disentanglement metrics module loaded successfully.")
    print("Use compute_all_disentanglement_metrics() to evaluate a model.")

