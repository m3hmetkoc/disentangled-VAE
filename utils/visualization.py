import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from typing import List, Optional, Tuple, Dict
import os

plt.style.use('seaborn-v0_8-whitegrid')

def video_to_frames(video: torch.Tensor, num_frames: int = 10) -> np.ndarray:
    if video.dim() == 5:
        video = video[0]  # Take first batch
    
    video = video.detach().cpu().numpy()
    T = video.shape[0]
    
    indices = np.linspace(0, T-1, num_frames, dtype=int)
    frames = video[indices, 0]
    
    return frames


def plot_video_frames(
    video: torch.Tensor,
    num_frames: int = 10,
    title: str = '',
    figsize: Tuple[int, int] = (15, 2),
    save_path: Optional[str] = None
) -> plt.Figure:
    frames = video_to_frames(video, num_frames)
    
    fig, axes = plt.subplots(1, num_frames, figsize=figsize)
    
    for i, ax in enumerate(axes):
        ax.imshow(frames[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f't={i * (len(video[0] if video.dim() == 5 else video) - 1) // (num_frames - 1)}')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_reconstruction_comparison(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    num_frames: int = 8,
    title: str = 'Reconstruction Comparison',
    figsize: Tuple[int, int] = (16, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    orig_frames = video_to_frames(original, num_frames)
    recon_frames = video_to_frames(reconstruction, num_frames)
    
    fig, axes = plt.subplots(2, num_frames, figsize=figsize)
    
    for i in range(num_frames):
        axes[0, i].imshow(orig_frames[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)
        
        axes[1, i].imshow(recon_frames[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=12)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_multiple_reconstructions(
    originals: List[torch.Tensor],
    reconstructions: List[torch.Tensor],
    num_frames: int = 6,
    model_names: Optional[List[str]] = None,
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    n_models = len(reconstructions)
    
    if figsize is None:
        figsize = (num_frames * 2, (n_models + 1) * 2)
    
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(n_models)]
    
    fig, axes = plt.subplots(n_models + 1, num_frames, figsize=figsize)
    
    orig_frames = video_to_frames(originals[0], num_frames)
    for i in range(num_frames):
        axes[0, i].imshow(orig_frames[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=11)
    
    for m, (recon, name) in enumerate(zip(reconstructions, model_names)):
        recon_frames = video_to_frames(recon, num_frames)
        for i in range(num_frames):
            axes[m+1, i].imshow(recon_frames[i], cmap='gray', vmin=0, vmax=1)
            axes[m+1, i].axis('off')
        axes[m+1, 0].set_ylabel(name, fontsize=11)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_interpolation(
    interpolated_videos: torch.Tensor,
    num_frames: int = 5,
    title: str = 'Latent Space Interpolation',
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    n_interp = interpolated_videos.shape[0]
    
    if figsize is None:
        figsize = (num_frames * 1.5, n_interp * 1.5)
    
    fig, axes = plt.subplots(n_interp, num_frames, figsize=figsize)
    
    for v in range(n_interp):
        frames = video_to_frames(interpolated_videos[v:v+1], num_frames)
        for f in range(num_frames):
            axes[v, f].imshow(frames[f], cmap='gray', vmin=0, vmax=1)
            axes[v, f].axis('off')
        
        alpha = v / (n_interp - 1)
        axes[v, 0].set_ylabel(f'α={alpha:.1f}', fontsize=10)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_swap_results(
    video_content: torch.Tensor,
    video_motion: torch.Tensor,
    swapped_video: torch.Tensor,
    num_frames: int = 8,
    title: str = 'Content/Motion Swap',
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    content_frames = video_to_frames(video_content, num_frames)
    motion_frames = video_to_frames(video_motion, num_frames)
    swapped_frames = video_to_frames(swapped_video, num_frames)
    
    fig, axes = plt.subplots(3, num_frames, figsize=figsize)
    
    labels = ['Content Source', 'Motion Source', 'Swapped Result']
    frames_list = [content_frames, motion_frames, swapped_frames]
    
    for row, (frames, label) in enumerate(zip(frames_list, labels)):
        for i in range(num_frames):
            axes[row, i].imshow(frames[i], cmap='gray', vmin=0, vmax=1)
            axes[row, i].axis('off')
        axes[row, 0].set_ylabel(label, fontsize=11)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_latent_space(
    z_content: np.ndarray,
    z_motion: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'tsne',
    title: str = 'Latent Space Visualization',
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    z_content_2d = reducer.fit_transform(z_content)
    z_motion_2d = reducer.fit_transform(z_motion)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if labels is not None:
        scatter_kwargs = {'c': labels, 'cmap': 'tab10', 's': 20, 'alpha': 0.7}
    else:
        scatter_kwargs = {'c': 'blue', 's': 20, 'alpha': 0.7}
    
    sc1 = axes[0].scatter(z_content_2d[:, 0], z_content_2d[:, 1], **scatter_kwargs)
    axes[0].set_title('Content Latent Space')
    axes[0].set_xlabel(f'{method.upper()} 1')
    axes[0].set_ylabel(f'{method.upper()} 2')
    
    sc2 = axes[1].scatter(z_motion_2d[:, 0], z_motion_2d[:, 1], **scatter_kwargs)
    axes[1].set_title('Motion Latent Space')
    axes[1].set_xlabel(f'{method.upper()} 1')
    axes[1].set_ylabel(f'{method.upper()} 2')
    
    if labels is not None:
        plt.colorbar(sc1, ax=axes[0], label='Label')
        plt.colorbar(sc2, ax=axes[1], label='Label')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = 'Training Curves',
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['train_recon'], label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_recon'], label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['train_kl'], label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_kl'], label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, history['learning_rate'], linewidth=2, color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    if metrics is None:
        metrics = ['mse', 'psnr', 'ssim', 'sample_variance']
    
    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    x = np.arange(n_models)
    
    for i, metric in enumerate(metrics):
        values = [results[name].get(metric, 0) for name in model_names]
        
        bars = axes[i].bar(x, values, color=plt.cm.Set2(range(n_models)))
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        axes[i].set_title(metric.upper())
        axes[i].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[i].annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_video_animation(
    video: torch.Tensor,
    save_path: str,
    fps: int = 10,
    dpi: int = 100
):
    if video.dim() == 5:
        video = video[0]
    
    video_np = video.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    
    im = ax.imshow(video_np[0, 0], cmap='gray', vmin=0, vmax=1)
    
    def update(frame):
        im.set_array(video_np[frame, 0])
        return [im]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(video_np), interval=1000/fps, blit=True
    )
    
    if save_path.endswith('.gif'):
        anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
    else:
        anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
    
    plt.close(fig)


def create_comparison_animation(
    videos: List[torch.Tensor],
    labels: List[str],
    save_path: str,
    fps: int = 10,
    dpi: int = 100
):
    n_videos = len(videos)
    
    videos_np = [v[0].detach().cpu().numpy() if v.dim() == 5 else v.detach().cpu().numpy() 
                 for v in videos]
    
    fig, axes = plt.subplots(1, n_videos, figsize=(3*n_videos, 3))
    if n_videos == 1:
        axes = [axes]
    
    ims = []
    for i, (ax, video_np, label) in enumerate(zip(axes, videos_np, labels)):
        ax.axis('off')
        ax.set_title(label)
        im = ax.imshow(video_np[0, 0], cmap='gray', vmin=0, vmax=1)
        ims.append(im)
    
    def update(frame):
        for im, video_np in zip(ims, videos_np):
            im.set_array(video_np[frame, 0])
        return ims
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(videos_np[0]), interval=1000/fps, blit=True
    )
    
    if save_path.endswith('.gif'):
        anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
    else:
        anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
    
    plt.close(fig)

if __name__ == '__main__':
    print("Testing visualization utilities")
    
    video = torch.rand(1, 20, 1, 64, 64)
    
    fig = plot_video_frames(video, num_frames=10, title='Test Video')
    plt.close(fig)
    
    recon = torch.rand(1, 20, 1, 64, 64)
    fig = plot_reconstruction_comparison(video, recon)
    plt.close(fig)
    
    print("Visualization tests passed!")


