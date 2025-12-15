import torch
import json
import os

# Import evaluation modules
from evaluation.metrics import reconstruction_metrics, compare_models, print_comparison_table
from evaluation.disentanglement import compute_all_disentanglement_metrics, print_disentanglement_report
from data import get_moving_mnist_dataloaders
from models import BaselineVAE, LSTMVAE, DisentangledVAE


def load_model_from_checkpoint(checkpoint_path, model_class, model_kwargs, device):
    """Load a model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('best_val_loss', 'unknown')
        return model
    except Exception:
        return None


def main():
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Create results directory
    os.makedirs('eval_results', exist_ok=True)
    
    # Load test data
    _, _, test_loader = get_moving_mnist_dataloaders(
        batch_size=16,
        train_size=100,      # Small train set (not used)
        val_size=100,        # Small val set (not used)
        test_size=1000,      # Full test set for evaluation
        num_workers=0,       # Avoid multiprocessing issues
        seed=42
    )
    models = {}
    
    # Load Baseline VAE
    baseline = load_model_from_checkpoint(
        'checkpoints/baseline_vae/best_model.pt',
        BaselineVAE,
        {'latent_dim': 256},
        device
    )
    if baseline is not None:
        models['Baseline VAE'] = baseline
    
    # Load LSTM-VAE (with correct parameters: lstm_layers=2, bidirectional=True)
    lstm = load_model_from_checkpoint(
        'checkpoints/lstm_vae/best_model.pt',
        LSTMVAE,
        {'latent_dim': 256, 'lstm_hidden': 256, 'lstm_layers': 2, 'bidirectional': True},
        device
    )
    if lstm is not None:
        models['LSTM-VAE'] = lstm
    
    # Load Disentangled VAE (with correct parameters: lstm_layers=1, bidirectional=True)
    disentangled = load_model_from_checkpoint(
        'checkpoints/disentangled_vae/best_model.pt',
        DisentangledVAE,
        {'content_dim': 128, 'motion_dim': 128, 'lstm_layers': 1, 'bidirectional': True},
        device
    )
    if disentangled is not None:
        models['Disentangled VAE'] = disentangled
    
    if len(models) == 0:
        return
    
    try:
        results = compare_models(models, test_loader, device, max_batches=50)
        
        # Print comparison table
        print_comparison_table(results)
        
        # Save to JSON
        results_path = 'eval_results/comparison_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass
    
    if 'Disentangled VAE' in models:
        try:
            disentanglement_metrics = compute_all_disentanglement_metrics(
                models['Disentangled VAE'],
                test_loader,
                device,
                max_samples=1000
            )
            
            print_disentanglement_report(disentanglement_metrics)
            
            # Save disentanglement metrics
            disentangle_path = 'eval_results/disentanglement_metrics.json'
            with open(disentangle_path, 'w') as f:
                json.dump(disentanglement_metrics, f, indent=2)
        except Exception:
            pass


if __name__ == '__main__':
    main()
