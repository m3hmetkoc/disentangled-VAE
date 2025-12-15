# Disentangled Content-Motion VAE for Video Generation

A research project implementing and comparing VAE architectures for video generation with explicit content-motion disentanglement.

## Research Question

> "Can we learn disentangled representations of visual content (what objects are present) and motion dynamics (how they move) in video VAE, and how does this explicit architectural disentanglement affect generation quality and controllability compared to implicit methods?"

## Project Overview

This project implements three VAE variants for Moving MNIST video generation:

1. **Baseline VAE**: Simple VAE without temporal modeling (comparison baseline)
2. **LSTM-VAE**: Standard approach with unified latent space and LSTM
3. **Disentangled VAE**: Novel architecture with explicit content/motion separation

### Key Innovation

The Disentangled VAE uses:
- **Content Encoder**: Extracts appearance (WHAT is in the video) using frame averaging
- **Motion Encoder**: Extracts dynamics (HOW things move) using frame differences + LSTM
- **Combined Decoder**: Generates video from both latent spaces

This enables:
- Content swapping (digit identity from video A + motion from video B)
- Motion swapping (motion from video A + digit identity from video B)  
- Independent interpolation in content and motion spaces

## Project Structure

```
RNN/
├── data/                    # Dataset generation
│   ├── moving_mnist.py      # Moving MNIST generator
│   └── __init__.py
├── models/                  # Model implementations
│   ├── encoders.py          # Shared encoder components
│   ├── decoders.py          # Shared decoder components
│   ├── baseline_vae.py      # Model 1: Baseline VAE
│   ├── lstm_vae.py          # Model 2: LSTM-VAE
│   ├── disentangled_vae.py  # Model 3: Disentangled VAE
│   └── __init__.py
├── training/                # Training infrastructure
│   ├── config.py            # Hyperparameter configs
│   ├── trainer.py           # Training loop
│   ├── train.py             # Main training script
│   └── __init__.py
├── evaluation/              # Evaluation metrics
│   ├── metrics.py           # MSE, SSIM, PSNR
│   ├── disentanglement.py   # SAP, MIG, correlation
│   └── __init__.py
├── utils/                   # Utilities
│   ├── visualization.py     # Plotting functions
│   └── __init__.py
├── experiments/             # Jupyter notebooks
│   ├── 01_reconstruction.ipynb
│   ├── 02_content_swap.ipynb
│   └── 04_interpolation.ipynb
├── checkpoints/             # Model checkpoints
├── results/                 # Experiment results
│   ├── figures/
│   └── videos/
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
cd RNN
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train Baseline VAE
python -m training.train --model baseline --name baseline_vae --epochs 100

# Train LSTM-VAE
python -m training.train --model lstm --name lstm_vae --epochs 100

# Train Disentangled VAE
python -m training.train --model disentangled --name disentangled_vae --epochs 100
```

### 3. Training Options

```bash
python -m training.train --help

Options:
  --model {baseline,lstm,disentangled}  Model type
  --name NAME                           Experiment name
  --epochs N                            Number of epochs (default: 100)
  --batch-size N                        Batch size (default: 32)
  --lr FLOAT                            Learning rate (default: 1e-4)
  --beta FLOAT                          KL divergence weight (default: 4.0)
  --content-dim N                       Content latent dimension (default: 128)
  --motion-dim N                        Motion latent dimension (default: 128)
  --recon-loss {mse,bce}                Reconstruction loss
  --device {mps,cuda,cpu}               Device (default: mps for M2 Mac)
  --resume                              Resume from checkpoint
```

---

## Model Evaluation Guide

This section walks you through the complete evaluation process after training your models.

### Quick Evaluation

The easiest way to evaluate your models is using the built-in `evaluate.py` script.

```bash
cd RNN

# Evaluate Baseline VAE
python evaluate.py \
    --checkpoint ./checkpoints/baseline_vae/best_model.pt \
    --device mps \
    --num_samples 5 \
    --save_dir ./eval_results

# Evaluate LSTM-VAE
python evaluate.py \
    --checkpoint ./checkpoints/lstm_vae/best_model.pt \
    --device mps \
    --num_samples 5 \
    --save_dir ./eval_results

# Evaluate Disentangled VAE
python evaluate.py \
    --checkpoint ./checkpoints/disentangled_vae/best_model.pt \
    --device mps \
    --num_samples 5 \
    --save_dir ./eval_results
```

#### What the Evaluation Script Does

For each model, it will:
1. Load the trained model
2. Compute reconstruction metrics (MSE, PSNR, SSIM)
3. Generate visualization of reconstructions
4. Generate random samples from the model
5. Show latent space interpolation
6. Save all results to `eval_results/`

#### Output Files

```
eval_results/
├── baseline_vae_metrics.json
├── baseline_vae_reconstructions.png
├── baseline_vae_samples.png
├── baseline_vae_interpolation.png
├── lstm_vae_metrics.json
├── lstm_vae_reconstructions.png
├── lstm_vae_samples.png
├── lstm_vae_interpolation.png
├── disentangled_vae_metrics.json
├── disentangled_vae_reconstructions.png
├── disentangled_vae_samples.png
└── disentangled_vae_interpolation.png
```

### Advanced Evaluation

Create a custom evaluation script for more control:

```python
#!/usr/bin/env python3
import torch
import json
from evaluation.metrics import reconstruction_metrics, compare_models
from evaluation.disentanglement import compute_all_disentanglement_metrics
from data import get_moving_mnist_dataloaders
from models import BaselineVAE, LSTMVAE, DisentangledVAE

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load test data
_, _, test_loader = get_moving_mnist_dataloaders(
    batch_size=16,
    train_size=100,
    val_size=100,
    test_size=1000,
    num_workers=0,
    seed=42
)

# Load all three models
models = {}

checkpoint = torch.load('checkpoints/baseline_vae/best_model.pt', map_location=device)
baseline = BaselineVAE(latent_dim=256)
baseline.load_state_dict(checkpoint['model_state_dict'])
models['Baseline VAE'] = baseline.to(device)

checkpoint = torch.load('checkpoints/lstm_vae/best_model.pt', map_location=device)
lstm = LSTMVAE(latent_dim=256, lstm_hidden=256)
lstm.load_state_dict(checkpoint['model_state_dict'])
models['LSTM-VAE'] = lstm.to(device)

checkpoint = torch.load('checkpoints/disentangled_vae/best_model.pt', map_location=device)
disentangled = DisentangledVAE(content_dim=128, motion_dim=128)
disentangled.load_state_dict(checkpoint['model_state_dict'])
models['Disentangled VAE'] = disentangled.to(device)

# Compare reconstruction metrics
results = compare_models(models, test_loader, device, max_batches=50)

# Compute disentanglement metrics
disentanglement_metrics = compute_all_disentanglement_metrics(
    models['Disentangled VAE'],
    test_loader,
    device,
    max_samples=1000
)

# Save results
with open('eval_results/comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('eval_results/disentanglement_metrics.json', 'w') as f:
    json.dump(disentanglement_metrics, f, indent=2)
```

### Experiment Notebooks

The `experiments/` folder contains Jupyter notebooks for detailed analysis.

```bash
cd RNN
jupyter notebook
```

**Notebook 1: Reconstruction Comparison** (`01_reconstruction.ipynb`)
- Compare reconstruction quality of all three models
- MSE, PSNR, SSIM metrics table
- Side-by-side visual comparison

**Notebook 2: Content/Motion Swapping** (`02_content_swap.ipynb`)
- Demonstrate disentanglement capability
- Content swap examples (digits from A, motion from B)
- KEY experiment showing disentanglement works
- Only works with Disentangled VAE

**Notebook 3: Interpolation** (`04_interpolation.ipynb`)
- Independent content and motion interpolation
- Content interpolation (morph digits, keep motion)
- Motion interpolation (change motion, keep digits)
- Shows independent control of content and motion

---

## Metrics to Report

### Reconstruction Quality Metrics

For all three models, report:

| Metric | Baseline VAE | LSTM-VAE | Disentangled VAE |
|--------|--------------|----------|------------------|
| **MSE** (lower is better) | ___ | ___ | ___ |
| **PSNR** (higher is better) | ___ dB | ___ dB | ___ dB |
| **SSIM** (higher is better) | ___ | ___ | ___ |
| **Temporal Consistency** (lower is better) | ___ | ___ | ___ |

**Interpretation:**
- MSE: Lower is better (less reconstruction error)
- PSNR: Higher is better (better signal quality, typically 20-40 dB)
- SSIM: Higher is better (closer to 1.0 = perfect)
- Temporal Consistency: Lower is better (smoother videos)

### Disentanglement Metrics (Disentangled VAE only)

Report from `disentanglement_metrics.json`:

**SAP Scores** (Separated Attribute Predictability):
- `sap_content_content`: Should be HIGH (content latents predict content)
- `sap_content_motion`: Should be LOW (content doesn't predict motion)
- `sap_motion_motion`: Should be HIGH (motion latents predict motion)
- `sap_motion_content`: Should be LOW (motion doesn't predict content)
- `sap_disentanglement`: Overall score (higher is better)

**Correlation Metrics**:
- `mean_abs_correlation`: Should be LOW (content and motion independent)
- `max_abs_correlation`: Should be LOW

**Swap Success**:
- `content_preservation`: How well content is preserved (higher is better)
- `motion_correlation`: How well motion is transferred (higher is better)

### Computational Efficiency

Report from comparison results:
- `trainable_parameters`: Model size
- `forward_time_ms`: Inference speed
- `generation_time_ms`: Sampling speed

---

## Creating Summary Tables

Create summary tables for your thesis/paper:

```python
#!/usr/bin/env python3
import json
import pandas as pd

# Load results
with open('eval_results/comparison_results.json') as f:
    comparison = json.load(f)

with open('eval_results/disentanglement_metrics.json') as f:
    disentanglement = json.load(f)

# Create reconstruction quality table
recon_table = pd.DataFrame({
    'Baseline VAE': [
        f"{comparison['Baseline VAE']['mse']:.6f}",
        f"{comparison['Baseline VAE']['psnr']:.2f}",
        f"{comparison['Baseline VAE']['ssim']:.4f}",
    ],
    'LSTM-VAE': [
        f"{comparison['LSTM-VAE']['mse']:.6f}",
        f"{comparison['LSTM-VAE']['psnr']:.2f}",
        f"{comparison['LSTM-VAE']['ssim']:.4f}",
    ],
    'Disentangled VAE': [
        f"{comparison['Disentangled VAE']['mse']:.6f}",
        f"{comparison['Disentangled VAE']['psnr']:.2f}",
        f"{comparison['Disentangled VAE']['ssim']:.4f}",
    ]
}, index=['MSE', 'PSNR (dB)', 'SSIM'])

print("RECONSTRUCTION QUALITY COMPARISON")
print(recon_table.to_string())

# Create disentanglement table
disentangle_table = pd.DataFrame({
    'Score': [
        f"{disentanglement['sap_disentanglement']:.4f}",
        f"{disentanglement['overall_disentanglement']:.4f}",
        f"{disentanglement['mean_abs_correlation']:.4f}",
        f"{disentanglement['swap_quality']:.4f}",
    ]
}, index=[
    'SAP Disentanglement',
    'Prediction Disentanglement',
    'Mean Correlation',
    'Swap Quality'
])

print("DISENTANGLEMENT METRICS")
print(disentangle_table.to_string())

# Save as LaTeX tables
recon_table.to_latex('eval_results/table_reconstruction.tex')
disentangle_table.to_latex('eval_results/table_disentanglement.tex')
```

---

## Complete Evaluation Workflow

### Workflow Checklist:

- [ ] Train all three models
- [ ] Run quick evaluation with `evaluate.py` for each model
- [ ] (Optional) Run custom evaluation script
- [ ] Open Jupyter and run `01_reconstruction.ipynb`
- [ ] Run `02_content_swap.ipynb` (KEY EXPERIMENT)
- [ ] Run `04_interpolation.ipynb`
- [ ] Run summary script to generate tables
- [ ] Collect all figures from `eval_results/` and `results/figures/`
- [ ] Write analysis and conclusions

---

## Troubleshooting

### Problem: FileNotFoundError - checkpoint not found
**Solution**: Verify checkpoint paths:
```bash
ls checkpoints/baseline_vae/best_model.pt
ls checkpoints/lstm_vae/best_model.pt
ls checkpoints/disentangled_vae/best_model.pt
```

### Problem: RuntimeError - state_dict doesn't match model
**Solution**: The checkpoint might be from a different model configuration. Check the config used during training.

### Problem: Module not found
**Solution**: Make sure you're in the RNN directory:
```bash
cd RNN
python -m training.train ...
```

### Problem: Notebook can't find models
**Solution**: In the notebook, add at the top:
```python
import sys
sys.path.insert(0, '..')
```

---

## Expected Results

| Metric | Baseline | LSTM-VAE | Disentangled |
|--------|----------|----------|--------------|
| MSE | Higher | Lower | Similar |
| SSIM | Lower | Higher | Similar |
| Content Swap | Fails | Partial | Clean |
| Motion Swap | Fails | Partial | Clean |
| Disentanglement | Low | Medium | High |

### Expected Research Findings

**Reconstruction Quality:**
- LSTM-VAE > Disentangled VAE ≈ Baseline VAE
- LSTM models temporal dynamics well

**Disentanglement:**
- Disentangled VAE >> LSTM-VAE >> Baseline VAE
- Explicit architectural separation forces disentanglement

**Content/Motion Swapping:**
- Disentangled VAE: Clean swaps
- LSTM-VAE: Partial/messy swaps
- Baseline VAE: Fails completely

### Key Contribution

> "We show that explicit architectural separation of content and motion encoders enables clean manipulation of video attributes (content swapping, independent interpolation) with minimal loss in reconstruction quality compared to implicit methods."

---

## Hardware Requirements

- Designed for M2 MacBook Air with MPS backend
- Also works on CUDA GPUs and CPU
- ~4GB RAM for training with batch_size=32

## License

This project is for educational and research purposes.

