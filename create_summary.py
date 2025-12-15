import json
import os
import pandas as pd


def load_json_safe(filepath):
    """Load JSON file safely, return None if file doesn't exist."""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath) as f:
        return json.load(f)


def create_reconstruction_table(comparison):
    """Create reconstruction quality comparison table."""
    if comparison is None:
        return None
    
    # Define metrics to include
    metrics = [
        ('mse', 'MSE ↓', '.6f'),
        ('psnr', 'PSNR (dB) ↑', '.2f'),
        ('ssim', 'SSIM ↑', '.4f'),
        ('temporal_consistency', 'Temporal Consistency', '.6f'),
        ('trainable_parameters', 'Parameters', ',d'),
        ('forward_time_ms', 'Forward Time (ms)', '.2f'),
    ]
    
    # Prepare data
    data = {}
    for model_name in comparison.keys():
        model_data = []
        for metric_key, _, fmt in metrics:
            if metric_key in comparison[model_name]:
                value = comparison[model_name][metric_key]
                if isinstance(value, float):
                    model_data.append(f"{value:{fmt}}")
                else:
                    model_data.append(f"{value}")
            else:
                model_data.append("N/A")
        data[model_name] = model_data
    
    # Create DataFrame
    index = [label for _, label, _ in metrics]
    df = pd.DataFrame(data, index=index)
    
    return df


def create_disentanglement_table(disentanglement):
    """Create disentanglement metrics table."""
    if disentanglement is None:
        return None
    
    # Define metrics to include
    metrics = [
        ('sap_disentanglement', 'SAP Disentanglement Score ↑'),
        ('overall_disentanglement', 'Prediction Disentanglement ↑'),
        ('mean_abs_correlation', 'Mean Abs. Correlation ↓'),
        ('max_abs_correlation', 'Max Abs. Correlation ↓'),
        ('content_preservation', 'Content Preservation (Swap) ↑'),
        ('motion_correlation', 'Motion Correlation (Swap) ↑'),
        ('swap_quality', 'Overall Swap Quality ↑'),
    ]
    
    # Prepare data
    data = []
    for metric_key, label in metrics:
        if metric_key in disentanglement:
            value = disentanglement[metric_key]
            data.append(f"{value:.4f}")
        else:
            data.append("N/A")
    
    # Create DataFrame
    index = [label for _, label in metrics]
    df = pd.DataFrame({'Score': data}, index=index)
    
    return df


def save_latex_table(df, filepath, caption, label):
    """Save DataFrame as LaTeX table."""
    if df is None:
        return
    
    latex_str = df.to_latex(
        column_format='l' + 'r' * len(df.columns),
        escape=False,
        caption=caption,
        label=label
    )
    
    with open(filepath, 'w') as f:
        f.write(latex_str)
    

def create_summary_text(comparison, disentanglement):
    """Create a text summary of key findings."""
    if comparison is None:
        return "No comparison results available."
    
    lines = []
    lines.append("="*80)
    lines.append("KEY FINDINGS SUMMARY")
    lines.append("="*80)
    
    if len(comparison) > 0:
        lines.append("\nReconstruction Quality:")
        
        # MSE (lower is better)
        mse_scores = {name: data['mse'] for name, data in comparison.items() if 'mse' in data}
        if mse_scores:
            best_mse = min(mse_scores.items(), key=lambda x: x[1])
            lines.append(f"   Best MSE: {best_mse[0]} ({best_mse[1]:.6f})")
        
        # PSNR (higher is better)
        psnr_scores = {name: data['psnr'] for name, data in comparison.items() if 'psnr' in data}
        if psnr_scores:
            best_psnr = max(psnr_scores.items(), key=lambda x: x[1])
            lines.append(f"   Best PSNR: {best_psnr[0]} ({best_psnr[1]:.2f} dB)")
        
        # SSIM (higher is better)
        ssim_scores = {name: data['ssim'] for name, data in comparison.items() if 'ssim' in data}
        if ssim_scores:
            best_ssim = max(ssim_scores.items(), key=lambda x: x[1])
            lines.append(f"   Best SSIM: {best_ssim[0]} ({best_ssim[1]:.4f})")
    
    if disentanglement:
        lines.append("\nDisentanglement (Disentangled VAE):")
        if 'sap_disentanglement' in disentanglement:
            lines.append(f"   SAP Score: {disentanglement['sap_disentanglement']:.4f}")
        if 'overall_disentanglement' in disentanglement:
            lines.append(f"   Prediction Disentanglement: {disentanglement['overall_disentanglement']:.4f}")
        if 'swap_quality' in disentanglement:
            lines.append(f"   Swap Quality: {disentanglement['swap_quality']:.4f}")
    
    lines.append("\n" + "="*80)
    
    return "\n".join(lines)


def main():
    os.makedirs('eval_results', exist_ok=True)
    
    comparison = load_json_safe('eval_results/comparison_results.json')
    disentanglement = load_json_safe('eval_results/disentanglement_metrics.json')
    
    if comparison is None and disentanglement is None:
        return
    
    recon_table = create_reconstruction_table(comparison)
    disentangle_table = create_disentanglement_table(disentanglement)
    
    if recon_table is not None:
        csv_path = 'eval_results/table_reconstruction.csv'
        recon_table.to_csv(csv_path)
    
    if disentangle_table is not None:
        csv_path = 'eval_results/table_disentanglement.csv'
        disentangle_table.to_csv(csv_path)
    
    if recon_table is not None:
        save_latex_table(
            recon_table,
            'eval_results/table_reconstruction.tex',
            'Reconstruction quality comparison across three VAE architectures.',
            'tab:reconstruction'
        )
    
    if disentangle_table is not None:
        save_latex_table(
            disentangle_table,
            'eval_results/table_disentanglement.tex',
            'Disentanglement metrics for the proposed Disentangled VAE model.',
            'tab:disentanglement'
        )
    
    summary = create_summary_text(comparison, disentanglement)
    
    summary_path = 'eval_results/summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    
if __name__ == '__main__':
    main()
