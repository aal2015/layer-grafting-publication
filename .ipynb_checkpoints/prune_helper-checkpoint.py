import os
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from train_eval_func import eval_loop, get_primary_metric

# Pruning Sweeps
def remove_all_masks(mdl):
    """Strip any active pruning re-parametrization from all Linear layers."""
    for _, module in mdl.named_modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass  # layer was not pruned — nothing to remove

def apply_pruning(mdl, amount):
    """Apply l1_unstructured pruning to all Linear layers. Returns list of pruned modules."""
    pruned = []
    for name, module in mdl.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            pruned.append((name, module))
    return pruned

def global_sparsity(pruned_modules):
    """Compute fraction of zero weights across all pruned Linear layers."""
    total_zeros  = sum(float(torch.sum(m.weight == 0)) for _, m in pruned_modules)
    total_params = sum(float(m.weight.nelement())      for _, m in pruned_modules)
    return 100.0 * total_zeros / total_params

def make_permanent(pruned_modules):
    """Fuse masks into weights permanently."""
    for _, module in pruned_modules:
        prune.remove(module, 'weight')

def compare_pruning_robustness(
    models_dict,
    eval_dataloader,
    task_name,
    device,
    pruning_levels=[0.10, 0.20, 0.30, 0.40, 0.50],
    save_checkpoints=True,
    save_dir='./weights/pruning/'
):
    """
    Compare pruning robustness across multiple model architectures.
    
    Args:
        models_dict: Dictionary of {label: model} or {label: (model, baseline_metrics)}
                    Examples:
                      {'12-layer': model_12, '8-layer': model_8}
                    OR
                      {'12-layer': (model_12, {'accuracy': 0.82, 'f1': 0.85}),
                       '8-layer': (model_8, {'accuracy': 0.83, 'f1': 0.90})}
        eval_dataloader: Evaluation dataloader
        task_name: GLUE task name (e.g., 'mrpc', 'qnli')
        device: torch device
        pruning_levels: List of pruning fractions (default: [0.1, 0.2, 0.3, 0.4, 0.5])
        save_checkpoints: Whether to save pruned models (default: True)
        save_dir: Directory to save checkpoints (default: './weights/pruning/')
    
    Returns:
        Dictionary with results for each model: {label: [result_dicts]}
    """
    # Setup
    if save_checkpoints:
        os.makedirs(save_dir, exist_ok=True)
    
    primary_metric = get_primary_metric(task_name)
    
    # Parse models_dict - handle both formats
    parsed_models = {}
    for label, value in models_dict.items():
        if isinstance(value, tuple):
            # Format: (model, baseline_metrics)
            model, baseline = value
            parsed_models[label] = {
                'model': model,
                'baseline': baseline
            }
        else:
            # Format: just model, need to evaluate baseline
            model = value
            model.eval()
            baseline_metrics, _ = eval_loop(model, eval_dataloader, task_name, device)
            parsed_models[label] = {
                'model': model,
                'baseline': baseline_metrics
            }
    
    # Cache original states (on CPU to save GPU memory)
    print("Caching original model states...")
    cached_states = {}
    for label, data in parsed_models.items():
        model = data['model']
        model.eval()
        cached_states[label] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ✓ Cached {label}")
    
    # Display baselines
    print("\n" + "=" * 70)
    print("BASELINE METRICS")
    print("=" * 70)
    for label, data in parsed_models.items():
        baseline = data['baseline']
        print(f"\n{label}:")
        for metric_name, value in baseline.items():
            marker = "★" if metric_name == primary_metric else " "
            print(f"  {marker} {metric_name}: {value:.4f}")
    
    # Store results
    results_dict = {label: [] for label in models_dict.keys()}
    
    # Pruning sweep
    print("\n" + "=" * 70)
    print("PRUNING SWEEP")
    print("=" * 70)
    
    for amount in pruning_levels:
        pct = int(amount * 100)
        print(f'\n{"═" * 60}')
        print(f'  Pruning Level: {pct}%')
        print(f'{"═" * 60}')
        
        for label, data in parsed_models.items():
            model = data['model']
            baseline = data['baseline']
            baseline_score = baseline[primary_metric]
            
            print(f'\n  [{label}]')
            
            # Restore original weights
            remove_all_masks(model)
            model.load_state_dict({k: v.to(device) for k, v in cached_states[label].items()})
            model.eval()
            
            # Apply pruning
            pruned_params = apply_pruning(model, amount)
            
            # Evaluate
            metrics, _ = eval_loop(model, eval_dataloader, task_name, device)
            current_score = metrics[primary_metric]
            
            # Calculate sparsity
            sparsity = global_sparsity(pruned_params)
            
            # Make pruning permanent (removes masks)
            make_permanent(pruned_params)
            
            # Print results
            print(f'    {primary_metric.capitalize()}: {current_score:.4f}  '
                  f'(Δ {baseline_score - current_score:+.4f})')
            print(f'    Sparsity: {sparsity:.2f}%')
            
            # Store results
            result = {
                'pruning_pct': pct,
                'pruning_amount': amount,
                primary_metric: current_score,
                f'{primary_metric}_drop': baseline_score - current_score,
                'global_sparsity': sparsity,
            }
            
            # Add all metrics
            for metric_name, value in metrics.items():
                if metric_name != primary_metric:
                    result[metric_name] = value
            
            results_dict[label].append(result)
            
            # Save checkpoint if requested
            if save_checkpoints:
                save_filename = f'{label.replace(" ", "-").lower()}-pruned-{pct}pct.pt'
                save_path = os.path.join(save_dir, save_filename)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'pruning_amount': amount,
                    'global_sparsity': sparsity,
                    'baseline_metrics': baseline,
                    'pruned_metrics': metrics,
                    'label': label,
                }, save_path)
                print(f'    ✓ Saved: {save_filename}')
    
    print('\n✓ Pruning sweep complete.')
    
    return results_dict

def print_pruning_results(models_dict, results_dict):
    """Print formatted comparison table."""
    from train_eval_func import get_primary_metric
    
    # Get task name from first call (assume all same task)
    # This is a limitation - would need to pass task_name separately
    # For now, just print all metrics
    
    print("\n" + "=" * 80)
    print("PRUNING RESULTS SUMMARY")
    print("=" * 80)
    
    for label in models_dict.keys():
        baseline = None
        if isinstance(models_dict[label], tuple):
            baseline = models_dict[label][1]
        
        print(f'\n{label}:')
        print(f'{"Pruning %":>10}  {"Sparsity %":>12}  ', end='')
        
        # Print header with all metric names from first result
        if results_dict[label]:
            first_result = results_dict[label][0]
            metric_names = [k for k in first_result.keys() 
                          if k not in ['pruning_pct', 'pruning_amount', 'global_sparsity'] 
                          and not k.endswith('_drop')]
            for metric in metric_names:
                print(f'{metric.capitalize():>12}  ', end='')
        print()
        print('-' * 80)
        
        # Baseline row
        if baseline:
            print(f'{"0 (base)":>10}  {"0.00":>12}  ', end='')
            for metric in metric_names:
                if metric in baseline:
                    print(f'{baseline[metric]:>12.4f}  ', end='')
            print()
        
        # Pruning rows
        for r in results_dict[label]:
            print(f'{r["pruning_pct"]:>10}  {r["global_sparsity"]:>12.2f}  ', end='')
            for metric in metric_names:
                if metric in r:
                    print(f'{r[metric]:>12.4f}  ', end='')
            print()


def plot_pruning_comparison(
    results_dict,
    models_dict=None,
    task_name=None,
    save_path=None,
    figsize=(14, 10),
    show_plots=True,
    plot_types=['metric_vs_pruning', 'metric_drop', 'marginal', 'retention']
):
    """
    Plot pruning comparison results for any GLUE task.
    Automatically handles different metrics (F1, accuracy, Pearson, Matthews, etc.)
    
    Args:
        results_dict: Results from compare_pruning_robustness()
        models_dict: Optional - original models dict for baseline info
        task_name: Optional - GLUE task name (auto-detects primary metric)
        save_path: Optional - path to save figure (default: None, don't save)
        figsize: Figure size (default: (14, 10))
        show_plots: Whether to display plots (default: True)
        plot_types: List of plots to include (default: all 4)
                   Options: 'metric_vs_pruning', 'metric_drop', 'marginal', 'retention'
    
    Returns:
        matplotlib figure object
    
    Examples:
        # MRPC (F1)
        plot_pruning_comparison(results, models, 'mrpc', 'mrpc_pruning.png')
        
        # SST-2 (Accuracy)
        plot_pruning_comparison(results, models, 'sst2', 'sst2_pruning.png')
        
        # CoLA (Matthews correlation)
        plot_pruning_comparison(results, models, 'cola', 'cola_pruning.png')
        
        # STS-B (Pearson correlation)
        plot_pruning_comparison(results, models, 'stsb', 'stsb_pruning.png')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Detect primary metric
    primary_metric = None
    metric_display_name = None
    
    if task_name:
        from train_eval_func import get_primary_metric
        primary_metric = get_primary_metric(task_name)
        
        # Set display names
        metric_names = {
            'f1': 'F1',
            'accuracy': 'Accuracy',
            'matthews_correlation': 'Matthews Corr.',
            'pearson': 'Pearson Corr.',
            'spearmanr': 'Spearman Corr.',
        }
        metric_display_name = metric_names.get(primary_metric, primary_metric.capitalize())
    else:
        # Auto-detect from results
        first_label = list(results_dict.keys())[0]
        first_result = results_dict[first_label][0]
        
        # Find the metric (exclude special keys)
        for key in first_result.keys():
            if key not in ['pruning_pct', 'pruning_amount', 'global_sparsity'] and not key.endswith('_drop'):
                primary_metric = key
                metric_display_name = key.capitalize()
                break
    
    if not primary_metric:
        raise ValueError("Could not detect primary metric. Please provide task_name.")
    
    # Extract data
    labels = list(results_dict.keys())
    
    # Colors (support more models)
    default_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4']
    color_map = {
        '12-layer': '#2196F3',
        '8-layer': '#4CAF50',
        '7-layer': '#FF9800',
        '6-layer': '#F44336',
    }
    colors = [color_map.get(label, default_colors[i % len(default_colors)]) 
              for i, label in enumerate(labels)]
    
    # Get baselines
    baselines = {}
    for label in labels:
        if models_dict and label in models_dict:
            if isinstance(models_dict[label], tuple):
                baselines[label] = models_dict[label][1][primary_metric]
            else:
                # Try to get from first result
                first_res = results_dict[label][0]
                if primary_metric in first_res:
                    baselines[label] = first_res[primary_metric]
                else:
                    raise ValueError(f"Cannot find {primary_metric} for {label}")
        else:
            # Infer from first result
            first_res = results_dict[label][0]
            if primary_metric in first_res:
                baselines[label] = first_res[primary_metric]
            else:
                raise ValueError(f"Cannot find {primary_metric} in results for {label}")
    
    # Prepare data
    pruning_levels = [0] + [r['pruning_pct'] for r in results_dict[labels[0]]]
    
    data = {}
    for label in labels:
        baseline = baselines[label]
        
        # Get metric scores
        scores = [baseline] + [r.get(primary_metric, baseline) for r in results_dict[label]]
        
        # Get drops (handle both formats)
        drop_key = f'{primary_metric}_drop'
        if drop_key in results_dict[label][0]:
            drops = [0] + [r[drop_key] for r in results_dict[label]]
        else:
            # Calculate manually
            drops = [baseline - score for score in scores]
        
        data[label] = {
            'scores': scores,
            'drops': drops,
            'baseline': baseline
        }
    
    # Determine subplot layout
    n_plots = len(plot_types)
    if n_plots == 4:
        nrows, ncols = 2, 2
    elif n_plots == 3:
        nrows, ncols = 2, 2
    elif n_plots == 2:
        nrows, ncols = 1, 2
    else:
        nrows, ncols = 1, 1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot 1: Metric vs Pruning Level
    if 'metric_vs_pruning' in plot_types:
        ax = axes[plot_idx]
        plot_idx += 1
        
        for i, label in enumerate(labels):
            marker = ['o', 's', '^', 'd', 'v', 'p'][i % 6]
            ax.plot(pruning_levels, data[label]['scores'], 
                   f'{marker}-', color=colors[i], linewidth=2, markersize=8, label=label)
        
        ax.axvline(x=40, color='red', linestyle='--', alpha=0.3, linewidth=1.5)
        ax.set_xlabel('Pruning Level (%)', fontsize=11)
        ax.set_ylabel(metric_display_name, fontsize=11)
        
        title = f'{metric_display_name} vs Pruning Level'
        if task_name:
            title += f' ({task_name.upper()})'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(pruning_levels)
    
    # Plot 2: Drop vs Pruning Level
    if 'metric_drop' in plot_types:
        ax = axes[plot_idx]
        plot_idx += 1
        
        for i, label in enumerate(labels):
            marker = ['o', 's', '^', 'd', 'v', 'p'][i % 6]
            ax.plot(pruning_levels, data[label]['drops'],
                   f'{marker}-', color=colors[i], linewidth=2, markersize=8, label=label)
        
        ax.axvline(x=40, color='red', linestyle='--', alpha=0.3, linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=1)
        ax.set_xlabel('Pruning Level (%)', fontsize=11)
        ax.set_ylabel(f'{metric_display_name} Drop', fontsize=11)
        ax.set_title(f'{metric_display_name} Degradation', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(pruning_levels)
    
    # Plot 3: Marginal Drop
    if 'marginal' in plot_types:
        ax = axes[plot_idx]
        plot_idx += 1
        
        bar_width = 0.8 / len(labels)
        x_pos = np.arange(len(pruning_levels) - 1)
        
        for i, label in enumerate(labels):
            marginal = [data[label]['drops'][j+1] - data[label]['drops'][j] 
                       for j in range(len(data[label]['drops'])-1)]
            offset = (i - len(labels)/2 + 0.5) * bar_width
            ax.bar(x_pos + offset, marginal, bar_width, 
                  label=label, color=colors[i])
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{p}%' for p in pruning_levels[1:]])
        ax.set_xlabel('Pruning Step', fontsize=11)
        ax.set_ylabel(f'Marginal {metric_display_name} Drop', fontsize=11)
        ax.set_title('Marginal Degradation per Step', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=1)
    
    # Plot 4: Retention Rate
    if 'retention' in plot_types:
        ax = axes[plot_idx]
        plot_idx += 1
        
        for i, label in enumerate(labels):
            marker = ['o', 's', '^', 'd', 'v', 'p'][i % 6]
            baseline = data[label]['baseline']
            retention = [score/baseline * 100 for score in data[label]['scores']]
            ax.plot(pruning_levels, retention,
                   f'{marker}-', color=colors[i], linewidth=2, markersize=8, label=label)
        
        ax.axhline(y=95, color='green', linestyle='--', alpha=0.3, linewidth=1.5, label='95% retention')
        ax.axvline(x=40, color='red', linestyle='--', alpha=0.3, linewidth=1.5)
        ax.set_xlabel('Pruning Level (%)', fontsize=11)
        ax.set_ylabel(f'{metric_display_name} Retention (%)', fontsize=11)
        ax.set_title(f'{metric_display_name} Retention Rate', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(pruning_levels)
        ax.set_ylim([80, 102])
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Plot saved to {save_path}')
    
    # Show if requested
    if show_plots:
        plt.show()
    
    return fig

    
# Global Pruning — Attention vs FFN Separately

def get_global_pruning_params(mdl):
    """
    Partition all Linear weight tensors into two global pools:
      - attn_params : query, key, value, attention output projections
      - ffn_params  : intermediate dense + FFN output dense

    Works for both the depth model (12 standard layers) and the
    width model (6 merged layers with expanded dimensions).

    Returns (attn_params, ffn_params) as tuples of (module, 'weight') pairs
    ready to pass into prune.global_unstructured().
    """
    attn_params = []
    ffn_params  = []

    for name, module in mdl.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Attention sublayers — identified by name segments
        if any(k in name for k in [
            'attention.self.query',
            'attention.self.key',
            'attention.self.value',
            'attention.output.dense',
        ]):
            attn_params.append((module, 'weight'))

        # FFN sublayers
        elif any(k in name for k in [
            'intermediate.dense',
            'output.dense',      # BertOutput (FFN output), not attention output
        ]) and 'attention' not in name:
            ffn_params.append((module, 'weight'))

    return tuple(attn_params), tuple(ffn_params)


def apply_global_pruning(mdl, attn_amount, ffn_amount):
    """
    Apply global unstructured L1 pruning separately to the attention pool
    and the FFN pool.

    prune.global_unstructured ranks weights across ALL layers in the pool
    jointly — layers with smaller-magnitude weights get pruned more.

    Returns (attn_params, ffn_params) for sparsity reporting / removal.
    """
    attn_params, ffn_params = get_global_pruning_params(mdl)

    prune.global_unstructured(
        attn_params,
        pruning_method=prune.L1Unstructured,
        amount=attn_amount,
    )
    prune.global_unstructured(
        ffn_params,
        pruning_method=prune.L1Unstructured,
        amount=ffn_amount,
    )
    return attn_params, ffn_params


def pool_sparsity(params):
    """Compute sparsity (% zeros) across a pool of (module, 'weight') pairs."""
    total_zeros  = sum(float(torch.sum(m.weight == 0)) for m, _ in params)
    total_params = sum(float(m.weight.nelement())      for m, _ in params)
    return 100.0 * total_zeros / total_params if total_params > 0 else 0.0


def remove_global_masks(params):
    """Make global pruning permanent by fusing masks into weights."""
    for module, param_name in params:
        try:
            prune.remove(module, param_name)
        except ValueError:
            pass


def remove_all_global_masks(mdl):
    """Strip all pruning masks from the model before a new run."""
    for _, module in mdl.named_modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass

def describe_pools(mdl, label):
    attn_params, ffn_params = get_global_pruning_params(mdl)

    attn_total = sum(m.weight.nelement() for m, _ in attn_params)
    ffn_total  = sum(m.weight.nelement() for m, _ in ffn_params)
    grand_total = attn_total + ffn_total

    print(f'\n── {label} ──')
    print(f'  Attention pool : {len(attn_params):3d} Linear layers  |  {attn_total:>12,} params  ({100*attn_total/grand_total:.1f}%)')
    print(f'  FFN pool       : {len(ffn_params):3d} Linear layers  |  {ffn_total:>12,} params  ({100*ffn_total/grand_total:.1f}%)')
    print(f'  Total          :     {len(attn_params)+len(ffn_params):3d} Linear layers  |  {grand_total:>12,} params')

def print_per_layer_sparsity(mdl, label):
    print(f'\n── {label} ──')
    print(f'{"Layer":<55} {"Sparsity":>9} {"Pool":>8}')
    print('-' * 75)
    for name, module in mdl.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        zeros   = float(torch.sum(module.weight == 0))
        total   = float(module.weight.nelement())
        sparsity = 100.0 * zeros / total
        pool = ''
        if any(k in name for k in ['attention.self.query','attention.self.key',
                                    'attention.self.value','attention.output.dense']):
            pool = 'ATTN'
        elif 'attention' not in name and any(k in name for k in
                                              ['intermediate.dense','output.dense']):
            pool = 'FFN'
        else:
            pool = 'other'
        print(f'{name:<55} {sparsity:>8.2f}% {pool:>8}')

def compare_global_pruning_robustness(
    models_dict,
    eval_dataloader,
    task_name,
    device,
    pruning_configs=None,
    save_checkpoints=True,
    save_dir='./weights/pruning/global/'
):
    """
    Compare global pruning (attention vs FFN) robustness across models.
    
    Args:
        models_dict: Dictionary of {label: model} or {label: (model, baseline)}
        eval_dataloader: Evaluation dataloader
        task_name: GLUE task name
        device: torch device
        pruning_configs: List of (attn_amount, ffn_amount) tuples
                        Default: [(0.1, 0.1), (0.2, 0.2), ..., (0.5, 0.5)]
                        Examples:
                          - Uniform: [(0.3, 0.3)]
                          - FFN-only: [(0, 0.3)] or [(0, 0.4)]
                          - Attn-only: [(0.3, 0)] or [(0.4, 0)]
                          - FFN-heavy: [(0.2, 0.4), (0.25, 0.5)]
                          - Attn-heavy: [(0.4, 0.2), (0.5, 0.3)]
                          - Mixed: [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
        save_checkpoints: Whether to save pruned models (default: True)
        save_dir: Directory to save checkpoints
    
    Returns:
        Dictionary with results: {label: [result_dicts]}
    
    Examples:
        # Test FFN-only pruning
        results = compare_global_pruning_robustness(
            models_dict=models,
            eval_dataloader=val_dataloader,
            task_name='mrpc',
            device=device,
            pruning_configs=[
                (0, 0.2),  # FFN-only 20%
                (0, 0.3),  # FFN-only 30%
                (0, 0.4),  # FFN-only 40%
            ]
        )
        
        # Test attention-only pruning
        results = compare_global_pruning_robustness(
            models_dict=models,
            eval_dataloader=val_dataloader,
            task_name='mrpc',
            device=device,
            pruning_configs=[
                (0.2, 0),  # Attn-only 20%
                (0.3, 0),  # Attn-only 30%
                (0.4, 0),  # Attn-only 40%
            ]
        )
        
        # Compare all three strategies at 30%
        results = compare_global_pruning_robustness(
            models_dict=models,
            eval_dataloader=val_dataloader,
            task_name='mrpc',
            device=device,
            pruning_configs=[
                (0.3, 0.3),  # Uniform 30%
                (0, 0.3),    # FFN-only 30%
                (0.3, 0),    # Attn-only 30%
            ]
        )
    """
    import os
    import torch
    from train_eval_func import eval_loop, get_primary_metric
    
    # Setup
    if save_checkpoints:
        os.makedirs(save_dir, exist_ok=True)
    
    # Default: uniform pruning
    if pruning_configs is None:
        pruning_configs = [(x/100, x/100) for x in [10, 20, 30, 40, 50]]
    
    primary_metric = get_primary_metric(task_name)
    
    # Parse models_dict
    parsed_models = {}
    for label, value in models_dict.items():
        if isinstance(value, tuple):
            model, baseline = value
            parsed_models[label] = {'model': model, 'baseline': baseline}
        else:
            model = value
            model.eval()
            baseline_metrics, _ = eval_loop(model, eval_dataloader, task_name, device)
            parsed_models[label] = {'model': model, 'baseline': baseline_metrics}
    
    # Cache original states
    print("Caching original model states...")
    cached_states = {}
    for label, data in parsed_models.items():
        model = data['model']
        model.eval()
        cached_states[label] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ✓ Cached {label}")
    
    # Display baselines
    print("\n" + "=" * 70)
    print("BASELINE METRICS")
    print("=" * 70)
    for label, data in parsed_models.items():
        baseline = data['baseline']
        print(f"\n{label}:")
        for metric_name, value in baseline.items():
            marker = "★" if metric_name == primary_metric else " "
            print(f"  {marker} {metric_name}: {value:.4f}")
    
    # Store results
    results_dict = {label: [] for label in models_dict.keys()}
    
    # Pruning sweep
    print("\n" + "=" * 70)
    print("GLOBAL PRUNING SWEEP (Attention vs FFN)")
    print("=" * 70)
    
    for attn_amount, ffn_amount in pruning_configs:
        attn_pct = int(attn_amount * 100)
        ffn_pct = int(ffn_amount * 100)
        
        # Determine pruning type for display
        if attn_amount == 0 and ffn_amount > 0:
            pruning_type = f"FFN-only {ffn_pct}%"
        elif ffn_amount == 0 and attn_amount > 0:
            pruning_type = f"Attn-only {attn_pct}%"
        elif attn_amount == ffn_amount:
            pruning_type = f"Uniform {attn_pct}%"
        else:
            avg_pct = int((attn_amount + ffn_amount) / 2 * 100)
            pruning_type = f"attn={attn_pct}% ffn={ffn_pct}% (avg={avg_pct}%)"
        
        print(f'\n{"═" * 62}')
        print(f'  {pruning_type}')
        print(f'{"═" * 62}')
        
        for label, data in parsed_models.items():
            model = data['model']
            baseline = data['baseline']
            baseline_score = baseline[primary_metric]
            
            print(f'\n  [{label}]')
            
            # Restore original weights
            remove_all_global_masks(model)
            model.load_state_dict({k: v.to(device) for k, v in cached_states[label].items()})
            model.eval()
            
            # Apply global pruning
            attn_params, ffn_params = apply_global_pruning(model, attn_amount, ffn_amount)
            
            # Evaluate
            metrics, _ = eval_loop(model, eval_dataloader, task_name, device)
            current_score = metrics[primary_metric]
            
            # Calculate sparsities
            attn_sparsity = pool_sparsity(attn_params)
            ffn_sparsity = pool_sparsity(ffn_params)
            
            # Calculate total sparsity (weighted average)
            attn_total = sum(m.weight.nelement() for m, _ in attn_params)
            ffn_total = sum(m.weight.nelement() for m, _ in ffn_params)
            total_sparsity = (attn_sparsity * attn_total + ffn_sparsity * ffn_total) / (attn_total + ffn_total)
            
            # Make pruning permanent
            remove_global_masks(attn_params)
            remove_global_masks(ffn_params)
            
            # Print results
            print(f'    {primary_metric.capitalize()}: {current_score:.4f}  '
                  f'(Δ {baseline_score - current_score:+.4f})')
            
            # Only show relevant sparsities
            if attn_amount > 0:
                print(f'    Attn sparsity: {attn_sparsity:.2f}%')
            if ffn_amount > 0:
                print(f'    FFN sparsity:  {ffn_sparsity:.2f}%')
            if attn_amount > 0 and ffn_amount > 0:
                print(f'    Total sparsity: {total_sparsity:.2f}%')
            
            # Store results
            result = {
                'attn_pct': attn_pct,
                'ffn_pct': ffn_pct,
                'attn_amount': attn_amount,
                'ffn_amount': ffn_amount,
                'pruning_type': pruning_type,  # Added for easier filtering
                primary_metric: current_score,
                f'{primary_metric}_drop': baseline_score - current_score,
                'attn_sparsity': attn_sparsity,
                'ffn_sparsity': ffn_sparsity,
                'total_sparsity': total_sparsity,
            }
            
            # Add all metrics
            for metric_name, value in metrics.items():
                if metric_name != primary_metric:
                    result[metric_name] = value
            
            results_dict[label].append(result)
            
            # Save checkpoint if requested
            if save_checkpoints:
                # Better filename based on pruning type
                if attn_amount == 0:
                    save_filename = f'{label.replace(" ", "-").lower()}-ffn-only-{ffn_pct}pct.pt'
                elif ffn_amount == 0:
                    save_filename = f'{label.replace(" ", "-").lower()}-attn-only-{attn_pct}pct.pt'
                else:
                    save_filename = f'{label.replace(" ", "-").lower()}-global-attn{attn_pct}-ffn{ffn_pct}.pt'
                
                save_path_full = os.path.join(save_dir, save_filename)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'attn_amount': attn_amount,
                    'ffn_amount': ffn_amount,
                    'pruning_type': pruning_type,
                    'attn_sparsity': attn_sparsity,
                    'ffn_sparsity': ffn_sparsity,
                    'total_sparsity': total_sparsity,
                    'baseline_metrics': baseline,
                    'pruned_metrics': metrics,
                    'label': label,
                }, save_path_full)
                print(f'    ✓ Saved: {save_filename}')
    
    print('\n✓ Global pruning sweep complete.')
    
    return results_dict