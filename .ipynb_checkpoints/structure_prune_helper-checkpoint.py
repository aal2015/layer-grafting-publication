import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from train_eval_func import get_primary_metric, eval_loop

def register_importance_masks(model, device, register_heads=True, register_ffn=True):
    """
    Register importance masks for heads and/or FFN.
    
    Flexible registration - can register only heads, only FFN, or both.
    
    Args:
        model: BERT model
        device: torch device
        register_heads: Whether to register head masks (default: True)
        register_ffn: Whether to register FFN masks (default: True)
    
    Examples:
        # Register both (most common)
        register_importance_masks(model, device)
        
        # Register only heads
        register_importance_masks(model, device, register_heads=True, register_ffn=False)
        
        # Register only FFN
        register_importance_masks(model, device, register_heads=False, register_ffn=True)
    """
    if not register_heads and not register_ffn:
        raise ValueError("Must register at least one of: heads or FFN")
    
    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        if register_heads:
            n_heads = layer.attention.self.num_attention_heads
            layer.head_mask_param = nn.Parameter(
                torch.ones(n_heads, dtype=torch.float32, device=device),
                requires_grad=True
            )
        
        if register_ffn:
            ffn_dim = layer.intermediate.dense.out_features
            layer.int_mask_param = nn.Parameter(
                torch.ones(ffn_dim, dtype=torch.float32, device=device),
                requires_grad=True
            )
        
        # Print registration info
        parts = []
        if register_heads:
            parts.append(f"{layer.attention.self.num_attention_heads} heads")
        if register_ffn:
            parts.append(f"{layer.intermediate.dense.out_features} neurons")
        
        print(f"  Layer {layer_idx}: Registered masks ({', '.join(parts)})")
    
    return model
 
 
def remove_importance_masks(model, remove_heads=True, remove_ffn=True):
    """
    Remove registered masks from layers.
    
    Args:
        model: BERT model
        remove_heads: Whether to remove head masks (default: True)
        remove_ffn: Whether to remove FFN masks (default: True)
    """
    for layer in model.bert.encoder.layer:
        if remove_heads:
            layer.head_mask_param = None
        if remove_ffn:
            layer.int_mask_param = None
    
    parts = []
    if remove_heads:
        parts.append("head")
    if remove_ffn:
        parts.append("FFN")
    
    print(f"✓ Removed {' and '.join(parts)} masks")

def apply_head_masking(model, head_importance, num_to_mask):
    """Apply masking to head_mask_param based on importance."""
    
    # Count total heads
    total_heads = sum(len(imp) for imp in head_importance)
    
    print(f"\nMasking {num_to_mask} / {total_heads} heads ({num_to_mask/total_heads*100:.1f}%)")
    
    # Collect all (importance, layer_idx, head_idx) for ACTIVE heads only
    all_heads = []
    for layer_idx, layer_imp in enumerate(head_importance):
        for head_idx in range(len(layer_imp)):
            # SKIP if already masked
            if model.bert.encoder.layer[layer_idx].head_mask_param[head_idx] == 0.0:
                continue
            all_heads.append((layer_imp[head_idx].item(), layer_idx, head_idx))
            
    
    # Sort by importance (ascending - least important first)
    all_heads.sort(key=lambda x: x[0])
    
    # Mask the least important ones
    for i in range(num_to_mask):
        _, layer_idx, head_idx = all_heads[i]
        
        # Set mask to 0 (this head will be masked out)
        with torch.no_grad():
            model.bert.encoder.layer[layer_idx].head_mask_param[head_idx] = 0.0
        
        # if i < 5:  # Print first few
        #     print(f"  Masked layer {layer_idx}, head {head_idx} (importance: {all_heads[i][0]:.6f})")
    
    # if num_to_mask > 5:
    #     print(f"  ... and {num_to_mask - 5} more")
 
 
def apply_neuron_masking(model, neuron_importance, num_to_mask):
    """Apply masking to int_mask_param based on importance."""
    
    # Count total neurons
    total_neurons = sum(len(imp) for imp in neuron_importance)
    
    print(f"\nMasking {num_to_mask} / {total_neurons} neurons ({num_to_mask/total_neurons*100:.1f}%)")
    
    # Collect all (importance, layer_idx, neuron_idx) for ACTIVE neurons only
    all_neurons = []
    for layer_idx, layer_imp in enumerate(neuron_importance):
        for neuron_idx in range(len(layer_imp)):
            # SKIP if already masked
            if model.bert.encoder.layer[layer_idx].int_mask_param[neuron_idx] == 0.0:
                continue
            
            all_neurons.append((layer_imp[neuron_idx].item(), layer_idx, neuron_idx))
    
    # Sort by importance (ascending)
    all_neurons.sort(key=lambda x: x[0])
    
    # Mask the least important ones
    for i in range(num_to_mask):
        _, layer_idx, neuron_idx = all_neurons[i]
        
        # Set mask to 0
        with torch.no_grad():
            model.bert.encoder.layer[layer_idx].int_mask_param[neuron_idx] = 0.0
        
        # if i < 5:
        #     print(f"  Masked layer {layer_idx}, neuron {neuron_idx} (importance: {all_neurons[i][0]:.6f})")
    
    # if num_to_mask > 5:
    #     print(f"  ... and {num_to_mask - 5} more")

def compute_importance_scores(
    model,
    dataloader,
    device,
    max_batches: Optional[int] = None,
    compute_heads: bool = True,
    compute_ffn: bool = True
) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    """
    Compute importance scores for attention heads and/or FFN neurons.
    
    Flexible function that can compute:
    - Only head importance (compute_heads=True, compute_ffn=False)
    - Only FFN importance (compute_heads=False, compute_ffn=True)
    - Both together (compute_heads=True, compute_ffn=True) - single pass!
    
    Args:
        model: BERT model with registered masks
        dataloader: Dataloader for importance computation
        device: torch device
        max_batches: Limit batches (None = use all)
        compute_heads: Whether to compute head importance (default: True)
        compute_ffn: Whether to compute FFN neuron importance (default: True)
    
    Returns:
        (head_importance, neuron_importance):
            - head_importance: List of tensors or None if compute_heads=False
            - neuron_importance: List of tensors or None if compute_ffn=False
    
    Examples:
        # Compute both (single pass - most efficient)
        head_imp, neuron_imp = compute_importance_scores(
            model, dataloader, device,
            compute_heads=True, compute_ffn=True
        )
        
        # Compute only heads
        head_imp, _ = compute_importance_scores(
            model, dataloader, device,
            compute_heads=True, compute_ffn=False
        )
        
        # Compute only FFN
        _, neuron_imp = compute_importance_scores(
            model, dataloader, device,
            compute_heads=False, compute_ffn=True
        )
    """
    if not compute_heads and not compute_ffn:
        raise ValueError("Must compute at least one of: heads or FFN")
    
    model.eval()
    n_layers = len(model.bert.encoder.layer)
    
    # Check which masks are needed and registered
    if compute_heads:
        if model.bert.encoder.layer[0].head_mask_param is None:
            raise ValueError(
                "Head masks not registered! Call register_importance_masks(model, device) first."
            )
    
    if compute_ffn:
        if model.bert.encoder.layer[0].int_mask_param is None:
            raise ValueError(
                "FFN masks not registered! Call register_importance_masks(model, device) first."
            )
    
    # Initialize importance accumulators
    head_importance = None
    neuron_importance = None
    
    if compute_heads:
        head_importance = []
        for layer in model.bert.encoder.layer:
            head_importance.append(torch.zeros_like(layer.head_mask_param))
    
    if compute_ffn:
        neuron_importance = []
        for layer in model.bert.encoder.layer:
            neuron_importance.append(torch.zeros_like(layer.int_mask_param))
    
    # Accumulate gradients
    tot_tokens = 0.0
    n_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    
    # Progress bar message
    if compute_heads and compute_ffn:
        desc = "Computing importance (heads + FFN)"
    elif compute_heads:
        desc = "Computing importance (heads only)"
    else:
        desc = "Computing importance (FFN only)"
    
    for batch_idx, batch in enumerate(tqdm(dataloader, total=n_batches, desc=desc)):
        if max_batches and batch_idx >= max_batches:
            break
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass - your forward() will use the registered masks!
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Accumulate importance from gradients
        for layer_idx, layer in enumerate(model.bert.encoder.layer):
            if compute_heads and layer.head_mask_param.grad is not None:
                head_importance[layer_idx] += layer.head_mask_param.grad.abs().detach()
            
            if compute_ffn and layer.int_mask_param.grad is not None:
                neuron_importance[layer_idx] += layer.int_mask_param.grad.abs().detach()
        
        # Zero gradients
        model.zero_grad()
        
        tot_tokens += batch['attention_mask'].float().sum().item()
    
    # Normalize
    if compute_heads:
        for layer_idx in range(n_layers):
            head_importance[layer_idx] /= tot_tokens
    
    if compute_ffn:
        for layer_idx in range(n_layers):
            neuron_importance[layer_idx] /= tot_tokens
    
    return head_importance, neuron_importance


def remaining_layers_stat(model, component='head'):
    stats = []
    for idx, layer in enumerate(model.bert.encoder.layer):
        if component == 'head':
            out_list = layer.head_mask_param.detach().cpu().tolist()
        else:
            out_list = layer.int_mask_param.detach().cpu().tolist()

        obj = {
            'layer': idx,
            'n_remain': int(sum(out_list)),
            'total': len(out_list)
        }
        stats.append(obj)

    return stats

def iterativeMask(
    model, dataloader, val_dataloader, device, task_name, max_batches=None, mask_head=True, mask_ffn=True, 
    num_att_mask=0, num_int_mask=0, num_round=5
):
    # Calculate total number of heads and ffn neurons
    total_heads, total_neurons = 0, 0
    for layer in model.bert.encoder.layer:
        if mask_head:
            total_heads   += len(layer.head_mask_param)
        if mask_ffn:
            total_neurons += len(layer.int_mask_param)

    if mask_head and mask_ffn:
        print(f"Total Number of Heads: {total_heads}, Total Number of FFN Neurons: {total_neurons}")
    elif mask_head:
        print(f"Total Number of Heads: {total_heads}")
    elif mask_ffn:
        print(f"Total Number of FFN Neurons: {total_neurons}")
    else:
        return "No masking operation"
    
    target_metric = get_primary_metric(task_name)
    
    # Original Performance
    print("Evaluating Original Performance")
    orig_metric = eval_loop(model, val_dataloader, task_name, device)[0]
    print("  Metrics:")
    for metric_name, value in orig_metric.items():
        marker = "★" if metric_name == target_metric else " "
        print(f"    {marker} {metric_name}: {value:.4f}")

    mask_history = {
        'head_mask_remaining': [],
        'head_mask_state': [],
        'int_mask_remaining': [],
        'int_mask_state': [],
        'effected_metric': []
    }
    for round_idx  in range(num_round):
        print("")
        print("--> Round:", round_idx )
        # Calculate Importance Score
        head_importance, neuron_importance = compute_importance_scores(
            model, dataloader, device, max_batches=max_batches, compute_heads=mask_head,
            compute_ffn= mask_ffn
        )

        # Perform Masking
        if mask_head:
            apply_head_masking(model, head_importance, num_att_mask)
            
            remaining_heads = 0
            total_heads     = 0
            for layer in model.bert.encoder.layer:
                remaining_heads += int(sum(layer.head_mask_param))
                total_heads += len(layer.head_mask_param)
            print(f"Remaining Heads: {remaining_heads}, Total Heads: {total_heads}")
            
            head_mask_stats = remaining_layers_stat(model, component='head')
            head_mask_stats_list = []
            print("Head Mask Remaining Stats:")
            for stat in head_mask_stats:
                remain = stat['n_remain'] * 100 / stat['total']
                remain = round(remain, 2)
                print(f"Layer: {stat['layer']}, total: {stat['total']}, Remain: {stat['n_remain']} ({remain}%)")
                stat['mask'] = model.bert.encoder.layer[stat['layer']].head_mask_param.detach().cpu().tolist()
                head_mask_stats_list.append(stat['mask'])
                
            mask_history['head_mask_remaining'].append(remaining_heads)
            mask_history['head_mask_state'].append(head_mask_stats_list)
            
        if mask_ffn:
            apply_neuron_masking(model, neuron_importance, num_int_mask)
            
            remaining_neurons = 0
            total_neurons     = 0
            for layer in model.bert.encoder.layer:
                remaining_neurons += int(sum(layer.int_mask_param))
                total_neurons += len(layer.int_mask_param)
            print(f"Remaining FFN Neurons: {remaining_neurons}, Total FFN Neurons: {total_neurons}")

            int_mask_stats = remaining_layers_stat(model, component='ffn')
            int_mask_stats_list = []
            print("FFN Neuron Mask Remaining Stats:")
            for stat in int_mask_stats:
                remain = stat['n_remain'] * 100 / stat['total']
                remain = round(remain, 2)
                print(f"Layer: {stat['layer']}, total: {stat['total']}, Remain: {stat['n_remain']} ({remain}%)")
                stat['mask'] = model.bert.encoder.layer[stat['layer']].int_mask_param.detach().cpu().tolist()
                int_mask_stats_list.append(stat['mask'])
                
            mask_history['int_mask_remaining'].append(remaining_neurons)
            mask_history['int_mask_state'].append(int_mask_stats_list)
        
        # Effected Performance
        eval_metric = eval_loop(model, val_dataloader, task_name, device)[0]
        print("  Metrics:")
        for metric_name, value in eval_metric.items():
            marker = "★" if metric_name == target_metric else " "
            print(f"    {marker} {metric_name}: {value:.4f}")
        mask_history['effected_metric'].append(eval_metric)

    return mask_history

def plot_results(
    results_dict,
    metric='f1',
    component='auto',
    figsize=(10, 6),
    save_path=None,
    show_plot=True,
    title=None,
    baseline_marker=True
):
    """
    Plot masking results - plots heads or neurons based on component parameter.
    
    Simplified interface - just pass your results and metric!
    
    Args:
        results_dict: Dictionary with masking results
            Format: {
                'head_mask_remaining': [130, 116, 102, ...],
                'int_mask_remaining': [30000, 25000, ...],
                'effected_metric': [
                    {'accuracy': 0.86, 'f1': 0.909},
                    {'accuracy': 0.85, 'f1': 0.905},
                    ...
                ]
            }
        metric: Metric to plot ('f1', 'accuracy', etc.)
        component: 'auto' (default), 'heads', or 'neurons'
            - 'auto': automatically detects from data
            - 'heads': plot heads (even if both present)
            - 'neurons': plot neurons (even if both present)
        figsize: Figure size (width, height)
        save_path: Path to save figure (None = don't save)
        show_plot: Whether to display plot
        title: Custom title (None = auto-generate)
        baseline_marker: Whether to mark baseline (first point)
    
    Returns:
        matplotlib figure object
    
    Examples:
        # Auto-detect (default)
        plot_results(mask_history, metric='f1')
        
        # Explicitly plot heads
        plot_results(mask_history, metric='f1', component='heads')
        
        # Explicitly plot neurons
        plot_results(mask_history, metric='f1', component='neurons')
        
        # With custom title and save
        plot_results(mask_history, metric='f1', component='heads',
                    title='12-layer Model: Head Masking',
                    save_path='head_masking.png')
    """
    # Determine component
    if component == 'auto':
        # Auto-detect component based on data
        has_heads = bool(results_dict.get('head_mask_remaining', []))
        has_neurons = bool(results_dict.get('int_mask_remaining', []))
        
        if has_heads and has_neurons:
            # Both present - default to heads
            component = 'heads'
            print("Note: Both heads and neurons data found. Plotting heads by default.")
            print("      To plot neurons, use: plot_results(..., component='neurons')")
        elif has_heads:
            component = 'heads'
        elif has_neurons:
            component = 'neurons'
        else:
            raise ValueError("No masking data found. Need 'head_mask_remaining' or 'int_mask_remaining'")
    elif component not in ['heads', 'neurons', 'head', 'neuron', 'ffn']:
        raise ValueError(f"component must be 'auto', 'heads', or 'neurons', got: {component}")
    
    # Normalize component name
    if component in ['heads', 'head', 'attention']:
        component = 'heads'
    elif component in ['neurons', 'neuron', 'ffn', 'int']:
        component = 'neurons'
    
    # Determine component name and data
    if component == 'heads':
        remaining = results_dict.get('head_mask_remaining', [])
        component_name = 'Heads'
        if not remaining:
            raise ValueError("No 'head_mask_remaining' data found in results_dict")
    else:  # neurons
        remaining = results_dict.get('int_mask_remaining', [])
        component_name = 'FFN Neurons'
        if not remaining:
            raise ValueError("No 'int_mask_remaining' data found in results_dict")
    
    total_initial = remaining[0] if remaining else 0
    
    # Extract metrics
    metrics_data = results_dict.get('effected_metric', [])
    if not metrics_data:
        raise ValueError("No 'effected_metric' data in results_dict")
    
    metric_values = [m.get(metric, None) for m in metrics_data]
    
    if None in metric_values:
        raise ValueError(f"Metric '{metric}' not found in all data points")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate percentage remaining
    pct_remaining = [r / total_initial * 100 for r in remaining]
    
    # Plot main line
    ax.plot(remaining, metric_values, 'o-', linewidth=2.5, markersize=8,
            color='#2E86AB', label=f'{metric.upper()}', markeredgewidth=1.5,
            markeredgecolor='white')
    
    # Mark baseline if requested
    if baseline_marker and len(remaining) > 0:
        ax.plot(remaining[0], metric_values[0], 'o', markersize=12,
                color='#06A77D', markeredgewidth=2, markeredgecolor='white',
                label='Baseline', zorder=5)
    
    # Add value labels on points
    for i, (r, v) in enumerate(zip(remaining, metric_values)):
        ax.text(r, v, f'{v:.3f}', fontsize=9, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray', alpha=0.7))
    
    # Add percentage labels on x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(remaining)
    ax2.set_xticklabels([f'{p:.0f}%' for p in pct_remaining], fontsize=9)
    ax2.set_xlabel('Percentage Remaining', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_xlabel(f'{component_name} Remaining (Absolute Count)', 
                  fontsize=11, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
    
    if title is None:
        title = f'{component_name} Masking: {metric.upper()} vs Remaining {component_name}'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    
    # Set y-axis limits
    y_min = min(metric_values) - 0.02
    y_max = max(metric_values) + 0.02
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Plot saved to {save_path}')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig

def plot_masking_results(
    results_dict,
    metric='f1',
    component='heads',
    figsize=(10, 6),
    save_path=None,
    show_plot=True,
    title=None,
    baseline_marker=True
):
    """
    Plot masking results showing component remaining vs performance metric.
    
    Args:
        results_dict: Dictionary with masking results
            Format: {
                'head_mask_remaining': [130, 116, 102, ...],
                'int_mask_remaining': [30000, 25000, ...],
                'effected_metric': [
                    {'accuracy': 0.86, 'f1': 0.909},
                    {'accuracy': 0.85, 'f1': 0.905},
                    ...
                ]
            }
        metric: Metric to plot ('f1', 'accuracy', etc.)
        component: 'heads' or 'neurons'
        figsize: Figure size (width, height)
        save_path: Path to save figure (None = don't save)
        show_plot: Whether to display plot
        title: Custom title (None = auto-generate)
        baseline_marker: Whether to mark baseline (first point)
    
    Returns:
        matplotlib figure object
    
    Examples:
        # Plot heads remaining vs F1
        plot_masking_results(results, metric='f1', component='heads')
        
        # Plot neurons remaining vs accuracy
        plot_masking_results(results, metric='accuracy', component='neurons')
        
        # Compare multiple models
        fig, ax = plt.subplots(figsize=(12, 6))
        for model_name, results in all_results.items():
            plot_masking_results(
                results, metric='f1', component='heads',
                show_plot=False, ax=ax, label=model_name
            )
        ax.legend()
        plt.show()
    """
    # Extract data based on component
    if component.lower() in ['heads', 'head', 'attention']:
        remaining = results_dict.get('head_mask_remaining', [])
        component_name = 'Heads'
        total_initial = remaining[0] if remaining else 0
    elif component.lower() in ['neurons', 'neuron', 'ffn', 'int']:
        remaining = results_dict.get('int_mask_remaining', [])
        component_name = 'FFN Neurons'
        total_initial = remaining[0] if remaining else 0
    else:
        raise ValueError(f"Unknown component: {component}. Use 'heads' or 'neurons'")
    
    # Check if data exists
    if not remaining:
        raise ValueError(f"No data for {component_name}. Check 'head_mask_remaining' or 'int_mask_remaining' in results.")
    
    # Extract metrics
    metrics_data = results_dict.get('effected_metric', [])
    if not metrics_data:
        raise ValueError("No 'effected_metric' data in results_dict")
    
    # Extract the specific metric
    metric_values = [m.get(metric, None) for m in metrics_data]
    
    # Check for None values
    if None in metric_values:
        raise ValueError(f"Metric '{metric}' not found in all data points")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate percentage remaining
    pct_remaining = [r / total_initial * 100 for r in remaining]
    
    # Plot main line
    ax.plot(remaining, metric_values, 'o-', linewidth=2.5, markersize=8,
            color='#2E86AB', label=f'{metric.upper()}', markeredgewidth=1.5,
            markeredgecolor='white')
    
    # Mark baseline if requested
    if baseline_marker and len(remaining) > 0:
        ax.plot(remaining[0], metric_values[0], 'o', markersize=12,
                color='#06A77D', markeredgewidth=2, markeredgecolor='white',
                label='Baseline', zorder=5)
    
    # Add value labels on points
    for i, (r, v) in enumerate(zip(remaining, metric_values)):
        ax.text(r, v, f'{v:.3f}', fontsize=9, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray', alpha=0.7))
    
    # Add percentage labels on x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(remaining)
    ax2.set_xticklabels([f'{p:.0f}%' for p in pct_remaining], fontsize=9)
    ax2.set_xlabel('Percentage Remaining', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_xlabel(f'{component_name} Remaining (Absolute Count)', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
    
    if title is None:
        title = f'{component_name} Masking: {metric.upper()} vs Remaining {component_name}'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    
    # Set y-axis to start from a reasonable minimum
    y_min = min(metric_values) - 0.02
    y_max = max(metric_values) + 0.02
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Plot saved to {save_path}')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig

def plot_masking_comparison(
    results_dict_list,
    labels,
    metric='f1',
    component='heads',
    figsize=(12, 7),
    save_path=None,
    show_plot=True,
    title=None,
    show_percentage=True,
    show_values=False,
    legend_loc='best'
):
    """
    Plot comparison of multiple models' masking results.
    
    Similar to plot_masking_results but for multiple models on same plot.
    
    Args:
        results_dict_list: List of result dictionaries
        labels: List of labels for each result (same length as results_dict_list)
        metric: Metric to plot ('f1', 'accuracy', etc.)
        component: 'heads' or 'neurons'
        figsize: Figure size (width, height)
        save_path: Path to save figure (None = don't save)
        show_plot: Whether to display plot
        title: Custom title (None = auto-generate)
        show_percentage: Whether to show percentage on top x-axis
        show_values: Whether to show value labels on points
        legend_loc: Legend location ('best', 'upper right', 'lower left', etc.)
    
    Returns:
        matplotlib figure
    
    Examples:
        # Compare 12-layer vs 8-layer
        plot_masking_comparison(
            [results_12layer, results_8layer],
            labels=['12-layer Baseline', '8-layer Merged'],
            metric='f1',
            component='heads',
            save_path='comparison.png'
        )
        
        # Compare all models with custom styling
        plot_masking_comparison(
            [results_12, results_8, results_7, results_6],
            labels=['12-layer', '8-layer', '7-layer', '6-layer'],
            metric='f1',
            component='heads',
            show_values=True,
            legend_loc='lower left'
        )
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors and markers for different models (supports up to 8 models)
    colors = ['#2E86AB', '#06A77D', '#F18F01', '#C73E1D', '#6A4C93', 
              '#E63946', '#457B9D', '#2A9D8F']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    
    # Determine component
    if component.lower() in ['heads', 'head', 'attention']:
        key = 'head_mask_remaining'
        component_name = 'Heads'
    else:
        key = 'int_mask_remaining'
        component_name = 'FFN Neurons'
    
    # Track min/max for axis limits
    all_remaining = []
    all_values = []
    
    # Plot each model
    for i, (results, label) in enumerate(zip(results_dict_list, labels)):
        remaining = results.get(key, [])
        metrics_data = results.get('effected_metric', [])
        metric_values = [m.get(metric, 0) for m in metrics_data]
        
        if not remaining:
            print(f"Warning: No data for '{label}'")
            continue
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Track for axis limits
        all_remaining.extend(remaining)
        all_values.extend(metric_values)
        
        # Plot line
        ax.plot(remaining, metric_values, marker=marker, linestyle='-',
                linewidth=2.5, markersize=8, color=color, label=label,
                markeredgewidth=1.5, markeredgecolor='white', alpha=0.9)
        
        # Mark baseline (first point)
        ax.plot(remaining[0], metric_values[0], marker=marker, markersize=12,
                color=color, markeredgewidth=2, markeredgecolor='white',
                zorder=5, alpha=0.7)
        
        # Add value labels if requested
        if show_values:
            for r, v in zip(remaining, metric_values):
                ax.text(r, v, f'{v:.3f}', fontsize=8, ha='center', va='bottom',
                       color=color, fontweight='bold', alpha=0.8)
    
    # Add percentage axis on top if requested
    if show_percentage and all_remaining:
        total_initial = max(all_remaining)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        
        # Create tick positions
        tick_positions = sorted(set(all_remaining))
        tick_labels = [f'{r/total_initial*100:.0f}%' for r in tick_positions]
        
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, fontsize=9)
        ax2.set_xlabel('Percentage Remaining', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_xlabel(f'{component_name} Remaining (Absolute Count)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
    
    if title is None:
        title = f'{component_name} Masking Comparison: {metric.upper()}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc=legend_loc, framealpha=0.95, 
             edgecolor='gray', fancybox=True)
    
    # Set reasonable y-axis limits
    if all_values:
        y_min = min(all_values) - 0.02
        y_max = max(all_values) + 0.02
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Plot saved to {save_path}')
    
    if show_plot:
        plt.show()
    
    return fig