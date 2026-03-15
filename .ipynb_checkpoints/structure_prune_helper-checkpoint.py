import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Tuple, List
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
        
        if i < 5:  # Print first few
            print(f"  Masked layer {layer_idx}, head {head_idx} (importance: {all_heads[i][0]:.6f})")
    
    if num_to_mask > 5:
        print(f"  ... and {num_to_mask - 5} more")
 
 
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
        
        if i < 5:
            print(f"  Masked layer {layer_idx}, neuron {neuron_idx} (importance: {all_neurons[i][0]:.6f})")
    
    if num_to_mask > 5:
        print(f"  ... and {num_to_mask - 5} more")

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

def iterativeMask(
    model, dataloader, val_dataloader, device, task_name, max_batches=None, mask_head=True, mask_ffn=True, 
    num_att_mask=0, num_int_mask=0, num_round=5
):
    # Calculate total number of heads and ffn neurons
    total_heads, total_neurons = 0, 0
    for layer in model.bert.encoder.layer:
        total_heads   += len(layer.head_mask_param)
        total_neurons += len(layer.int_mask_param)

    print(f"Total Number of Heads: {total_heads}, Total Number of FFN Neurons: {total_neurons}")
    
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
        'int_mask_remaining': [],
        'effected_metric': []
    }
    for round in range(num_round):
        print("")
        print("Round:", round)
        # Calculate Importance Score
        head_importance, neuron_importance = compute_importance_scores(
            model, dataloader, device, max_batches=max_batches, compute_heads=mask_head,
            compute_ffn= mask_ffn
        )

        # Perform Masking
        if mask_head:
            apply_head_masking(model, head_importance, num_att_mask)
            remaining_heads = 0
            for layer in model.bert.encoder.layer:
                remaining_heads += int(sum(layer.head_mask_param))
            mask_history['head_mask_remaining'].append(remaining_heads)
            print("Remaining Heads:", remaining_heads)
        if mask_ffn:
            apply_neuron_masking(model, neuron_importance, num_int_mask)
            remaining_neurons = 0
            for layer in model.bert.encoder.layer:
                remaining_neurons += int(sum(layer.int_mask_param))
            mask_history['int_mask_remaining'].append(remaining_neurons)
            print("Remaining FFN Neurons:", remaining_neurons)
        
        # Effected Performance
        eval_metric = eval_loop(model, val_dataloader, task_name, device)[0]
        print("  Metrics:")
        for metric_name, value in eval_metric.items():
            marker = "★" if metric_name == target_metric else " "
            print(f"    {marker} {metric_name}: {value:.4f}")
        mask_history['effected_metric'].append(eval_metric)

    return mask_history