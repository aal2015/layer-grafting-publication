import torch
import torch.nn as nn
from copy import deepcopy

def merge_bert_layers(model, layer1_idx, layer2_idx, 
                             ln_strategy='use_deeper',
                             cka_similarity=None):
    """
    Merge two BERT layers by concatenating internal dimensions.
    
    Args:
        model: BERT model
        layer1_idx: Index of first layer to merge
        layer2_idx: Index of second layer to merge
        ln_strategy: How to handle LayerNorm parameters
            'use_deeper'    : Use layer2's LN (recommended default)
            'use_shallower' : Use layer1's LN
            'split'         : Keep both LNs, apply separately (most principled)
            'average'       : Average LN params (problematic, baseline)
            'weighted'      : Weight by CKA similarity (requires cka_similarity arg)
        cka_similarity: Float in [0,1], only used if ln_strategy='weighted'
    
    Returns:
        merged_layer: Single BERT layer with 2× internal width
    """
    layer1 = model.bert.encoder.layer[layer1_idx]
    layer2 = model.bert.encoder.layer[layer2_idx]
    merged_layer = deepcopy(layer1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Helper: concatenate two linear layers along output dimension
    # ─────────────────────────────────────────────────────────────────────────
    def concat_linear_layers(linear1, linear2):
        new_weight = torch.cat([linear1.weight, linear2.weight], dim=0)
        new_bias = torch.cat([linear1.bias, linear2.bias], dim=0) if linear1.bias is not None else None
        
        new_linear = nn.Linear(
            linear1.in_features,
            linear1.out_features + linear2.out_features,
            bias=(linear1.bias is not None)
        ).to(linear1.weight.device)
        
        new_linear.weight.data = new_weight
        if new_bias is not None:
            new_linear.bias.data = new_bias
        
        return new_linear
    
    # ─────────────────────────────────────────────────────────────────────────
    # Merge attention Q, K, V (expand output dimension)
    # ─────────────────────────────────────────────────────────────────────────
    merged_layer.attention.self.query = concat_linear_layers(
        layer1.attention.self.query, layer2.attention.self.query
    )
    merged_layer.attention.self.key = concat_linear_layers(
        layer1.attention.self.key, layer2.attention.self.key
    )
    merged_layer.attention.self.value = concat_linear_layers(
        layer1.attention.self.value, layer2.attention.self.value
    )
    
    # Update attention head configuration
    merged_layer.attention.self.num_attention_heads = (
        layer1.attention.self.num_attention_heads + 
        layer2.attention.self.num_attention_heads
    )
    merged_layer.attention.self.all_head_size = (
        layer1.attention.self.all_head_size + 
        layer2.attention.self.all_head_size
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Merge attention output dense (takes 2× input, outputs original size)
    # ─────────────────────────────────────────────────────────────────────────
    merged_output_weight = torch.cat([
        layer1.attention.output.dense.weight,
        layer2.attention.output.dense.weight
    ], dim=1)
    
    merged_layer.attention.output.dense = nn.Linear(
        merged_output_weight.shape[1],
        layer1.attention.output.dense.out_features,
        bias=(layer1.attention.output.dense.bias is not None)
    ).to(merged_output_weight.device)
    
    merged_layer.attention.output.dense.weight.data = merged_output_weight
    
    # Bias: average (both contribute to same output space, so averaging is OK here)
    if layer1.attention.output.dense.bias is not None:
        merged_layer.attention.output.dense.bias.data = (
            layer1.attention.output.dense.bias + layer2.attention.output.dense.bias
        ) / 2
    
    # ─────────────────────────────────────────────────────────────────────────
    # CRITICAL: Merge attention output LayerNorm
    # ─────────────────────────────────────────────────────────────────────────
    if ln_strategy == 'use_deeper':
        # Use the deeper layer's LN (layer2) - RECOMMENDED
        merged_layer.attention.output.LayerNorm = deepcopy(layer2.attention.output.LayerNorm)
    
    elif ln_strategy == 'use_shallower':
        # Use the shallower layer's LN (layer1)
        merged_layer.attention.output.LayerNorm = deepcopy(layer1.attention.output.LayerNorm)
    
    # elif ln_strategy == 'split':
    #     # Most principled: keep both LNs, apply separately to each half
    #     merged_layer.attention.output.LayerNorm = SplitLayerNormOutput(
    #         layer1.attention.output.LayerNorm,
    #         layer2.attention.output.LayerNorm,
    #         split_dim=768
    #     )
    
    elif ln_strategy == 'average':
        # PROBLEMATIC: Average the LN params (your original code)
        # Kept here for ablation comparison only
        merged_layer.attention.output.LayerNorm.weight.data = (
            layer1.attention.output.LayerNorm.weight + 
            layer2.attention.output.LayerNorm.weight
        ) / 2
        merged_layer.attention.output.LayerNorm.bias.data = (
            layer1.attention.output.LayerNorm.bias + 
            layer2.attention.output.LayerNorm.bias
        ) / 2
    
    elif ln_strategy == 'weighted':
        # Weight LN params by CKA similarity
        assert cka_similarity is not None, "Must provide cka_similarity for weighted strategy"
        w1 = 0.5 + (1 - cka_similarity) * 0.3  # Higher CKA = more equal weights
        w2 = 1.0 - w1
        
        merged_layer.attention.output.LayerNorm.weight.data = (
            w1 * layer1.attention.output.LayerNorm.weight + 
            w2 * layer2.attention.output.LayerNorm.weight
        )
        merged_layer.attention.output.LayerNorm.bias.data = (
            w1 * layer1.attention.output.LayerNorm.bias + 
            w2 * layer2.attention.output.LayerNorm.bias
        )
    
    else:
        raise ValueError(f"Unknown ln_strategy: {ln_strategy}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Merge FFN intermediate (expand output dimension)
    # ─────────────────────────────────────────────────────────────────────────
    merged_layer.intermediate.dense = concat_linear_layers(
        layer1.intermediate.dense, layer2.intermediate.dense
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Merge FFN output dense (takes 2× input, outputs original size)
    # ─────────────────────────────────────────────────────────────────────────
    merged_ffn_output_weight = torch.cat([
        layer1.output.dense.weight,
        layer2.output.dense.weight
    ], dim=1)
    
    merged_layer.output.dense = nn.Linear(
        merged_ffn_output_weight.shape[1],
        layer1.output.dense.out_features,
        bias=(layer1.output.dense.bias is not None)
    ).to(merged_ffn_output_weight.device)
    
    merged_layer.output.dense.weight.data = merged_ffn_output_weight
    
    if layer1.output.dense.bias is not None:
        merged_layer.output.dense.bias.data = (
            layer1.output.dense.bias + layer2.output.dense.bias
        ) / 2
    
    # ─────────────────────────────────────────────────────────────────────────
    # CRITICAL: Merge FFN output LayerNorm (same strategy as attention LN)
    # ─────────────────────────────────────────────────────────────────────────
    if ln_strategy == 'use_deeper':
        merged_layer.output.LayerNorm = deepcopy(layer2.output.LayerNorm)
    
    elif ln_strategy == 'use_shallower':
        merged_layer.output.LayerNorm = deepcopy(layer1.output.LayerNorm)
    
    elif ln_strategy == 'split':
        merged_layer.output.LayerNorm = SplitLayerNormOutput(
            layer1.output.LayerNorm,
            layer2.output.LayerNorm,
            split_dim=768
        )
    
    elif ln_strategy == 'average':
        merged_layer.output.LayerNorm.weight.data = (
            layer1.output.LayerNorm.weight + 
            layer2.output.LayerNorm.weight
        ) / 2
        merged_layer.output.LayerNorm.bias.data = (
            layer1.output.LayerNorm.bias + 
            layer2.output.LayerNorm.bias
        ) / 2
    
    elif ln_strategy == 'weighted':
        w1 = 0.5 + (1 - cka_similarity) * 0.3
        w2 = 1.0 - w1
        merged_layer.output.LayerNorm.weight.data = (
            w1 * layer1.output.LayerNorm.weight + 
            w2 * layer2.output.LayerNorm.weight
        )
        merged_layer.output.LayerNorm.bias.data = (
            w1 * layer1.output.LayerNorm.bias + 
            w2 * layer2.output.LayerNorm.bias
        )
    
    return merged_layer


class LayerMergeTracker:
    """Track which original layers are merged together."""
    
    def __init__(self, num_layers=12):
        # Start: [[0], [1], [2], ..., [11]]
        self.layer_groups = [[i] for i in range(num_layers)]
    
    def merge(self, idx):
        """Merge position idx with idx+1."""
        # Combine the two groups
        self.layer_groups[idx] = self.layer_groups[idx] + self.layer_groups[idx + 1]
        # Delete idx+1
        del self.layer_groups[idx + 1]
    
    def get_mapping(self):
        """Return current layer groups."""
        return self.layer_groups
    
    def __len__(self):
        return len(self.layer_groups)


def apply_layer_tracking_to_model(model, layer_track):
    """
    Apply a layer tracking configuration to a model.
    
    This is the function you should use in your notebook.
    
    Args:
        model: Fresh copy of the original 12-layer BERT model
        layer_track: Layer tracking list, e.g., [[0], [1,2], [3], [4,5,6], ...]
    
    Returns:
        Model with merged layers according to layer_track
    
    Usage in your notebook:
        # Load the saved tracking
        checkpoint = torch.load('bert-mrpc-tracking.pt')
        layer_track = checkpoint['layer_track']
        
        # Create fresh model
        graft_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
            ...
        )
        
        # Apply the merge pattern
        graft_model = apply_layer_tracking_to_model(graft_model, layer_track)
        
        # Load the trained weights
        graft_model.load_state_dict(checkpoint['model_state_dict'])
    """
    # Save original layers
    original_layers = [deepcopy(layer) for layer in model.bert.encoder.layer]
    
    new_layers = []
    
    for group in layer_track:
        if len(group) == 1:
            # No merge - use original layer
            idx = group[0]
            new_layers.append(original_layers[idx])
        
        else:
            # Need to merge - start with first layer
            # Create a small temporary model for merging
            temp_model = deepcopy(model)
            temp_model.bert.encoder.layer = nn.ModuleList([
                deepcopy(original_layers[group[0]])
            ])
            
            # Iteratively merge in remaining layers
            for i in range(1, len(group)):
                next_layer = deepcopy(original_layers[group[i]])
                temp_model.bert.encoder.layer.append(next_layer)
                
                # Merge the last two layers (indices len-2 and len-1)
                last_idx = len(temp_model.bert.encoder.layer) - 2
                merged = merge_bert_layers(temp_model, last_idx, last_idx + 1)
                
                # Replace with merged layer
                temp_model.bert.encoder.layer = nn.ModuleList(
                    list(temp_model.bert.encoder.layer[:last_idx]) + [merged]
                )
            
            new_layers.append(temp_model.bert.encoder.layer[0])
    
    # Replace model's layers
    model.bert.encoder.layer = nn.ModuleList(new_layers)
    
    return model