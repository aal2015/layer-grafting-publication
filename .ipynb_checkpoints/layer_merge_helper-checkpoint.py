import torch

#### Ordering Functions

def reorder_merged_layer_heads(merged_layer, head_importance, device):
    """
    Reorder heads in merged layer so most important are first.
    """
    n_heads = merged_layer.attention.self.num_attention_heads
    head_size = merged_layer.attention.self.attention_head_size
    
    # Ensure head_importance is on the correct device
    head_importance = head_importance.to(device)
    
    # Get sorted indices (descending importance)
    sorted_head_indices = torch.argsort(head_importance, descending=True)
    
    # Create reordering index for weights [all_head_size]
    # Create directly on device to avoid transfer
    index = torch.arange(n_heads * head_size, device=device).view(n_heads, head_size)
    index = index[sorted_head_indices].view(-1).contiguous()
    
    def reorder_layer(linear_layer, index, dim):
        W = linear_layer.weight.index_select(dim, index).clone()
        
        if linear_layer.bias is not None:
            if dim == 1:
                b = linear_layer.bias.clone()
            else:
                b = linear_layer.bias[index].clone()
        
        linear_layer.weight.data = W.contiguous()
        if linear_layer.bias is not None:
            linear_layer.bias.data = b.contiguous()
    
    # Reorder Q, K, V (output dimension)
    reorder_layer(merged_layer.attention.self.query, index, dim=0)
    reorder_layer(merged_layer.attention.self.key, index, dim=0)
    reorder_layer(merged_layer.attention.self.value, index, dim=0)
    
    # Reorder output (input dimension)
    reorder_layer(merged_layer.attention.output.dense, index, dim=1)
    
    return merged_layer

def reorder_merged_layer_neurons(merged_layer, neuron_importance, device):
    """
    Reorder FFN neurons so most important are first.
    """
    # Ensure neuron_importance is on the correct device
    neuron_importance = neuron_importance.to(device)
    
    # Get sorted indices (descending importance)
    sorted_neuron_indices = torch.argsort(neuron_importance, descending=True)
    
    def reorder_layer(linear_layer, index, dim):
        W = linear_layer.weight.index_select(dim, index).clone()
        
        if linear_layer.bias is not None:
            if dim == 1:
                b = linear_layer.bias.clone()
            else:
                b = linear_layer.bias[index].clone()
        
        linear_layer.weight.data = W.contiguous()
        if linear_layer.bias is not None:
            linear_layer.bias.data = b.contiguous()
    
    # Reorder intermediate (output dimension)
    reorder_layer(merged_layer.intermediate.dense, sorted_neuron_indices, dim=0)
    
    # Reorder output (input dimension)
    reorder_layer(merged_layer.output.dense, sorted_neuron_indices, dim=1)
    
    return merged_layer

#### Merge Functions
import torch.nn as nn

def resize_qkv(linear_layer, new_out_dim):
    new_layer = nn.Linear(
        in_features=linear_layer.in_features,
        out_features=new_out_dim,
        bias=True
    ).to(linear_layer.weight.device)

    return new_layer

def resize_output(linear_layer, new_in_dim):
    new_layer = nn.Linear(
        in_features=new_in_dim,
        out_features=linear_layer.out_features,
        bias=True
    ).to(linear_layer.weight.device)

    return new_layer

def cw_helper(sourceLinearLayer, targetLinearLayer, s_len, t_len, dim=0):
    with torch.no_grad():

        if dim == 0:
            # Row-wise (Q, K, V)
            W_source = sourceLinearLayer.weight.data[:s_len]
            W_target = targetLinearLayer.weight.data[:t_len]

            b_source = sourceLinearLayer.bias.data[:s_len]
            b_target = targetLinearLayer.bias.data[:t_len]

            new_W = torch.cat([W_target, W_source], dim=0)
            new_b = torch.cat([b_target, b_source], dim=0)

            targetLinearLayer.weight.data[:t_len + s_len] = new_W
            targetLinearLayer.bias.data[:t_len + s_len] = new_b

        else:
            # Column-wise (output projection)
            W_source = sourceLinearLayer.weight.data[:, :s_len]
            W_target = targetLinearLayer.weight.data[:, :t_len]

            new_W = torch.cat([W_target, W_source], dim=1)

            targetLinearLayer.weight.data[:, :t_len + s_len] = new_W

            # Bias unchanged (correct for output layer)

def merge_mha(model, sourceLayer, targetLayer,
              source_head_imp_sorted, target_head_imp_sorted,
              device, extra_heads=0):

    s_n_head = sourceLayer.attention.self.num_attention_heads
    t_n_head = targetLayer.attention.self.num_attention_heads
    head_size = sourceLayer.attention.self.attention_head_size

    # Final number of heads
    n_head = max(s_n_head, t_n_head) + extra_heads
    new_dim = n_head * head_size

    # ---------------------------
    # Decide how many heads to take
    # ---------------------------
    s_idx, t_idx = 0, 0

    for _ in range(n_head):

        if s_idx >= s_n_head:
            t_idx += 1
            continue

        if t_idx >= t_n_head:
            s_idx += 1
            continue

        if source_head_imp_sorted[s_idx] > target_head_imp_sorted[t_idx]:
            s_idx += 1
        else:
            t_idx += 1

    print(s_idx, t_idx)

    # Convert head count → dimension
    s_len = s_idx * head_size
    t_len = new_dim - s_len

    # ---------------------------
    # Resize layers FIRST
    # ---------------------------
    targetLayer.attention.self.query = resize_qkv(
        targetLayer.attention.self.query, new_dim
    )
    targetLayer.attention.self.key = resize_qkv(
        targetLayer.attention.self.key, new_dim
    )
    targetLayer.attention.self.value = resize_qkv(
        targetLayer.attention.self.value, new_dim
    )

    targetLayer.attention.output.dense = resize_output(
        targetLayer.attention.output.dense, new_dim
    )

    # ---------------------------
    # Merge weights
    # ---------------------------
    cw_helper(sourceLayer.attention.self.query,
              targetLayer.attention.self.query,
              s_len, t_len)

    cw_helper(sourceLayer.attention.self.key,
              targetLayer.attention.self.key,
              s_len, t_len)

    cw_helper(sourceLayer.attention.self.value,
              targetLayer.attention.self.value,
              s_len, t_len)

    cw_helper(sourceLayer.attention.output.dense,
              targetLayer.attention.output.dense,
              s_len, t_len,
              dim=1)

    # ---------------------------
    # Update config
    # ---------------------------
    targetLayer.attention.self.num_attention_heads = n_head
    targetLayer.attention.self.all_head_size = new_dim

    # ---------------------------
    # Update head mask
    # ---------------------------
    targetLayer.head_mask_param = nn.Parameter(
        torch.ones(n_head, dtype=torch.float32, device=device),
        requires_grad=True
    )              

def cw_ff_helper(sourceLinearLayer, targetLinearLayer, s_len, t_len, dim=0):
    with torch.no_grad():

        if dim == 0:
            # Row-wise (intermediate layer)
            W_source = sourceLinearLayer.weight.data[:s_len]
            W_target = targetLinearLayer.weight.data[:t_len]

            b_source = sourceLinearLayer.bias.data[:s_len]
            b_target = targetLinearLayer.bias.data[:t_len]

            new_W = torch.cat([W_target, W_source], dim=0)
            new_b = torch.cat([b_target, b_source], dim=0)

            targetLinearLayer.weight.data[:t_len + s_len] = new_W
            targetLinearLayer.bias.data[:t_len + s_len] = new_b

        else:
            # Column-wise (output layer)
            W_source = sourceLinearLayer.weight.data[:, :s_len]
            W_target = targetLinearLayer.weight.data[:, :t_len]

            new_W = torch.cat([W_target, W_source], dim=1)

            targetLinearLayer.weight.data[:, :t_len + s_len] = new_W

            # Bias unchanged

def resize_ffn_intermediate(linear_layer, new_out_dim):
    return nn.Linear(
        linear_layer.in_features,
        new_out_dim,
        bias=True
    ).to(linear_layer.weight.device)


def resize_ffn_output(linear_layer, new_in_dim):
    return nn.Linear(
        new_in_dim,
        linear_layer.out_features,
        bias=True
    ).to(linear_layer.weight.device)

def merge_ff(model, sourceLayer, targetLayer,
             source_int_imp_sorted,
             target_int_imp_sorted,
             device,
             extra_neurons=0):

    s_dim = sourceLayer.intermediate.dense.out_features
    t_dim = targetLayer.intermediate.dense.out_features

    # Final FFN size
    f_dim = max(s_dim, t_dim) + extra_neurons

    s_idx, t_idx = 0, 0
                 
    # ---------------------------
    # Decide allocation
    # ---------------------------
    for _ in range(f_dim):

        if s_idx >= s_dim:
            t_idx += 1
            continue

        if t_idx >= t_dim:
            s_idx += 1
            continue

        if source_int_imp_sorted[s_idx] > target_int_imp_sorted[t_idx]:
            s_idx += 1
        else:
            t_idx += 1

    print(s_idx, t_idx)

    s_len = s_idx
    t_len = f_dim - s_len

    # ---------------------------
    # Resize layers FIRST
    # ---------------------------
    targetLayer.intermediate.dense = resize_ffn_intermediate(
        targetLayer.intermediate.dense,
        f_dim
    )

    targetLayer.output.dense = resize_ffn_output(
        targetLayer.output.dense,
        f_dim
    )

    # ---------------------------
    # Merge weights
    # ---------------------------
    cw_ff_helper(
        sourceLayer.intermediate.dense,
        targetLayer.intermediate.dense,
        s_len, t_len
    )

    cw_ff_helper(
        sourceLayer.output.dense,
        targetLayer.output.dense,
        s_len, t_len,
        dim=1
    )

    # ---------------------------
    # Update config
    # ---------------------------
    targetLayer.intermediate.dense.out_features = f_dim

    # ---------------------------
    # Update FFN mask
    # ---------------------------
    targetLayer.int_mask_param = nn.Parameter(
        torch.ones(f_dim, dtype=torch.float32, device=device),
        requires_grad=True
    )