import torch

#### Ordering Functions

def reorder_merged_layer_heads(merged_layer, head_importance, device):
    """
    Reorder heads in merged layer so most important are first.
    
    Args:
        merged_layer   : BertLayer with merged heads
        head_importance: (num_heads,) tensor of importance scores for the merged layer
                         Must be recomputed AFTER merging, not from pre-merge scores
        device         : torch device
    
    Returns:
        merged_layer with heads reordered descending by importance
    """
    n_heads   = merged_layer.attention.self.num_attention_heads
    head_size = merged_layer.attention.self.attention_head_size

    head_importance = head_importance.to(device)

    assert head_importance.shape[0] == n_heads, (
        f"head_importance length {head_importance.shape[0]} "
        f"doesn't match num_heads {n_heads}. "
        f"Recompute importance after merging."
    )

    # sorted indices descending — most important head first
    sorted_head_indices = torch.argsort(head_importance, descending=True)

    # flat index for weight reordering
    index = (
        torch.arange(n_heads * head_size, device=device)
             .view(n_heads, head_size)[sorted_head_indices]
             .view(-1)
             .contiguous()
    )

    def reorder_linear(linear_layer, index, dim):
        with torch.no_grad():
            W = linear_layer.weight.index_select(dim, index).clone()
            if linear_layer.bias is not None and dim == 0:
                b = linear_layer.bias[index].clone()
            elif linear_layer.bias is not None:
                b = linear_layer.bias.clone()  # W_O bias unchanged

            linear_layer.weight.requires_grad = False
            linear_layer.weight.copy_(W.contiguous())
            linear_layer.weight.requires_grad = True

            if linear_layer.bias is not None:
                linear_layer.bias.requires_grad = False
                linear_layer.bias.copy_(b.contiguous())
                linear_layer.bias.requires_grad = True

    # Q, K, V — row-wise (output dim is per-head)
    reorder_linear(merged_layer.attention.self.query, index, dim=0)
    reorder_linear(merged_layer.attention.self.key,   index, dim=0)
    reorder_linear(merged_layer.attention.self.value, index, dim=0)

    # W_O — column-wise (input dim is per-head)
    reorder_linear(merged_layer.attention.output.dense, index, dim=1)

    # reorder head_mask_param to match new head order
    if hasattr(merged_layer, 'head_mask_param') and merged_layer.head_mask_param is not None:
        merged_layer.head_mask_param = nn.Parameter(
            merged_layer.head_mask_param.data[sorted_head_indices].clone(),
            requires_grad=True
        )

    print(f"Reordered {n_heads} heads, new order: {sorted_head_indices.tolist()}")
    return merged_layer

def reorder_layer_neurons(layer, neuron_importance, device):
    """
    Reorder FFN neurons in ANY BertLayer so most important are first.
    Works for both original and merged (variable width) layers.
    
    Args:
        layer            : BertLayer object (merged or original)
        neuron_importance: (intermediate_size,) tensor — must match layer's current FFN width
        device           : torch device
    
    Returns:
        layer with neurons reordered descending by importance
    """
    if layer.intermediate is None:
        print("  Layer FFN is pruned — skipping reorder")
        return layer

    ffn_dim = layer.intermediate.dense.out_features

    neuron_importance = neuron_importance.to(device)

    assert neuron_importance.shape[0] == ffn_dim, (
        f"neuron_importance length {neuron_importance.shape[0]} "
        f"doesn't match ffn_dim {ffn_dim}. "
        f"Recompute importance after merging."
    )

    sorted_neuron_indices = torch.argsort(neuron_importance, descending=True)

    def reorder_linear(linear_layer, index, dim):
        with torch.no_grad():
            W = linear_layer.weight.index_select(dim, index).clone()
            if linear_layer.bias is not None and dim == 0:
                b = linear_layer.bias[index].clone()
            elif linear_layer.bias is not None:
                b = linear_layer.bias.clone()  # output bias unchanged

            linear_layer.weight.requires_grad = False
            linear_layer.weight.copy_(W.contiguous())
            linear_layer.weight.requires_grad = True

            if linear_layer.bias is not None:
                linear_layer.bias.requires_grad = False
                linear_layer.bias.copy_(b.contiguous())
                linear_layer.bias.requires_grad = True

    # W1 — row-wise (output dim is per-neuron)
    reorder_linear(layer.intermediate.dense, sorted_neuron_indices, dim=0)

    # W2 — column-wise (input dim is per-neuron)
    reorder_linear(layer.output.dense, sorted_neuron_indices, dim=1)

    # reorder int_mask_param to match new neuron order
    if hasattr(layer, 'int_mask_param') and layer.int_mask_param is not None:
        layer.int_mask_param = nn.Parameter(
            layer.int_mask_param.data[sorted_neuron_indices].clone(),
            requires_grad=True
        )

    print(f"  Reordered {ffn_dim} neurons")
    return layer

#### Layer Composition 

def merge_high_sim_layer_composition(layer_composition, sublayer_imp, reps_similarity, layer_scores, similarity_score_threshold=0.9):
    while len(layer_composition) > 6:
        # find high similarity pairs
        high_sim_pairs = []
        for idx in range(0, len(layer_composition) - 1):
            layer1_idx = min(layer_composition[idx])
            layer2_idx = max(layer_composition[idx + 1])
            # print(layer1_idx, layer2_idx)
            
            similarity_score = reps_similarity[layer1_idx][layer2_idx]
            if similarity_score > similarity_score_threshold:
                high_sim_pairs.append((layer1_idx, layer2_idx))
        # print(high_sim_pairs)
        if len(high_sim_pairs) == 0:
            break
    
        # find weakest pair
        weakest_pair_idx = 0
        pair1, pair2 = high_sim_pairs[weakest_pair_idx]
        weakest_score = (layer_scores[pair1] + layer_scores[pair2]) / 2
        for idx, pair in enumerate(high_sim_pairs):
            pair1, pair2 = pair
            pair_score = (layer_scores[pair1] + layer_scores[pair2]) / 2
            if pair_score < weakest_score:
                weakest_pair_idx = idx
                weakest_score = pair_score
        # print("Weakest pair:", high_sim_pairs[weakest_pair_idx])
        # print("Idx in List:", weakest_pair_idx)
        # print("Score:", weakest_score)
    
        # identify in indices in layer_composition for merging
        pair1, pair2 = high_sim_pairs[weakest_pair_idx]
        for idx, layers in enumerate(layer_composition):
            if pair1 in layers:
                idx1 = idx
            elif pair2 in layers:
                idx2 = idx
    
        layer_composition[idx1] = layer_composition[idx1] + layer_composition[idx2]
        layer_composition[idx1].sort()
        del layer_composition[idx2]
    
        # update merged layer score
        highest_mha_score = 0
        highest_mlp_score  = 0
        for layer in layer_composition[idx1]:
            # print(layer)
            sublayer_idx = layer * 2
            # print(sublayer_imp[sublayer_idx], sublayer_imp[sublayer_idx+1])
        
            if sublayer_imp[sublayer_idx] > highest_mha_score:
                highest_mha_score = sublayer_imp[sublayer_idx]
        
            if sublayer_imp[sublayer_idx+1] > highest_mlp_score:
                highest_mlp_score = sublayer_imp[sublayer_idx+1]
    
        layer_score = (highest_mha_score + highest_mlp_score) / 2
        for layer in layer_composition[idx1]:
            layer_scores[layer] = layer_score

    return layer_composition

        
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

def merge_mha_topk(model, sourceLayer, targetLayer,
              source_head_imp_sorted,
              target_head_imp_sorted,
              device,
              k_heads=0):

    s_n_head = sourceLayer.attention.self.num_attention_heads
    t_n_head = targetLayer.attention.self.num_attention_heads
    head_size = sourceLayer.attention.self.attention_head_size

    # ---------------------------
    # Target stays intact
    # Source contributes top-k heads
    # ---------------------------
    s_idx = min(k_heads, s_n_head)
    t_idx = t_n_head

    # Final number of heads
    n_head = t_idx + s_idx
    new_dim = n_head * head_size

    print(f"Taking {s_idx} source heads")
    print(f"Keeping {t_idx} target heads")

    # Convert head count → dimension
    s_len = s_idx * head_size
    t_len = t_idx * head_size

    # ---------------------------
    # Resize layers FIRST
    # ---------------------------
    targetLayer.attention.self.query = resize_qkv(
        targetLayer.attention.self.query,
        new_dim
    )

    targetLayer.attention.self.key = resize_qkv(
        targetLayer.attention.self.key,
        new_dim
    )

    targetLayer.attention.self.value = resize_qkv(
        targetLayer.attention.self.value,
        new_dim
    )

    targetLayer.attention.output.dense = resize_output(
        targetLayer.attention.output.dense,
        new_dim
    )

    # ---------------------------
    # Merge weights
    # Target first, source appended
    # ---------------------------
    cw_helper(
        sourceLayer.attention.self.query,
        targetLayer.attention.self.query,
        s_len,
        t_len
    )

    cw_helper(
        sourceLayer.attention.self.key,
        targetLayer.attention.self.key,
        s_len,
        t_len
    )

    cw_helper(
        sourceLayer.attention.self.value,
        targetLayer.attention.self.value,
        s_len,
        t_len
    )

    cw_helper(
        sourceLayer.attention.output.dense,
        targetLayer.attention.output.dense,
        s_len,
        t_len,
        dim=1
    )

    # ---------------------------
    # Update config
    # ---------------------------
    targetLayer.attention.self.num_attention_heads = n_head
    targetLayer.attention.self.all_head_size = new_dim

    # ---------------------------
    # Update head mask
    # ---------------------------
    targetLayer.head_mask_param = nn.Parameter(
        torch.ones(
            n_head,
            dtype=torch.float32,
            device=device
        ),
        requires_grad=True
    )

    return targetLayer

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

def merge_ff_topk(model,
             sourceLayer,
             targetLayer,
             source_int_imp_sorted,
             target_int_imp_sorted,
             device,
             k_neurons=0):

    s_dim = sourceLayer.intermediate.dense.out_features
    t_dim = targetLayer.intermediate.dense.out_features

    # ---------------------------
    # Target stays intact
    # Source contributes top-k neurons
    # ---------------------------
    s_idx = min(k_neurons, s_dim)
    t_idx = t_dim

    # Final FFN dimension
    f_dim = t_idx + s_idx

    print(f"Taking {s_idx} source neurons")
    print(f"Keeping {t_idx} target neurons")

    s_len = s_idx
    t_len = t_idx

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
    # Target first, source appended
    # ---------------------------
    cw_ff_helper(
        sourceLayer.intermediate.dense,
        targetLayer.intermediate.dense,
        s_len,
        t_len
    )

    cw_ff_helper(
        sourceLayer.output.dense,
        targetLayer.output.dense,
        s_len,
        t_len,
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
        torch.ones(
            f_dim,
            dtype=torch.float32,
            device=device
        ),
        requires_grad=True
    )

    return targetLayer

def merge_mha_cluster(model, layers, target_idx, k_per_layer, device):
    assert len(layers) == len(k_per_layer), "layers and k_per_layer must be same length"
    assert 0 <= target_idx < len(layers), "target_idx out of range"

    targetLayer = layers[target_idx]
    head_size   = targetLayer.attention.self.attention_head_size

    # clamp each k to the number of heads actually available in that layer
    k_per_layer_clamped = []
    for k, layer in zip(k_per_layer, layers):
        avail = layer.attention.self.num_attention_heads
        k_clamped = min(k, avail)
        if k_clamped != k:
            print(f"  WARNING: requested k={k} but layer only has {avail} heads, clamped to {k_clamped}")
        k_per_layer_clamped.append(k_clamped)
    k_per_layer = k_per_layer_clamped

    n_head  = sum(k_per_layer)
    new_dim = n_head * head_size

    print(f"Merging cluster of {len(layers)} layers into target (idx={target_idx})")
    print(f"  k_per_layer (clamped): {k_per_layer}")
    print(f"  n_head total: {n_head}, new_dim: {new_dim}")

    for i, (layer, k) in enumerate(zip(layers, k_per_layer)):
        tag = " (target)" if i == target_idx else ""
        print(f"  Layer{tag}: taking {k} heads, head_size={layer.attention.self.attention_head_size}")

    # sanity check — all layers must have the SAME head_size
    head_sizes = set(layer.attention.self.attention_head_size for layer in layers)
    if len(head_sizes) > 1:
        raise ValueError(f"All layers must have the same head_size, got {head_sizes}")

    t_keep_len = k_per_layer[target_idx] * head_size

    orig_q   = targetLayer.attention.self.query.weight.data[:t_keep_len].clone()
    orig_q_b = targetLayer.attention.self.query.bias.data[:t_keep_len].clone()
    orig_k   = targetLayer.attention.self.key.weight.data[:t_keep_len].clone()
    orig_k_b = targetLayer.attention.self.key.bias.data[:t_keep_len].clone()
    orig_v   = targetLayer.attention.self.value.weight.data[:t_keep_len].clone()
    orig_v_b = targetLayer.attention.self.value.bias.data[:t_keep_len].clone()
    orig_o   = targetLayer.attention.output.dense.weight.data[:, :t_keep_len].clone()

    targetLayer.attention.self.query   = resize_qkv(targetLayer.attention.self.query, new_dim)
    targetLayer.attention.self.key     = resize_qkv(targetLayer.attention.self.key, new_dim)
    targetLayer.attention.self.value   = resize_qkv(targetLayer.attention.self.value, new_dim)
    targetLayer.attention.output.dense = resize_output(targetLayer.attention.output.dense, new_dim)

    with torch.no_grad():
        offset = 0

        for i, (layer, k) in enumerate(zip(layers, k_per_layer)):
            write_len = k * head_size

            # guard: catch overflow BEFORE it crashes with a confusing message
            if offset + write_len > new_dim:
                raise RuntimeError(
                    f"Offset overflow at layer {i}: offset={offset}, write_len={write_len}, "
                    f"new_dim={new_dim}. Sum of k_per_layer*head_size exceeds new_dim — "
                    f"check k_per_layer values."
                )

            if i == target_idx:
                targetLayer.attention.self.query.weight.data[offset:offset+write_len] = orig_q
                targetLayer.attention.self.query.bias.data[offset:offset+write_len]   = orig_q_b
                targetLayer.attention.self.key.weight.data[offset:offset+write_len]   = orig_k
                targetLayer.attention.self.key.bias.data[offset:offset+write_len]     = orig_k_b
                targetLayer.attention.self.value.weight.data[offset:offset+write_len] = orig_v
                targetLayer.attention.self.value.bias.data[offset:offset+write_len]   = orig_v_b
                targetLayer.attention.output.dense.weight.data[:, offset:offset+write_len] = orig_o
            else:
                # guard: source layer must actually have write_len rows available
                avail_rows = layer.attention.self.query.weight.data.shape[0]
                if write_len > avail_rows:
                    raise RuntimeError(
                        f"Layer {i}: trying to write {write_len} rows but source only has "
                        f"{avail_rows} rows (num_heads={layer.attention.self.num_attention_heads})"
                    )

                targetLayer.attention.self.query.weight.data[offset:offset+write_len] = \
                    layer.attention.self.query.weight.data[:write_len]
                targetLayer.attention.self.query.bias.data[offset:offset+write_len] = \
                    layer.attention.self.query.bias.data[:write_len]

                targetLayer.attention.self.key.weight.data[offset:offset+write_len] = \
                    layer.attention.self.key.weight.data[:write_len]
                targetLayer.attention.self.key.bias.data[offset:offset+write_len] = \
                    layer.attention.self.key.bias.data[:write_len]

                targetLayer.attention.self.value.weight.data[offset:offset+write_len] = \
                    layer.attention.self.value.weight.data[:write_len]
                targetLayer.attention.self.value.bias.data[offset:offset+write_len] = \
                    layer.attention.self.value.bias.data[:write_len]

                targetLayer.attention.output.dense.weight.data[:, offset:offset+write_len] = \
                    layer.attention.output.dense.weight.data[:, :write_len]

            print(f"    Wrote {k} heads from layer {i}{' (target)' if i==target_idx else ''} "
                  f"into slots [{offset}:{offset+write_len}]")
            offset += write_len

    targetLayer.attention.self.num_attention_heads = n_head
    targetLayer.attention.self.all_head_size        = new_dim

    targetLayer.head_mask_param = nn.Parameter(
        torch.ones(n_head, dtype=torch.float32, device=device),
        requires_grad=True
    )

    print(f"  Done: target now has {n_head} heads (dim={new_dim})")
    return targetLayer


import copy

def test_merge_mha_cluster(model, layer_indices, target_idx, k_per_layer, device):
    """
    Verifies merge_mha_cluster correctness by checking that:
    1. Target's own (kept) weights are preserved exactly
    2. Each source layer's top-k weights are copied exactly into the right offset
    3. Final shapes match expectations
    4. Non-participating layers in the model are untouched
    """
    # work on a deep copy so we don't corrupt the real model
    test_model = copy.deepcopy(model)

    layers = [test_model.bert.encoder.layer[idx] for idx in layer_indices]
    head_size = layers[0].attention.self.attention_head_size

    # --- snapshot BEFORE merge ---
    before = {}
    for i, layer in enumerate(layers):
        before[i] = {
            'q_w': layer.attention.self.query.weight.data.clone(),
            'q_b': layer.attention.self.query.bias.data.clone(),
            'k_w': layer.attention.self.key.weight.data.clone(),
            'k_b': layer.attention.self.key.bias.data.clone(),
            'v_w': layer.attention.self.value.weight.data.clone(),
            'v_b': layer.attention.self.value.bias.data.clone(),
            'o_w': layer.attention.output.dense.weight.data.clone(),
            'num_heads': layer.attention.self.num_attention_heads,
        }

    # snapshot an UNRELATED layer to confirm it's untouched
    untouched_idx = next(
        i for i in range(len(test_model.bert.encoder.layer))
        if i not in layer_indices
    )
    untouched_before = test_model.bert.encoder.layer[untouched_idx].attention.self.query.weight.data.clone()

    # --- run merge ---
    merged_layer = merge_mha_cluster(test_model, layers, target_idx, k_per_layer, device)

    # --- clamp k_per_layer same way the function does, for verification ---
    k_clamped = [min(k, before[i]['num_heads']) for i, k in enumerate(k_per_layer)]

    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    all_passed = True

    # --- check 1: final shape ---
    n_head_expected = sum(k_clamped)
    actual_n_head = merged_layer.attention.self.num_attention_heads
    shape_ok = (actual_n_head == n_head_expected)
    print(f"[{'PASS' if shape_ok else 'FAIL'}] num_heads: expected={n_head_expected}, actual={actual_n_head}")
    all_passed &= shape_ok

    expected_dim = n_head_expected * head_size
    actual_dim = merged_layer.attention.self.query.weight.shape[0]
    dim_ok = (actual_dim == expected_dim)
    print(f"[{'PASS' if dim_ok else 'FAIL'}] query out_features: expected={expected_dim}, actual={actual_dim}")
    all_passed &= dim_ok

    # --- check 2: each segment matches source exactly ---
    offset = 0
    for i, k in enumerate(k_clamped):
        write_len = k * head_size
        segment_q = merged_layer.attention.self.query.weight.data[offset:offset+write_len]
        expected_q = before[i]['q_w'][:write_len]

        match = torch.allclose(segment_q, expected_q, atol=1e-7)
        tag = " (target)" if i == target_idx else ""
        print(f"[{'PASS' if match else 'FAIL'}] Layer {i}{tag} Q weights "
              f"[{offset}:{offset+write_len}] match source")
        all_passed &= match

        # also check K, V, bias for thoroughness
        segment_k = merged_layer.attention.self.key.weight.data[offset:offset+write_len]
        expected_k = before[i]['k_w'][:write_len]
        match_k = torch.allclose(segment_k, expected_k, atol=1e-7)
        print(f"[{'PASS' if match_k else 'FAIL'}] Layer {i}{tag} K weights match")
        all_passed &= match_k

        segment_v = merged_layer.attention.self.value.weight.data[offset:offset+write_len]
        expected_v = before[i]['v_w'][:write_len]
        match_v = torch.allclose(segment_v, expected_v, atol=1e-7)
        print(f"[{'PASS' if match_v else 'FAIL'}] Layer {i}{tag} V weights match")
        all_passed &= match_v

        segment_qb = merged_layer.attention.self.query.bias.data[offset:offset+write_len]
        expected_qb = before[i]['q_b'][:write_len]
        match_qb = torch.allclose(segment_qb, expected_qb, atol=1e-7)
        print(f"[{'PASS' if match_qb else 'FAIL'}] Layer {i}{tag} Q bias match")
        all_passed &= match_qb

        segment_o = merged_layer.attention.output.dense.weight.data[:, offset:offset+write_len]
        expected_o = before[i]['o_w'][:, :write_len]
        match_o = torch.allclose(segment_o, expected_o, atol=1e-7)
        print(f"[{'PASS' if match_o else 'FAIL'}] Layer {i}{tag} W_O columns match")
        all_passed &= match_o

        offset += write_len

    # --- check 3: head_mask_param shape ---
    mask_ok = (merged_layer.head_mask_param.shape[0] == n_head_expected)
    print(f"[{'PASS' if mask_ok else 'FAIL'}] head_mask_param shape: "
          f"expected={n_head_expected}, actual={merged_layer.head_mask_param.shape[0]}")
    all_passed &= mask_ok

    # --- check 4: unrelated layer untouched ---
    untouched_after = test_model.bert.encoder.layer[untouched_idx].attention.self.query.weight.data
    untouched_ok = torch.allclose(untouched_before, untouched_after)
    print(f"[{'PASS' if untouched_ok else 'FAIL'}] Unrelated layer {untouched_idx} unchanged")
    all_passed &= untouched_ok

    # --- check 5: non-target source layers unchanged (they're pruned separately, not modified here) ---
    for i, idx in enumerate(layer_indices):
        if i == target_idx:
            continue
        current_layer = test_model.bert.encoder.layer[idx]
        if current_layer.attention is not None:  # not yet pruned
            src_unchanged = torch.allclose(
                current_layer.attention.self.query.weight.data,
                before[i]['q_w']
            )
            print(f"[{'PASS' if src_unchanged else 'FAIL'}] Source layer {idx} "
                  f"itself unmodified (only target was written to)")
            all_passed &= src_unchanged

    print("="*60)
    print(f"OVERALL: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("="*60)

    return all_passed