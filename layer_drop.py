import torch

from train_eval_func import get_primary_metric, set_lr_scheduler, eval_loop
import os
from layer_merge_helper import merge_mha, merge_ff

from torch.optim import AdamW
from train_eval_func import train
import torch.nn as nn

from structure_prune_helper import compute_importance_scores

def layer_drop(
    model,
    train_dataloader,
    val_dataloader,
    task_name,
    device,
    init_metric,
    cka_evaluator=None,
    num_merges=6,
    target_layers=[3, 4, 5, 6, 7],
    recovery_epochs=10,
    recovery_lr=1e-5,
    patience=2,
    save_dir='./weights/',
    keep_temp_checkpoints=False,
    cka_max_iter=float("Inf"),
    teacher_model=None,
    alpha=0.5,
    temperature=6,
    drop_strategy="top",
    regression=False
):
    recovery_dir = save_dir
    save_dir = save_dir + drop_strategy+ "/"
    
    orig_n_layer = len(model.bert.encoder.layer)
    target_metric = get_primary_metric(task_name)
    threshold = init_metric[target_metric] * 0.01

    # Create recovery checkpoint directory
    recovery_dir = os.path.join(recovery_dir, 'recovery')
    os.makedirs(recovery_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Target Metric for {task_name}: {target_metric}")
    print(f"Original {target_metric} score: {init_metric[target_metric]:.4f}")
    print(f"Recovery threshold: {threshold:.4f}")
    print("")

    performance_track = []
    remaining_layers = []
    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    performance_track.append(init_metric)
    remaining_layers.append(layers)
    for merge_itr in range(num_merges):
        print("=" * 70)
        print(f"Merge Iteration: {merge_itr + 1}/{num_merges}")
        print("=" * 70)

        # Drop Layer
        if drop_strategy == "top":
            model.bert.encoder.layer = model.bert.encoder.layer[:-1]
            layers = layers[:-1]
            
        elif drop_strategy == "top_merge":
            merge_layer = len(model.bert.encoder.layer) - 2
            print("Drop Layer:", merge_layer)

            head_importance, neuron_importance = compute_importance_scores(model, train_dataloader, device, compute_heads=True, compute_ffn=True)

            head_imp1, indices1 = torch.sort(head_importance[merge_layer], descending=True)
            head_imp2, indices2 = torch.sort(head_importance[merge_layer+1], descending=True)
    
            # neuron_imp1, indices1 = torch.sort(neuron_importance[merge_layer], descending=True)        
            # neuron_imp2, indices1 = torch.sort(neuron_importance[merge_layer+1], descending=True)
            
            # merge_mha(model, model.bert.encoder.layer[merge_layer], model.bert.encoder.layer[merge_layer+1], head_imp1, head_imp2, device, 4)
            # merge_ff(model, model.bert.encoder.layer[merge_layer], model.bert.encoder.layer[merge_layer+1], neuron_imp1, neuron_imp2, device, extra_neurons=1024)
            merge_mha(model, model.bert.encoder.layer[merge_layer], model.bert.encoder.layer[merge_layer+1], head_imp1, head_imp2, device, 4)
            # merge_ff(model, model.bert.encoder.layer[merge_layer], model.bert.encoder.layer[merge_layer+1], neuron_imp1, neuron_imp2, device, extra_neurons=0)
            del model.bert.encoder.layer[merge_layer]
        elif drop_strategy == "contribution":
            cls_reps_similarity = cka_evaluator.pairwise(
                model, 
                train_dataloader, 
                device, 
                only_cls_token=True, 
                max_iter=cka_max_iter
            )
            stats = cka_evaluator.similarity_stats(cls_reps_similarity)
            
            print("  Similarity Stats:")
            print(f"    Average: {stats['average']:.6f}")
            print(f"    Highest: {stats['max']:.6f}")
            print(f"    Lowest:  {stats['min']:.6f}")
        
            merge_layer = stats['argmax']
            print("Drop Layer:", merge_layer)
        
            model.bert.encoder.layer = nn.ModuleList(
                [layer for i, layer in enumerate(model.bert.encoder.layer) if i != merge_layer]
            )
            model.config.num_hidden_layers = len(model.bert.encoder.layer)

            layers.pop(merge_layer)
        elif drop_strategy == "contribution_merge":
            cls_reps_similarity = cka_evaluator.pairwise(
                model, 
                train_dataloader, 
                device, 
                only_cls_token=True, 
                max_iter=cka_max_iter
            )
            stats = cka_evaluator.similarity_stats(cls_reps_similarity)
            
            print("  Similarity Stats:")
            print(f"    Average: {stats['average']:.6f}")
            print(f"    Highest: {stats['max']:.6f}")
            print(f"    Lowest:  {stats['min']:.6f}")
        
            merge_layer = stats['argmax']
            print("Drop Layer:", merge_layer)

            head_importance, neuron_importance = compute_importance_scores(model, train_dataloader, device, compute_heads=True, compute_ffn=True)

            head_imp1, indices1 = torch.sort(head_importance[merge_layer], descending=True)
            head_imp2, indices2 = torch.sort(head_importance[merge_layer+1], descending=True)
    
            neuron_imp1, indices1 = torch.sort(neuron_importance[merge_layer], descending=True)        
            neuron_imp2, indices1 = torch.sort(neuron_importance[merge_layer+1], descending=True)
            
            merge_mha(model, model.bert.encoder.layer[merge_layer], model.bert.encoder.layer[merge_layer+1], head_imp1, head_imp2, device, 4)
            merge_ff(model, model.bert.encoder.layer[merge_layer], model.bert.encoder.layer[merge_layer+1], neuron_imp1, neuron_imp2, device, extra_neurons=1024)
    
            del model.bert.encoder.layer[merge_layer]
        elif drop_strategy == "alt_merge":
            merge_layers = [10, 8, 6, 4, 2, 0]
            merge_layer = merge_layers[merge_itr]
            print("Drop Layer:", merge_layer)

            head_importance, neuron_importance = compute_importance_scores(model, train_dataloader, device, compute_heads=True, compute_ffn=True)

            head_imp1, indices1 = torch.sort(head_importance[merge_layer], descending=True)
            head_imp2, indices2 = torch.sort(head_importance[merge_layer+1], descending=True)
    
            neuron_imp1, indices1 = torch.sort(neuron_importance[merge_layer], descending=True)        
            neuron_imp2, indices1 = torch.sort(neuron_importance[merge_layer+1], descending=True)
            
            merge_mha(model, model.bert.encoder.layer[merge_layer], model.bert.encoder.layer[merge_layer+1], head_imp1, head_imp2, device, 4)
            merge_ff(model, model.bert.encoder.layer[merge_layer], model.bert.encoder.layer[merge_layer+1], neuron_imp1, neuron_imp2, device, extra_neurons=1024)
    
            del model.bert.encoder.layer[merge_layer]
        else:
            return "Specify drop strategy"

        print("Number of Layers Remaining:", len(model.bert.encoder.layer))

        # Post Drop Performance 
        eval_metric = eval_loop(model, val_dataloader, task_name, device, regression=regression)[0]
        
        print("  Metrics:")
        for metric_name, value in eval_metric.items():
            marker = "★" if metric_name == target_metric else " "
            print(f"    {marker} {metric_name}: {value:.4f}")
        
        # Recovery Training
        diff = init_metric[target_metric] - eval_metric[target_metric]
        print(f"  Performance drop: {diff:.4f} (threshold: {threshold:.4f})")

        is_retrained = False
        if diff > threshold:
            print("  → Recovery training NEEDED")

            # Setup optimizer and scheduler
            optimizer = AdamW(model.parameters(), lr=recovery_lr)
            num_training_steps = recovery_epochs * len(train_dataloader)
            lr_scheduler = set_lr_scheduler(
                optimizer=optimizer,
                num_training_steps=num_training_steps
            )
            
            print(f"  → Training: {recovery_epochs} epochs, lr={recovery_lr}")
            
            # Temporary checkpoint path
            temp_save_path = os.path.join(recovery_dir, f'temp_iter{merge_itr}-{task_name}.pt')
            
            # Train with checkpoint saving
            # lr_scheduler= None
            train_stats = train(
                model,
                train_dataloader,
                val_dataloader,
                task_name,
                device,
                num_epochs=recovery_epochs,
                lr=recovery_lr,
                patience=patience,
                use_early_stopping=True,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                save_path=temp_save_path,
                display_epoch_iter=True,
                regression=regression,
                teacher_model=teacher_model, alpha=alpha, temperature=temperature
            )

             # CRITICAL: Reload best checkpoint
            print("\n  → Loading best checkpoint...")
            best_checkpoint = torch.load(temp_save_path)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            
            print(f"  ✓ Loaded best checkpoint from epoch {best_checkpoint['epoch']}")
            print(f"    Best val loss: {best_checkpoint['val_loss']:.4f}")
            print(f"    Best val metrics: {best_checkpoint['val_metrics']}")
            
            # merge_history['metrics_after_recovery'].append(best_checkpoint['val_metrics'])
            # merge_history['training_stats'].append(train_stats)
            
            # Clean up temp checkpoint if not keeping
            if not keep_temp_checkpoints:
                os.remove(temp_save_path)
                print(f"  ✓ Cleaned up temporary checkpoint")

            is_retrained = True
        else:
            print("  → Recovery training NOT needed")

        eval_metric = eval_loop(model, val_dataloader, task_name, device, regression=regression)[0]
        performance_track.append(eval_metric)
        remaining_layers.append(layers.copy())
        
        n_layer = orig_n_layer - merge_itr - 1
        if n_layer in target_layers:
            save_path = os.path.join(save_dir, f'layer-{n_layer}-{task_name}.pt')
            
            save_object = {
                'model_state_dict': model.state_dict(),
                # 'layer_track': tracker.get_mapping(),
                # 'train_stats': merge_history['training_stats'][-1],
                # 'metrics_after_merge': merge_history['metrics_after_merge'][-1],
                # 'metrics_after_recovery': merge_history['metrics_after_recovery'][-1],
                'merge_iteration': merge_itr + 1,
                'num_layers': n_layer,
                'is_retrained': is_retrained
            }
            
            torch.save(save_object, save_path)
            print(f"  ✓ Saved {n_layer}-layer model to {save_path}")
        else:
            print(f"  • {n_layer} layers (not in target list, skipping save)")
        
        ## Store iteration info
        # merge_history['iterations'].append({
        #     'iteration': merge_itr + 1,
        #     'merged_layers': (merge_layer, merge_layer + 1),
        #     'num_layers_after': n_layer
        # })
        # merge_history['layer_compositions'].append(tracker.get_mapping())

    return_obj = {
        "performance_track": performance_track,
        "remaining_layers": remaining_layers
    }

    return return_obj