####################### Loading Model #######################

from transformers import get_scheduler

def set_lr_scheduler(optimizer, num_training_steps, name = "linear", num_warmup_steps =  0):
    lr_scheduler = get_scheduler(
        name,
        optimizer          = optimizer,
        num_warmup_steps   = num_warmup_steps,
        num_training_steps = num_training_steps,
    )
    return lr_scheduler

####################### Training Loop #######################

class EarlyStopping:
    def __init__(self, patience=3, threshold=0.0, mode="max"):
        """
        mode: "max" for accuracy/F1, "min" for loss
        """
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.best_score = None
        self.patience_count = 0

    def _is_improvement(self, score):
        if self.mode == "max":
            return score > self.best_score + self.threshold
        else:  # mode == "min"
            return score < self.best_score - self.threshold

    def check(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self._is_improvement(score):
            self.best_score = score
            self.patience_count = 0
            return False
        else:
            self.patience_count += 1
            return self.patience_count >= self.patience

import evaluate
import torch

# def eval_loop(model, dataloader, dataset, device, benchmark="glue"):
#     metric = evaluate.load(benchmark, dataset)
#     model.eval()
#     for batch in dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch)

#         model_loss = outputs.loss
#         logits     = outputs.logits
#         predictions = torch.argmax(logits, dim=-1)
#         metric.add_batch(predictions=predictions, references=batch["labels"])
        
#     return metric.compute(), float(model_loss)

def eval_loop(model, dataloader, dataset, device, benchmark="glue", regression=False):
    metric = evaluate.load(benchmark, dataset)
    model.eval()
    
    total_loss = 0.0  # ← Accumulate
    n_batches = 0     # ← Count 
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        loss = outputs.loss
        logits = outputs.logits

        if regression:
            # STS-B
            predictions = logits.squeeze(-1)
        else:
            # MRPC, QNLI, CoLA, SST-2
            predictions = torch.argmax(logits, dim=-1)
        
        metric.add_batch(predictions=predictions, references=batch["labels"])
        total_loss += float(loss)  # ← Add to total
        n_batches += 1
    
    avg_loss = total_loss / n_batches  # ← Compute average
    return metric.compute(), avg_loss   # ← Return average

from torch.optim import AdamW

import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import time

def get_primary_metric(task_name):
    """
    Return the primary metric name for a GLUE task.
    This is used for model selection (which metric to optimize).
    """
    primary_metrics = {
        'mrpc': 'f1',
        'qqp': 'f1',
        'qnli': 'accuracy',
        'sst2': 'accuracy',
        'cola': 'matthews_correlation',
        'stsb': 'pearson',
        'mnli': 'accuracy',
        'rte': 'accuracy',
        'wnli': 'accuracy',
    }
    return primary_metrics.get(task_name.lower(), 'accuracy')

def train(model, train_dataloader, val_dataloader, task_name, device, num_epochs=5, 
          lr=3e-5, patience=3, threshold=0.0, save_path=None, optimizer=None, 
          lr_scheduler=None, use_early_stopping=False, display_epoch_iter=False,
          max_norm=0.05, benchmark="glue", regression=False, teacher_model=None, alpha=0.5, temperature=6):
    """
    Train BERT model with optional early stopping and checkpoint saving.
    Automatically handles all GLUE task metrics.
    
    Args:
        model: BERT model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        task_name: GLUE task name (e.g., "mrpc", "sst2", "cola")
        device: torch device
        num_epochs: Maximum number of epochs (default: 5)
        lr: Learning rate (default: 3e-5)
        patience: Early stopping patience (default: 3)
        threshold: Early stopping threshold (default: 0.1)
        save_path: Full path to save checkpoint (e.g., './weights/bert-mrpc.pt'). 
                   If None, no saving (default: None)
        optimizer: Optional pre-initialized optimizer (default: None, creates AdamW)
        lr_scheduler: Optional learning rate scheduler (default: None)
        use_early_stopping: Boolean to enable early stopping (default: False)
        display_epoch_iter: Boolean to display epoch iternation
        max_norm: max norm of the gradients for torch gradient clipping
        benchmark: Benchmark name (default: "glue")
        regression: is task classification or regression
    
    Returns:
        Dictionary with training history
    """
    # Setup optimizer if not provided
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=lr)
    
    use_scheduler = lr_scheduler is not None
    
    # Setup tracking
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps), desc="Training")
    n_batches = len(train_dataloader)

    # Early stopping (if use_early_stopping is True)
    if use_early_stopping:    
        early_stopping = EarlyStopping(patience=patience, threshold=threshold)
    
    # History tracking
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    time_per_epoch_hist = []
    
    # Best model tracking (for optional saving)
    primary_metric = get_primary_metric(task_name)
    best_metric_score = float('-inf')
    
    # Training loop
    train_start = time.time()

    if teacher_model is not None:
        print("Knowledge distillation used!")
        teacher_model.eval()
        
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Initialize metric for this task
        train_metric = evaluate.load(benchmark, task_name)
        model.train()
        
        total_loss = 0
        
        # Training iteration
        for b, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            # Knowledge Distillation
            if teacher_model is not None:
                loss_function = nn.KLDivLoss(reduction="batchmean")
                with torch.no_grad():
                    outputs_teacher = teacher_model(**batch)
                assert outputs.logits.size() == outputs_teacher.logits.size()
                loss_logits = (loss_function(
                    F.log_softmax(outputs.logits / temperature, dim=-1),
                    F.softmax(outputs_teacher.logits / temperature, dim=-1)) * (temperature ** 2))
                loss = alpha * loss + (1. - alpha) * loss_logits
            
            # Compute predictions
            if regression:
                predictions = logits.squeeze(-1)
            else:
                predictions = torch.argmax(logits, dim=-1)
                
            train_metric.add_batch(
                predictions=predictions.detach().cpu(),
                references=batch["labels"].detach().cpu()
            )
            
            # Backward pass
            total_loss += float(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            if use_scheduler:
                lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        # Compute training metrics
        avg_train_loss = total_loss / n_batches
        train_metrics = train_metric.compute()
        
        # Validation
        val_metrics, val_loss = eval_loop(
            model, val_dataloader, task_name, device, benchmark, regression=regression
        )
        
        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        epoch_time = time.time() - epoch_start
        time_per_epoch_hist.append(epoch_time)
        
        # Save checkpoint if path provided and loss improved
        current_metric_score = val_metrics[primary_metric]
        if save_path is not None and current_metric_score > best_metric_score:
            best_metric_score = current_metric_score
            
            save_object = {
                'epoch': epoch + 1,
                'batch': b + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'train_metrics': train_metrics,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }
            
            if use_scheduler:
                save_object['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            
            torch.save(save_object, save_path)
            print(f"✓ Saved checkpoint (best loss: {val_loss:.4f})")
        
        # Display metrics
        if display_epoch_iter:
            print(f"\n<----------------- Epoch {epoch + 1} ----------------->")
        print(f"Loss: {round(avg_train_loss, 2)}, Training Metrics:")
        for metric_name, value in train_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        print(f"Validation Loss: {round(val_loss, 2)}, Validation Metrics:")
        for metric_name, value in val_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        print(f"Elapsed Time: {round(epoch_time, 4)} sec")
        
        # Early stopping check (if use_early_stopping is True)
        if use_early_stopping:
            if early_stopping.check(val_metrics[primary_metric ]):
                print(">>>>> Early stopping callback <<<<<")
                break
    
    total_train_time = time.time() - train_start
    print(f"\nTotal Training Time: {total_train_time:.2f} sec")
    if save_path is not None:
        print(f"Chosen Primary Metric: {primary_metric}")
        print(f"Best Metric Run: {best_metric_score:.4f}")
    
    # Return training history
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics_history': train_metrics_history,
        'val_metrics_history': val_metrics_history,
        'time_per_epoch': time_per_epoch_hist,
        'total_time': total_train_time,
        'best_metric': primary_metric,
        'best_metric_score': best_metric_score,
        'num_epochs_trained': len(train_losses)
    }

import os
from merge_helper import merge_bert_layers

def get_recovery_epoch(task_name):
    small_datasets = ['mrpc', 'cola', 'stsb', 'rte', 'wnli']
    large_datasets = ['qqp', 'qnli', 'sst2', 'mnli']

    if task_name in small_datasets:
        return 3
    elif task_name in large_datasets:
        return 1

def iterative_cka_merge_and_train(
    model,
    train_dataloader,
    val_dataloader,
    task_name,
    device,
    cka_evaluator,
    tracker,
    init_metric,
    num_merges=6,
    target_layers=[6, 7, 8],
recovery_epochs=3,
    recovery_lr=1e-5,
    patience=2,
    save_dir='./weights/',
    keep_temp_checkpoints=False,
    cka_max_iter=float("Inf"),
    teacher_model=None,
    alpha=0.5,
    temperature=6
):
    """
    Iteratively merge BERT layers using CKA similarity and recover performance.
    
    Args:
        model: BERT model to merge
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        task_name: GLUE task name (e.g., "mrpc", "qnli", "sst2")
        device: torch device
        cka_evaluator: CKAEvaluator instance
        tracker: LayerMergeTracker instance
        init_metric: Initial metrics dict before any merging
        num_merges: Number of merge iterations (default: 6)
        target_layers: Layer counts to save checkpoints for (default: [6, 7, 8])
        recovery_epochs: Epochs for recovery training (default: 3)
        recovery_lr: Learning rate for recovery (default: 1e-5)
        patience: Early stopping patience (default: 2)
        save_dir: Directory to save final models (default: './weights/')
        keep_temp_checkpoints: Keep temporary recovery checkpoints (default: False)
    
    Returns:
        Dictionary with merge history
    """
    # Setup
    orig_n_layer = len(model.bert.encoder.layer)
    target_metric = get_primary_metric(task_name)
    threshold = init_metric[target_metric] * 0.01
    
    # Create recovery checkpoint directory
    recovery_dir = os.path.join(save_dir, 'recovery')
    os.makedirs(recovery_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Target Metric for {task_name}: {target_metric}")
    print(f"Original {target_metric} score: {init_metric[target_metric]:.4f}")
    print(f"Recovery threshold: {threshold:.4f}")
    print("")
    
    # History tracking
    merge_history = {
        'iterations': [],
        'layer_compositions': [],
        'metrics_after_merge': [],
        'metrics_after_recovery': [],
        'training_stats': []
    }
    
    # Merge loop
    for merge_itr in range(num_merges):
        print("=" * 70)
        print(f"Merge Iteration: {merge_itr + 1}/{num_merges}")
        print("=" * 70)
        
        # Layer Similarity Estimation
        print("\n[1/5] Computing CKA Similarity...")
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
        
        # Merge Most Adjacent Similar Layer Pair
        print("\n[2/5] Merging Layers...")
        merge_layer = stats['argmax']
        merged = merge_bert_layers(model, merge_layer, merge_layer + 1)
        model.bert.encoder.layer[merge_layer] = merged
        del model.bert.encoder.layer[merge_layer + 1]
        
        tracker.merge(merge_layer)
        print(f"  Merged Layers: {merge_layer} + {merge_layer + 1}")
        print(f"  Layer Composition: {tracker.get_mapping()}")
        print(f"  Total Layers: {len(tracker)}")
        
        # Evaluate Impacted Performance
        print("\n[3/5] Evaluating Post-Merge Performance...")
        eval_metric = eval_loop(model, val_dataloader, task_name, device)[0]
        print("  Metrics:")
        for metric_name, value in eval_metric.items():
            marker = "★" if metric_name == target_metric else " "
            print(f"    {marker} {metric_name}: {value:.4f}")
        
        merge_history['metrics_after_merge'].append(eval_metric)
        
        # Recovery Training
        print("\n[4/5] Recovery Training...")
        diff = init_metric[target_metric] - eval_metric[target_metric]
        print(f"  Performance drop: {diff:.4f} (threshold: {threshold:.4f})")

        is_retrained = False
        if diff > threshold:
            print("  → Recovery training NEEDED")
            
            # Setup optimizer and scheduler
            optimizer = AdamW(model.parameters(), lr=recovery_lr)
            num_training_steps = recovery_epochs * len(train_dataloader)
            # lr_scheduler = set_lr_scheduler(
            #     optimizer=optimizer,
            #     num_training_steps=num_training_steps
            # )
            
            print(f"  → Training: {recovery_epochs} epochs, lr={recovery_lr}")
            
            # Temporary checkpoint path
            temp_save_path = os.path.join(recovery_dir, f'temp_iter{merge_itr}.pt')
            
            # Train with checkpoint saving
            lr_scheduler= None
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
                teacher_model=teacher_model, alpha=alpha, temperature=temperature
            )
            
            # CRITICAL: Reload best checkpoint
            print("\n  → Loading best checkpoint...")
            best_checkpoint = torch.load(temp_save_path)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            
            print(f"  ✓ Loaded best checkpoint from epoch {best_checkpoint['epoch']}")
            print(f"    Best val loss: {best_checkpoint['val_loss']:.4f}")
            print(f"    Best val metrics: {best_checkpoint['val_metrics']}")
            
            merge_history['metrics_after_recovery'].append(best_checkpoint['val_metrics'])
            merge_history['training_stats'].append(train_stats)
            
            # Clean up temp checkpoint if not keeping
            if not keep_temp_checkpoints:
                os.remove(temp_save_path)
                print(f"  ✓ Cleaned up temporary checkpoint")

            is_retrained = True
        else:
            print("  → Recovery training NOT needed")
            merge_history['metrics_after_recovery'].append(eval_metric)
            merge_history['training_stats'].append(None)
        
        # Save final model at target layer counts
        print("\n[5/5] Saving Model...")
        n_layer = orig_n_layer - merge_itr - 1
        if n_layer in target_layers:
            save_path = os.path.join(save_dir, f'layer-{n_layer}-{task_name}.pt')
            
            save_object = {
                'model_state_dict': model.state_dict(),
                'layer_track': tracker.get_mapping(),
                'train_stats': merge_history['training_stats'][-1],
                'metrics_after_merge': merge_history['metrics_after_merge'][-1],
                'metrics_after_recovery': merge_history['metrics_after_recovery'][-1],
                'merge_iteration': merge_itr + 1,
                'num_layers': n_layer,
                'is_retrained': is_retrained
            }
            
            torch.save(save_object, save_path)
            print(f"  ✓ Saved {n_layer}-layer model to {save_path}")
        else:
            print(f"  • {n_layer} layers (not in target list, skipping save)")
        
        # Store iteration info
        merge_history['iterations'].append({
            'iteration': merge_itr + 1,
            'merged_layers': (merge_layer, merge_layer + 1),
            'num_layers_after': n_layer
        })
        merge_history['layer_compositions'].append(tracker.get_mapping())
        
        print("")
    
    # Final summary
    print("=" * 70)
    print("MERGE LOOP COMPLETED")
    print("=" * 70)
    print(f"Total merges: {num_merges}")
    print(f"Final layers: {len(tracker)}")
    print(f"Final composition: {tracker.get_mapping()}")

    torch.cuda.empty_cache()
    
    return merge_history

######################### Submission Prep #########################

import torch
from tqdm import tqdm
import pandas as pd

def generate_glue_submission(
    model,
    test_dataloader,
    task_name,
    output_dir='.',
    output_filename=None,
    device='cuda'
):
    """
    Generate GLUE submission file in TSV format for test set.
    
    This function creates the exact TSV format required for GLUE benchmark submission.
    Each task has specific requirements for column names and prediction format.
    
    Args:
        model: Trained BERT model
        test_dataloader: DataLoader for test set (NO LABELS)
        task_name: GLUE task name (case-insensitive)
            - 'cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'mnli-mm', 
              'qnli', 'rte', 'wnli', 'ax'
        output_dir: Directory to save TSV file
        output_filename: Custom filename (None = auto-generate as {TASK}.tsv)
        device: Device to run inference on
    
    Returns:
        str: Path to generated TSV file
    
    GLUE Submission Format Requirements:
        CoLA:  index, prediction (0/1)
        SST-2: index, prediction (0/1)
        MRPC:  index, prediction (0/1)
        STS-B: index, prediction (0.0-5.0, regression)
        QQP:   index, prediction (0/1)
        MNLI:  index, prediction (entailment/neutral/contradiction)
        QNLI:  index, prediction (entailment/not_entailment)
        RTE:   index, prediction (entailment/not_entailment)
        WNLI:  index, prediction (0/1)
        AX:    index, prediction (entailment/neutral/contradiction)
    
    Examples:
        # Generate MRPC submission
        generate_glue_submission(
            model, test_dataloader, 'mrpc',
            output_dir='./submissions'
        )
        # Creates: ./submissions/MRPC.tsv
        
        # Generate QNLI with custom filename
        generate_glue_submission(
            model, test_dataloader, 'qnli',
            output_dir='./submissions',
            output_filename='QNLI_8layer_merged.tsv'
        )
    """
    model.eval()
    model.to(device)
    
    task_name = task_name.lower()
    
    # Task-specific configurations
    task_configs = {
        'cola': {'num_labels': 2, 'label_type': 'binary', 'label_list': None},
        'sst2': {'num_labels': 2, 'label_type': 'binary', 'label_list': None},
        'mrpc': {'num_labels': 2, 'label_type': 'binary', 'label_list': None},
        'stsb': {'num_labels': 1, 'label_type': 'regression', 'label_list': None},
        'qqp': {'num_labels': 2, 'label_type': 'binary', 'label_list': None},
        'mnli': {'num_labels': 3, 'label_type': 'multiclass', 
                 'label_list': ['entailment', 'neutral', 'contradiction']},
        'mnli-mm': {'num_labels': 3, 'label_type': 'multiclass',
                    'label_list': ['entailment', 'neutral', 'contradiction']},
        'qnli': {'num_labels': 2, 'label_type': 'multiclass',
                 'label_list': ['entailment', 'not_entailment']},
        'rte': {'num_labels': 2, 'label_type': 'multiclass',
                'label_list': ['entailment', 'not_entailment']},
        'wnli': {'num_labels': 2, 'label_type': 'binary', 'label_list': None},
        'ax': {'num_labels': 3, 'label_type': 'multiclass',
               'label_list': ['entailment', 'neutral', 'contradiction']},
    }
    
    if task_name not in task_configs:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Supported tasks: {', '.join(task_configs.keys())}"
        )
    
    config = task_configs[task_name]
    
    print(f"Generating GLUE submission for {task_name.upper()}")
    print(f"Task type: {config['label_type']}")
    
    # Generate predictions
    predictions = []
    indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Predicting")):
            # Move batch to device (excluding labels if present)
            batch_inputs = {
                k: v.to(device) for k, v in batch.items() 
                if k not in ['labels', 'label', 'idx', 'index']
            }
            
            # Forward pass
            outputs = model(**batch_inputs)
            logits = outputs.logits
            
            # Get predictions based on task type
            if config['label_type'] == 'regression':
                # STS-B: regression task (0.0 to 5.0)
                preds = logits.squeeze(-1).cpu().numpy()
                # Clip to valid range
                preds = preds.clip(0.0, 5.0)
            else:
                # Classification: get argmax
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            predictions.extend(preds.tolist())
            
            # # Track indices (batch index * batch_size + position in batch)
            # batch_size = len(preds)
            # batch_indices = list(range(batch_idx * batch_size, 
            #                           batch_idx * batch_size + batch_size))
            # indices.extend(batch_indices)

    for i in range(len(predictions)):
        indices.append(i)
    
    # Convert predictions to required format
    if config['label_type'] == 'regression':
        # STS-B: keep as float
        formatted_predictions = predictions
    elif config['label_type'] == 'multiclass' and config['label_list']:
        # MNLI, QNLI, RTE, AX: map to label strings
        formatted_predictions = [config['label_list'][int(p)] for p in predictions]
    else:
        # Binary tasks (CoLA, SST-2, MRPC, QQP, WNLI): keep as 0/1
        formatted_predictions = [int(p) for p in predictions]
    
    # Create DataFrame
    df = pd.DataFrame({
        'index': indices,
        'prediction': formatted_predictions
    })
    
    # Determine output filename
    if output_filename is None:
        output_filename = f"{task_name.upper()}.tsv"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Full output path
    output_path = os.path.join(output_dir, output_filename)
    
    # Save to TSV
    df.to_csv(output_path, sep='\t', index=False)
    
    print(f"\n✓ Submission file created: {output_path}")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Format: index, prediction")
    
    # Show sample
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    return output_path
    
####################### Functions for Plots  #######################

import matplotlib.pyplot as plt

def plotLoss(trainingLoss, valLoss, legend=["Training Loss", "Val Loss"], title=None):
    plt.plot(trainingLoss)
    plt.plot(valLoss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(legend)
    if title is not None:
        plt.title(title)

def plotLossMulti(losses, legend, title=None):
    for loss in losses:
        plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend(legend)
    if title is not None:
        plt.title(title)
    
def plotAccuracy(trainingAcc, valAcc, legend=["Training Accuracy", "Val Accuracy"], title=None, xlabel=None, ylabel=None):
    plt.plot(trainingAcc)
    plt.plot(valAcc)
    
    if xlabel != None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel("Epoch")

    if ylabel != None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("Accuracy (%)")
    
    plt.legend(legend)
    if title is not None:
        plt.title(title)

        
def plotLossMulti(accs, legend, title=None):
    for acc in accs:
        plt.plot(acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(legend)
    if title is not None:
        plt.title(title)