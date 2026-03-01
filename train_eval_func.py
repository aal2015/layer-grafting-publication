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

class EarlyStopping():
    def __init__(self, patience = 3, threshold = 0.1):
        self.patience      = patience
        self.threshold     = threshold
        self.min_val_loss = None
        self.patienceCount = 0
        
    def _checkPatience(self,):
        if self.patienceCount == self.patience:
            return True
        else:
            self.patienceCount += 1
            return False
    
    def checkCondition(self, val_loss):
        if self.min_val_loss == None:
            self.min_val_loss = val_loss
        elif val_loss - self.min_val_loss > self.threshold:
            return self._checkPatience()
        else:
            self.patienceCount = 0
        return False

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

def eval_loop(model, dataloader, dataset, device, benchmark="glue"):
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
        predictions = torch.argmax(logits, dim=-1)
        
        metric.add_batch(predictions=predictions, references=batch["labels"])
        total_loss += float(loss)  # ← Add to total
        n_batches += 1
    
    avg_loss = total_loss / n_batches  # ← Compute average
    return metric.compute(), avg_loss   # ← Return average

from torch.optim import AdamW

import torch.nn as nn
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

def train(model, train_dataloader, val_dataloader, task_name, device,
          num_epochs=5, lr=3e-5, patience=3, threshold=0.1,
          save_path=None, optimizer=None, lr_scheduler=None, 
          use_early_stopping=False, display_epoch_iter=False,
          benchmark="glue"):
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
        benchmark: Benchmark name (default: "glue")
    
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
    val_old_loss = float("Inf")
    
    # Training loop
    train_start = time.time()

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
            
            # Compute predictions
            predictions = torch.argmax(logits, dim=-1)
            train_metric.add_batch(
                predictions=predictions,
                references=batch["labels"]
            )
            
            # Backward pass
            total_loss += float(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
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
            model, val_dataloader, task_name, device, benchmark
        )
        
        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        epoch_time = time.time() - epoch_start
        time_per_epoch_hist.append(epoch_time)
        
        # Save checkpoint if path provided and loss improved
        if save_path is not None and val_loss < val_old_loss:
            val_old_loss = val_loss
            
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
            if early_stopping.checkCondition(val_loss):
                print(">>>>> Early stopping callback <<<<<")
                break
    
    total_train_time = time.time() - train_start
    print(f"\nTotal Training Time: {total_train_time:.2f} sec")
    if save_path is not None:
        print(f"Best validation loss: {val_old_loss:.4f}")
    
    # Return training history
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics_history': train_metrics_history,
        'val_metrics_history': val_metrics_history,
        'time_per_epoch': time_per_epoch_hist,
        'total_time': total_train_time,
        'best_val_loss': val_old_loss,
        'num_epochs_trained': len(train_losses)
    }
    

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
    
def plotAccuracy(trainingAcc, valAcc, legend=["Training Accuracy", "Val Accuracy"], title=None):
    plt.plot(trainingAcc)
    plt.plot(valAcc)
    plt.xlabel("Epoch")
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