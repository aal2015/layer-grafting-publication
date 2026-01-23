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

def eval_loop(model, dataloader, dataset, device, benchmark="glue"):
    metric = evaluate.load(benchmark, dataset)
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        model_loss = outputs.loss
        logits     = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        
    return metric.compute(), float(model_loss)

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