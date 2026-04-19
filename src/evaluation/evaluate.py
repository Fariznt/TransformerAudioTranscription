"""
Role: Entry point for model evaluation on the test set.
Computes multi-instrument, multi-metric performance on complete files using mir_eval.
"""
import torch

def evaluate_test_set(model: torch.nn.Module, test_loader, config, device: str):
    """
    Iterates over the test dataloader.
    - Autoregressively generates predictions.
    - Decodes tokens to MIDI.
    - Computes precision, recall, and F1.
    - Averages and reports final metrics.
    """
    pass

def log_metrics(metrics_dict: dict):
    """
    Helper to print or log evaluation metrics to wandb/console.
    """
    pass
