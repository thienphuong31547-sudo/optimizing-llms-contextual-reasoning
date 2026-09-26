import json
import torch

def load_data(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def move_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()
