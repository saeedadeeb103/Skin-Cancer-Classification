import torch
import torch.nn as nn
import os
import shutil
def normalize_ratios(ratios):
    total = sum(ratios)
    return [r / total for r in ratios]

def collate_fn(batch):
    inputs, targets, input_lengths, target_lengths = zip(*batch)
    inputs = torch.stack(inputs)  # Convert list of tensors to a batch tensor
    targets = torch.cat(targets)  # Flatten target sequences
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return inputs, targets, input_lengths, target_lengths
