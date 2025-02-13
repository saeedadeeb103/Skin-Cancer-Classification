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



def save_test_data(test_dataset, dataset, save_dir):
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)  # Delete the existing directory and its contents
        print(f"Existing test data directory '{save_dir}' removed.")

    os.makedirs(save_dir, exist_ok=True)

    for idx in test_dataset.indices:
        audio_file_path = dataset.audio_files[idx]  # Assuming dataset has `audio_files` attribute
        label = dataset.labels[idx]  # Assuming dataset has `labels` attribute

        # Create a directory for the label if it doesn't exist
        label_dir = os.path.join(save_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # Copy the audio file to the label directory
        shutil.copy(audio_file_path, label_dir)

    print(f"Test data saved in {save_dir}")