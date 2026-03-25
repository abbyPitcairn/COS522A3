import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class MixedCSVDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        split: str = "train",
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        # Step 1: Load CSV


        # Step 2: One-hot encode categorical features

        # Step 3: split the dataset (reproducible shuffle)

        # Step 4: Standardize numeric features using TRAIN stats only

        # Step 5: map labels to a number


    def __len__(self):
        ?
    def __getitem__(self, idx):
        ?


# Create Dataset object and wrap up DataLoader
?

# ---- Print one batch ----
for features, labels in train_loader:
    print("features:", ?)
    print("labels:", ?)
    print("features shape:", ?)
    print("labels shape:", ?)
    break
