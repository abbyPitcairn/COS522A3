import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class MixedCSVDataset(Dataset):
    """
    Create a custom dataset loader for a CSV file with both numeric and categorical data.
    Inputs:
        - csv_file: str
        - split: str
        - seed: int (default 42)
        - train_ratio: float (default 0.8)
        - val_ratio: float (default 0.1)
    """
    def __init__(
        self,
        csv_file: str,
        split: str = "train",
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        # Step 1: Load CSV
        df = pd.read_csv(csv_file, header=None, skiprows=1)
        numeric_columns = [0, 1, 2, 3, 4]   # 5 numeric columns
        cat_columns = [5, 6]                # 2 categorical columns
        label_column = 7                    # 1 label column

        # Step 2: One-hot encode categorical features
        cat_onehot = pd.get_dummies(df[cat_columns].astype(str))
        # Combine numeric + categorical features back together:
        features_df = pd.concat([df[numeric_columns], cat_onehot], axis=1)

        # Step 3: split the dataset (reproducible shuffle)
        n = len(df) # Get the total number of rows
        idx = np.arange(n)  # Index the rows
        rng = np.random.default_rng(seed)   # Create a reproducible random range
        rng.shuffle(idx)    # Shuffle the data based on reproducible random range

        train_end = int(train_ratio * n)    # Set the end of training dataset to the training ratio * total rows
        val_end = int((train_ratio + val_ratio) * n)    # Set the end of the validation set to (training + validation ratios) * total rows

        train_idx = idx[:train_end] # Index the training subset [beginning -> train_end]
        val_idx = idx[train_end:val_end] # Index the validation subset [train_end -> val_end]
        test_idx = idx[val_end:] # Index the test subset [val_end -> end of dataset]

        # Assign the proper subset of the dataset, depending on which split is specified:
        if split == "train":
            split_idx = train_idx
        elif split == "val":
            split_idx = val_idx
        elif split == "test":
            split_idx = test_idx
        else:   # If the split is not a valid selection, raise error:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # Step 4: Standardize numeric features using TRAIN stats only
        # Compute mean/std from training data (avoids leakage of test/validation data)
        train_numeric = features_df.iloc[train_idx, :5].to_numpy()
        mean = train_numeric.mean(axis=0)
        std = train_numeric.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        # Extract split features
        X_numeric = features_df.iloc[split_idx, :5].to_numpy()
        X_cat = features_df.iloc[split_idx, 5:].to_numpy()
        # Standardize numeric
        X_numeric = (X_numeric - mean) / std
        # Combine the numeric and categorical features back together
        self.features = np.concatenate([X_numeric, X_cat], axis=1).astype(np.float32)

        # Step 5: map labels to a number
        labels_raw = df.iloc[split_idx, label_column]
        label_map = {"A": 0, "B": 1, "C": 2}
        self.labels = labels_raw.map(label_map).to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.from_numpy(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

# Dataset path:
dataset_path = "Data/dataset.csv"

# Create dataset (training only)
train_dataset = MixedCSVDataset(dataset_path, split='train')

# Create DataLoader (training only)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Print one batch (batch size = 4)
for features, labels in train_loader:
    print("features:", features)
    print("labels:", labels)
    print("features shape:", features.shape)
    print("labels shape:", labels.shape)
    break