#load data and prepare for pytorch 

import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def get_dataloader(batch_size=128):
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "processed" / "quickdraw_dataset.npz"
    
    # load the .npz file
    data = np.load(data_path)
    X = torch.tensor(data['X'])
    y = torch.tensor(data['y'])

    # create dataset
    dataset = TensorDataset(X,y)

    # 80/20 train test split
    indices = np.arange(len(dataset))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y.numpy()
    )

    # split remaining 20% equally into validation and test
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42,
        stratify=y.numpy()[temp_idx]
    )

    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader
# → move onto vae.py
