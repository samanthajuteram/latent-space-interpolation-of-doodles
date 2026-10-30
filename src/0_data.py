#load data and prepare for pytorch 

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

def get_dataloader(batch_size=128):
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "processed" / "quickdraw_dataset.npz"
    
    # load the .npz file
    data = np.load(data_path)
    X = torch.tensor(data['X'])
    y = torch.tensor(data['y'])
    
    # make loader
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# → move onto vae.py
