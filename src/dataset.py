# src/dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TelemetryDataset(Dataset):
    """
    Expects folder of CSVs/NPYs where each file is (time_steps, features)
    We'll pad/cut to fixed seq_len
    """
    def __init__(self, data_dir: str, seq_len: int = 100, normalize: bool = False):
        self.files = glob.glob(os.path.join(data_dir, "*.csv")) + \
                     glob.glob(os.path.join(data_dir, "*.npy")) + \
                     glob.glob(os.path.join(data_dir, "*.npz"))
        self.seq_len = seq_len
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def _load_file(self, path):
        if path.endswith(".csv"):
            arr = np.loadtxt(path, delimiter=",")
        elif path.endswith(".npy"):
            arr = np.load(path)
        elif path.endswith(".npz"):
            arr = np.load(path)["arr_0"]
        else:
            raise ValueError("Unsupported file: " + path)
        return arr  # (T, F)

    def __getitem__(self, idx):
        path = self.files[idx]
        arr = self._load_file(path)  # (T, F)
        # make sure 2D
        if arr.ndim == 1:
            arr = arr[:, None]

        T, F = arr.shape

        if self.normalize:
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True) + 1e-6
            arr = (arr - mean) / std

        # pad / cut
        if T >= self.seq_len:
            arr = arr[:self.seq_len, :]
        else:
            pad = np.zeros((self.seq_len - T, F))
            arr = np.vstack([arr, pad])

        return torch.tensor(arr, dtype=torch.float32)  # (seq_len, F)

def get_dataloader(data_dir, seq_len=100, batch_size=16, shuffle=True, normalize=False):
    ds = TelemetryDataset(data_dir, seq_len=seq_len, normalize=normalize)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
