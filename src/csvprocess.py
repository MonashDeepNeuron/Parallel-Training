import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class StockDataset(Dataset):
    def __init__(self, csv_path, window_size=32, horizon=1):
        df = pd.read_csv(csv_path, parse_dates=['Datetime'])

        df = df.drop(columns=['Datetime', 'Ticker'])

        self.data = (df - df.mean()) / df.std()

        self.data = self.data.values.astype(np.float32)
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]             # (window_size, features)
        y = self.data[idx + self.window_size + self.horizon - 1][3]  # Target: Close price (index 3)
        return torch.tensor(x), torch.tensor(y)
