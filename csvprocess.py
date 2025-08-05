import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class StockDataset(Dataset):
    def __init__(self, csv_path, window_size=32, horizon=1, train_split=0.8, mode='train'):
        df = pd.read_csv(csv_path, parse_dates=['Datetime'])
        self.window_size = window_size
        self.horizon = horizon
        self.sequences = []
        self.scalers = {} # Store the mean for each ticker

        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            
            # ITraining data is only used to calculate mean
            train_data_end_index = int(len(ticker_df) * train_split)
            train_df = ticker_df.iloc[:train_data_end_index]

            # Storing close price mean and standard deviation
            mean = train_df['Close'].mean()
            std = train_df['Close'].std()
            self.scalers[ticker] = {'mean': mean, 'std': std}

            # Remove other columns (not needed anymore)
            ticker_df = ticker_df.drop(columns=['Datetime', 'Ticker'])

            # Normalize the entire dataset
            data = (ticker_df - ticker_df.mean()) / ticker_df.std()
            data = data.values.astype(np.float32)

            # Train based on each index
            split_idx = int(len(data) * train_split)
            if mode == 'train':
                data_to_process = data[:split_idx]
            else: # mode == 'test'
                data_to_process = data[split_idx:]

            # Create sequences for this ticker from the correct data split
            for i in range(len(data_to_process) - self.window_size - self.horizon + 1):
                x = data_to_process[i : i + self.window_size]
                y = data_to_process[i + self.window_size + self.horizon - 1][3] 
                self.sequences.append((x, y, ticker))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y, ticker = self.sequences[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), ticker