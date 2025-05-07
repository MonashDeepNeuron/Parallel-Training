from training import train
from transformer import TimeSeriesTransformer
from csvprocess import StockDataset
from torch.utils.data import DataLoader


WINDOW_SIZE = 32
BATCH_SIZE = 64

dataset = StockDataset("data.csv", window_size=WINDOW_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

example_input, _ = dataset[0]
input_dim = example_input.shape[1]

model = TimeSeriesTransformer(input_dim=input_dim)
train(model, dataloader, epochs=20)
