from training import train
from transformer import TimeSeriesTransformer
from csvprocess import StockDataset
from evaluate import evaluate_and_export
from plotter import plot_results
from torch.utils.data import DataLoader
import torch

# Standard config, change to speed up/slow down. Higher is more accurate
WINDOW_SIZE = 16
BATCH_SIZE = 128
TRAIN_SPLIT = 0.8
EPOCHS = 5

# Checking if its using the GPU version of torch instead of CPU, only for my PC, only laptop itll use the CPU one though. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Creating data sets/splitting data
print("Loading and splitting data")
train_dataset = StockDataset("data.csv", window_size=WINDOW_SIZE, train_split=TRAIN_SPLIT, mode='train')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# The test dataset uses the same scalers calculated from the training set
test_dataset = StockDataset("data.csv", window_size=WINDOW_SIZE, train_split=TRAIN_SPLIT, mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialising the model
# We need to get the input dimension from a sample in the dataset
example_input, _, _ = train_dataset[0]
input_dim = example_input.shape[1]

model = TimeSeriesTransformer(input_dim=input_dim, d_model=64, num_heads=2, num_layers=2, dropout=0.1)
print("Model initialized.")

# Trainign ther model
print("\nStarting training...")
train(model, train_dataloader, epochs=EPOCHS)
print("Training finished.")

# Evaluate how good the model is , then export the data
# We use the scalers from the test_dataset object to inverse transform the data
evaluate_and_export(model, test_dataloader, test_dataset.scalers, device, output_csv_path='predictions.csv')

# Plotting results 
print("Generating plots...")
plot_results('predictions.csv')