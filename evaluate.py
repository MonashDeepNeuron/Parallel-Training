import torch
import pandas as pd

def evaluate_and_export(model, dataloader, scalers, device, output_csv_path='predictions.csv'):
    model.to(device)
    model.eval() 
    
    predictions = []
    actuals = []
    
    with torch.no_grad(): # no grad needed for an evaulate
        for x, y, tickers in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            for i in range(len(tickers)):
                ticker = tickers[i]
                scaler = scalers[ticker]

                pred_unscaled = output[i].item() * scaler['std'] + scaler['mean']
                
                # Inverse transform the actual value to compare
                actual_unscaled = y[i].item() * scaler['std'] + scaler['mean']

                predictions.append(pred_unscaled)
                actuals.append(actual_unscaled)
    
    # Create a DataFrame and save the results
    results_df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions
    })

    results_df.to_csv(output_csv_path, index=False)
    print(f"Evaluation complete. Results exported to {output_csv_path}")