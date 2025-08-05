import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        return 0.0

def evaluate_and_export(model, dataloader, scalers, device, output_csv_path='predictions.csv'):
    model.to(device)
    model.eval() 
    
    predictions = []
    actuals = []
    tickers_list = []
    
    with torch.no_grad():
        for x, y, tickers in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            for i in range(len(tickers)):
                ticker = tickers[i]
                scaler = scalers[ticker]
                pred_unscaled = output[i].item() * scaler['std'] + scaler['mean']
                actual_unscaled = y[i].item() * scaler['std'] + scaler['mean']

                predictions.append(pred_unscaled)
                actuals.append(actual_unscaled)
                tickers_list.append(ticker)
    
    # Create the results DataFrame to make calculations easier
    results_df = pd.DataFrame({
        'Ticker': tickers_list,
        'Actual': actuals,
        'Predicted': predictions
    })

    # --- Create the statistics header ---
    header_lines = ["# --- Model Performance Metrics by Company ---"]
    
    unique_tickers = results_df['Ticker'].unique()

    # Loop through tickers and add their stats to the header
    for ticker in unique_tickers:
        ticker_df = results_df[results_df['Ticker'] == ticker]
        
        y_true = ticker_df['Actual']
        y_pred = ticker_df['Predicted']
        
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # Add a blank line before each company's stats for separation
        header_lines.append("#") 
        header_lines.append(f"# Company: {ticker}")
        header_lines.append(f"#   - Mean Squared Error (MSE): {mse:.4f}")
        header_lines.append(f"#   - R-squared: {r2:.4f}")
        header_lines.append(f"#   - Mean Absolute Error (MAE): ${mae:.4f}")
        header_lines.append(f"#   - Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Then show raw predictions
    header_lines.append("#")
    header_lines.append("# Raw Predictions ")

    # Export to CSV innit
    try:
        with open(output_csv_path, 'w', encoding = 'utf-8') as f:
            f.write('\n'.join(header_lines))
            f.write('\n')  
            
            # Append the DataFrame to the same file
            results_df.to_csv(f, index=False, lineterminator='\n')
            
        print(f"Evaluation complete. Results and stats exported to {output_csv_path}")

    except IOError as e:
        print(f"Error writing to file {output_csv_path}: {e}")