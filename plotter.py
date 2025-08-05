import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(csv_path='predictions.csv'):

    try:
        df = pd.read_csv(csv_path, comment='#', encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found. Please run the evaluation first.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    if not all(col in df.columns for col in ['Ticker', 'Actual', 'Predicted']):
        print("Error: The CSV file must contain 'Ticker', 'Actual', and 'Predicted' columns.")
        return
    
    # Get a list of unique tickers from the DataFrame
    unique_tickers = df['Ticker'].unique()

    for ticker in unique_tickers:
        print(f"Generating plots for {ticker}...")
        # Add a check to ensure there's data for the ticker
        if df[df['Ticker'] == ticker].empty:
            print(f"  Skipping {ticker} - no data found.")
            continue
            
        ticker_df = df[df['Ticker'] == ticker].copy()

        # Predicted vs Actual for each company
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Residuals Scatter Plot
        sns.regplot(x='Actual', y='Predicted', data=ticker_df, ax=ax,
                    scatter_kws={'alpha': 0.6, 's': 50},
                    line_kws={'color': 'red', 'linewidth': 2})

        # Line of perfect prediction
        min_val = min(ticker_df['Actual'].min(), ticker_df['Predicted'].min())
        max_val = max(ticker_df['Actual'].max(), ticker_df['Predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction (y=x)')

        ax.set_title(f'Predicted vs. Actual Stock Prices for {ticker}', fontsize=16)
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.legend()
        ax.grid(True)
        plt.show()

        # Residual plot for company
        ticker_df['Residuals'] = ticker_df['Actual'] - ticker_df['Predicted']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Residual Analysis for {ticker}', fontsize=18)

        # Histogram of residuals
        sns.histplot(ticker_df['Residuals'], kde=True, ax=ax1)
        ax1.set_title('Distribution of Residuals', fontsize=14)
        ax1.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.axvline(0, color='red', linestyle='--')

        # Scatter plot of residuals vs predicted values
        ax2.scatter(ticker_df['Predicted'], ticker_df['Residuals'], alpha=0.5)
        ax2.axhline(0, color='red', linestyle='--')
        ax2.set_title('Residuals vs. Predicted Values', fontsize=14)
        ax2.set_xlabel('Predicted Values', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()