import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    return df

# Plot Price Trends
def plot_price_trend(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['price'], label='Price')
    plt.title('Cryptocurrency Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Moving Average
def plot_moving_average(df, window=30):
    df['moving_avg'] = df['price'].rolling(window=window).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['price'], label='Daily Price', alpha=0.5)
    plt.plot(df['date'], df['moving_avg'], label=f'{window}-Day Moving Avg', color='orange')
    plt.title(f'{window}-Day Moving Average of Price')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Returns and Volatility
def plot_daily_returns(df):
    df['daily_return'] = df['price'].pct_change()
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['daily_return'], label='Daily Return', color='purple')
    plt.title('Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.show()

# Correlation Heatmap (if additional features are added)
def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

# Example Usage
if __name__ == "__main__":
    file_path = 'data/raw/bitcoin_prices.csv'  # Adjust path if necessary
    df = load_data(file_path)
    
    # Run EDA
    plot_price_trend(df)
    plot_moving_average(df, window=30)
    plot_daily_returns(df)
