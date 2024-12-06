import requests
import pandas as pd
import datetime as dt
import os

# Define function to fetch historical data
def fetch_crypto_data(crypto_id='bitcoin', currency='usd', days=365, save_path='data/raw/'):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart'
    params = {
        'vs_currency': currency,
        'days': days,
        'interval': 'daily'
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        # Extract prices and convert to DataFrame
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.drop(columns='timestamp', inplace=True)
        
        # Save to CSV
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f'{crypto_id}_prices.csv')
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        return df
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None

# Example usage
if __name__ == "__main__":
    df = fetch_crypto_data(crypto_id='bitcoin', days=365)
    print(df.head())
