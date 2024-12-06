import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Function to preprocess data
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract year, month, and day as numerical features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear  # Alternatively, use this for "days since start"
    
    # Calculate RSI and add as a feature
    df['RSI'] = compute_rsi(df['price'])
    
    # Drop the original 'date' column
    df = df.drop(columns=['date'])
    
    # Assuming 'price' is the target variable, and the rest are features
    X = df.drop(columns=['price'])
    y = df['price']
    
    return X, y

# Function to compute RSI
def compute_rsi(prices, period=14):
    """
    Compute the Relative Strength Index (RSI) for a given price series.
    :param prices: pandas Series of prices.
    :param period: The period over which to calculate the RSI (default is 14 days).
    :return: pandas Series containing the RSI values.
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Function to train the model
def train_model(file_path, model_path):
    X, y = preprocess_data(file_path)
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize and train your model (e.g., LinearRegression)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')
    
    # Make predictions on the test set and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')

# Function to make predictions using the trained model
def make_predictions(model, data):
    # Ensure data is in the right shape (2D array, not a Series or single-column dataframe)
    predictions = model.predict(data)
    return predictions

# Function to load the model and make predictions
def predict_crypto_price(file_path, model_path):
    # Preprocess the input data
    X, _ = preprocess_data(file_path)  # We only need the features for prediction

    # Load the trained model
    model = joblib.load(model_path)

    # Make predictions using the model
    predictions = make_predictions(model, X)

    # Print the predictions
    print(f'Predictions: {predictions}')

# Example usage:
if __name__ == '__main__':
    file_path = 'data/crypto_data.csv'  # Replace with the correct file path
    model_path = 'reports/crypto_price_model.pkl'  # Replace with the desired model file path

    # Train the model if necessary (uncomment this if you need to retrain the model)
    train_model(file_path, model_path)

    # Make predictions using the trained model
    predict_crypto_price(file_path, model_path)
