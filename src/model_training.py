import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model_evaluation import evaluate_model


# Load Data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        df.sort_values('date', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Handle Missing Values
def handle_missing_values(df):
    try:
        df.fillna(df.mean(), inplace=True)
        return df
    except Exception as e:
        print(f"Error handling missing values: {e}")
        return df

# Feature Engineering
def feature_engineering(df):
    try:
        df['30_day_MA'] = df['price'].rolling(window=30).mean()
        df['60_day_MA'] = df['price'].rolling(window=60).mean()

        # Relative Strength Index (RSI)
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df.dropna(inplace=True)  # Drop rows with NaN values from rolling calculations
        return df
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return df

# Scale Features
def scale_data(df):
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[['price', '30_day_MA', '60_day_MA', 'RSI']])
        df[['price', '30_day_MA', '60_day_MA', 'RSI']] = scaled_features
        return df
    except Exception as e:
        print(f"Error scaling data: {e}")
        return df

# Train-Test Split
def split_data(df):
    try:
        X = df[['price', '30_day_MA', '60_day_MA', 'RSI']]  # Features
        y = df['price']  # Target variable

        # Split into training and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None, None, None


def train_and_save_model(file_path, model_path):
    # Load and preprocess data
    df = load_data(file_path)
    if df is not None:
        df = handle_missing_values(df)
        df = feature_engineering(df)
        df = scale_data(df)
        
        X_train, X_test, y_train, y_test = split_data(df)

        if X_train is not None and y_train is not None:
            # Initialize and train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Save the model
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")

            # Make predictions and evaluate the model
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            print(f"Mean Absolute Error: {mae}")
        else:
            print("Error during train-test split.")
    else:
        print("Error loading data.")


if __name__ == "__main__":
    file_path = 'data/raw/crypto_data.csv'
    model_path = 'models/crypto_price_model.pkl'

    # Train the model
    train_and_save_model(file_path, model_path)  # Corrected function name

    # Evaluate the model after training
    evaluate_model(file_path, model_path)
