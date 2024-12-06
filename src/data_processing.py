import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
