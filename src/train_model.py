import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from preprocess_data import load_data, handle_missing_values, feature_engineering, scale_data, split_data

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    df = load_data(file_path)
    if df is not None:
        df = handle_missing_values(df)
        df = feature_engineering(df)
        df = scale_data(df)
        X_train, X_test, y_train, y_test = split_data(df)
        return X_train, X_test, y_train, y_test
    else:
        print("Data loading failed.")
        return None, None, None, None

# Train Linear Regression Model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")

# Save Model
def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Main Function
if __name__ == "__main__":
    file_path = 'data/processed/processed_data.csv'  # Adjust path if necessary
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    
    if X_train is not None and y_train is not None:
        # Train the model
        model = train_linear_regression(X_train, y_train)
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test)
        
        # Save the model
        save_model(model, 'models/crypto_price_predictor.pkl')
    else:
        print("Data preprocessing failed. Model training aborted.")
