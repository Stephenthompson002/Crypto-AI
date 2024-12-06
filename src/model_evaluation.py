import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')


# Function to preprocess data
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear  # Alternatively, use this for "days since start"
    df = df.drop(columns=['date'])
    
    X = df.drop(columns=['price'])
    y = df['price']
    
    return X, y

# Function to visualize predictions
def visualize_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label="Actual", color='blue')
    plt.plot(y_test.index, y_pred, label="Predicted", color='red', linestyle='--')
    plt.title('Actual vs Predicted Crypto Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Save the plot
    plt.savefig('predictions_plot.png')  # Save as PNG file
    plt.close()  # Close the plot to free memory


# Function to evaluate the model
def evaluate_model(file_path, model_path):
    X, y = preprocess_data(file_path)
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Load the trained model
    model = joblib.load(model_path)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    
    # Visualize the results
    visualize_predictions(y_test, y_pred)

# Example usage:
if __name__ == '__main__':
    file_path = 'data/crypto_data.csv'  # Replace with your actual data file
    model_path = 'reports/crypto_price_model.pkl'  # Replace with your model file path

    # Evaluate the model and visualize results
    evaluate_model(file_path, model_path)
