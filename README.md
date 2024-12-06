# Crypto Price Prediction Application

## Overview
This project is a Flask-based API for predicting cryptocurrency prices using a Linear Regression model. It takes input features (e.g., price, moving averages, RSI, etc.) and returns the predicted price. The project demonstrates the integration of a machine learning model into a web API for real-time prediction.

---

## Objectives
- To build a lightweight API for predicting cryptocurrency prices.
- To utilize historical financial indicators as features for prediction.
- To demonstrate the deployment of machine learning models using Flask.

---

## Features
- Input: Accepts JSON data with features like `price`, `30_day_MA`, `60_day_MA`, `RSI`, and other relevant indicators.
- Processing: Uses a pre-trained Linear Regression model to make predictions.
- Output: Returns the predicted price in JSON format.

---

## Implementation

### 1. Model Training
The Linear Regression model was trained using historical cryptocurrency data:
- Dataset: Historical cryptocurrency prices with indicators like moving averages and RSI.
- Features:
  - `price`
  - `30_day_MA` (30-day moving average)
  - `60_day_MA` (60-day moving average)
  - `RSI` (Relative Strength Index)
  - Other additional indicators (`feature5`, `feature6`, `feature7`).
- Target: Next-day price.
- Libraries Used: `scikit-learn`, `pandas`, and `numpy`.

The trained model was saved using the `joblib` library for deployment.

### 2. Flask API Development
- Endpoints:
  - `POST /predict`: Accepts input features in JSON format and returns the predicted price.
- Libraries Used: Flask, NumPy, joblib.

#### Example Input:
{
  "price": 50000,
  "30_day_MA": 48000,
  "60_day_MA": 47000,
  "RSI": 55,
  "feature5": 1.5,
  "feature6": 100000,
  "feature7": 0.02
}

#### Example Output:
{
  "prediction": 51000
}

### 3. Error Handling
- Missing Features: The API validates the input for all required features and returns a clear error message if any are missing.
  {
    "error": "Missing feature: 'price'"
  }
- Invalid Data Types: The API checks for correct data types (e.g., numerical values) and handles errors gracefully.

### 4. Deployment
The Flask application is designed to run locally or on cloud platforms (e.g., AWS, Azure, or Heroku).

#### Steps:
1. Install dependencies using `requirements.txt`.
2. Run the application:
   python app.py
3. Access the API at http://localhost:5000/predict.

---

## Folder Structure
crypto-price-prediction/
├── data/                 # Dataset (not included in repo due to size)
├── model/                # Saved Linear Regression model (model.pkl)
├── app.py                # Flask application
├── requirements.txt      # Python dependencies
├── README.md             # Project overview
└── tests/                # Test scripts for the API

---

## Future Enhancements
- Advanced Models: Replace Linear Regression with advanced models like LSTM for time series forecasting.
- UI Integration: Build a front-end interface for easier input and visualization.
- Feature Engineering: Add more robust financial indicators for better predictions.
- Live Data Integration: Fetch live cryptocurrency data via APIs for real-time predictions.

---

## Conclusion
This project demonstrates how to integrate a machine learning model into a web application using Flask. By combining Python-based tools for training and deployment, this API serves as a foundational example for building more sophisticated financial prediction systems.
