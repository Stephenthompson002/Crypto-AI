import requests
import json

url = "http://127.0.0.1:5000/predict"

data = {
    "price": 1000,
    "30_day_MA": 950,
    "60_day_MA": 970,
    "RSI": 45,
    "feature5": 1200,
    "feature6": 1100,
    "feature7": 1050
}


response = requests.post(url, json=data)  # Ensure using POST, not GET

if response.status_code == 200:
    print("Prediction Result:", response.json())
else:
    print(f"Failed to get prediction. Status Code: {response.status_code}")
    print(response.text)
