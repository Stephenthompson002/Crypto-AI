from flask import Flask, request, jsonify
import numpy as np
import joblib  # or whichever method you're using to load the model

app = Flask(__name__)

# Load your pre-trained model here (make sure to specify the correct path)
model = joblib.load("models/crypto_price_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    try:
        # Access all 7 features
        features = [
            data['price'], data['30_day_MA'], data['60_day_MA'], data['RSI'],
            data['feature5'], data['feature6'], data['feature7']
        ]
    except KeyError as e:
        # Handle missing features gracefully and return a helpful error message
        return jsonify({'error': f'Missing feature: {str(e)}'}), 400

    # Reshape the features if needed (1 sample with the expected number of features)
    features = np.array(features).reshape(1, -1)

    # Predict using the model
    prediction = model.predict(features)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
