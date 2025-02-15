from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd

# Load both models
model1 = lgb.Booster(model_file='model1.txt')  # Model 1
model2 = lgb.Booster(model_file='model2.txt')  # Model 2

app = Flask(__name__)

@app.route('/predict1', methods=['POST'])
def predict_model1():
    """Predict using Model 1 (model1.txt)."""
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])  # Convert JSON to DataFrame
    prediction = model1.predict(input_df)  # Predict
    return jsonify({'prediction': prediction.tolist()})

@app.route('/predict2', methods=['POST'])
def predict_model2():
    """Predict using Model 2 (model2.txt)."""
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])  # Convert JSON to DataFrame
    prediction = model2.predict(input_df)  # Predict
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
