from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd

# Load the saved model
model = lgb.Booster(model_file='model.txt')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the JSON data received in the POST request
    input_data = request.get_json()

    # Convert the input data into a pandas DataFrame for prediction
    input_df = pd.DataFrame([input_data])

    # Make a prediction using the model
    prediction = model.predict(input_df)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
