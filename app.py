from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd
from prometheus_client import start_http_server, Summary, Gauge, Counter, REGISTRY
import time

# Load both models
model1 = lgb.Booster(model_file='model1.txt')  # Model 1
model2 = lgb.Booster(model_file='model2.txt')  # Model 2

# Create Flask app
app = Flask(__name__)

# Create Prometheus metrics
REQUEST_LATENCY = Summary('request_latency_seconds', 'Time taken to process a request')
REQUEST_RESPONSE_TIME = Gauge('request_response_time_seconds', 'Response time for each request')
REQUEST_THROUGHPUT = Counter('request_throughput_total', 'Total number of requests processed')

# Start Prometheus HTTP server on port 8000
start_http_server(8000)

@app.route('/predict1', methods=['POST'])
def predict_model1():
    """Predict using Model 1 (model1.txt)."""
    start_time = time.time()

    # Get input data
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])  # Convert JSON to DataFrame

    # Predict
    prediction = model1.predict(input_df)

    # Calculate latency and response time
    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)
    REQUEST_RESPONSE_TIME.set(latency)

    # Increment throughput
    REQUEST_THROUGHPUT.inc()

    # Return prediction
    return jsonify({'prediction': prediction.tolist()})

@app.route('/predict2', methods=['POST'])
def predict_model2():
    """Predict using Model 2 (model2.txt)."""
    start_time = time.time()

    # Get input data
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])  # Convert JSON to DataFrame

    # Predict
    prediction = model2.predict(input_df)

    # Calculate latency and response time
    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)
    REQUEST_RESPONSE_TIME.set(latency)

    # Increment throughput
    REQUEST_THROUGHPUT.inc()

    # Return prediction
    return jsonify({'prediction': prediction.tolist()})

@app.route('/reset', methods=['POST'])
def reset_metrics():
    """Reset all Prometheus metrics."""
    REGISTRY.reset()  # Reset all metrics
    return jsonify({'message': 'Metrics reset successfully'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)