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

def enforce_qos(slice_type, allocated_bandwidth, latency):
    """Enforce QoS policy based on slice type."""
    if slice_type == 'URLLC' and latency > 10:
        allocated_bandwidth *= 1 + ((latency - 10) / 10) * 0.1  
    elif slice_type == 'eMBB' and allocated_bandwidth < 10000:
        allocated_bandwidth *= 1 + ((10000 - allocated_bandwidth) / 10000) * 0.1  
    elif slice_type == 'mMTC' and allocated_bandwidth < 100:
        allocated_bandwidth *= 1 + ((100 - allocated_bandwidth) / 100) * 0.1  
    return allocated_bandwidth

@app.route('/predict', methods=['POST'])
def predict():
    """Perform prediction using Model 1 and Model 2 with QoS adjustments."""
    start_time = time.time()

    # Get input data
    input_data = request.get_json()
    user_data_1 = pd.DataFrame([input_data['user_data_1']])  # Convert JSON to DataFrame
    user_data_2 = pd.DataFrame([input_data['user_data_2']])  # Convert JSON to DataFrame
    slice_type = str(input_data['slice_type']).strip()
    updated_latency = input_data['updated_latency']
    signal_strength = input_data['signal_strength']
    updated_signal_strength = input_data['updated_signal_strength']

    # Apply the missing part: scale Required_Bandwidth and adjust signal strength
    user_data_1['Required_Bandwidth'] *= 0.8
    adjusted_signal_strength = signal_strength - abs(updated_signal_strength - signal_strength)
    user_data_1['Updated_Signal_Strength'] = adjusted_signal_strength

    # Step 1: Predict new allocated bandwidth using Model 1
    prediction1 = model1.predict(user_data_1)
    new_allocated_bandwidth = prediction1[0] if len(prediction1) > 0 else 0

    # Step 2: Boost allocated bandwidth by 10% based on QoS
    new_allocated_bandwidth = enforce_qos(slice_type, new_allocated_bandwidth, updated_latency)

    # Step 3: Add new allocated bandwidth as a feature for Model 2
    user_data_2['Allocated_Bandwidth'] = new_allocated_bandwidth

    # Step 4: Predict new resource allocation using Model 2
    prediction2 = model2.predict(user_data_2)
    new_resource_allocation = prediction2[0] if len(prediction2) > 0 else 0

    # Step 5: Cap resource allocation at 99 if necessary
    if new_resource_allocation > 99:
        new_resource_allocation = 99

    # Calculate latency and response time
    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)
    REQUEST_RESPONSE_TIME.set(latency)

    # Increment throughput
    REQUEST_THROUGHPUT.inc()

    # Return both predictions
    return jsonify({
        'new_allocated_bandwidth': new_allocated_bandwidth,
        'new_resource_allocation': new_resource_allocation
    })

@app.route('/reset', methods=['POST'])
def reset_metrics():
    """Reset all Prometheus metrics."""
    REGISTRY.reset()  # Reset all metrics
    return jsonify({'message': 'Metrics reset successfully'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
