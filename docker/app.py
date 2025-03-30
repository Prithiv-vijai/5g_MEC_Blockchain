from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd
from prometheus_client import start_http_server, Summary, Gauge, Counter, REGISTRY
import time
from web3 import Web3
import json
import os

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

# Load blockchain configuration
with open('mec_config.json') as config_file:
    config = json.load(config_file)

# Initialize Web3
web3 = Web3(Web3.HTTPProvider(config['ganache_url']))
if not web3.is_connected():
    app.logger.warning("Failed to connect to Ganache. Blockchain logging will be disabled.")
    blockchain_enabled = False
else:
    blockchain_enabled = True
    contract = web3.eth.contract(
        address=config['contract_address'],
        abi=config['abi']
    )
    accounts = web3.eth.accounts
    CONTAINER_ID = int(os.getenv('CONTAINER_ID', 0))  # Default to account 0 if not set

def enforce_qos(slice_type, allocated_bandwidth, latency):
    """Enforce QoS policy based on slice type."""
    if slice_type == 'URLLC' and latency > 10:
        allocated_bandwidth *= 1 + ((latency - 10) / 10) * 0.1  
    elif slice_type == 'eMBB' and allocated_bandwidth < 10000:
        allocated_bandwidth *= 1 + ((10000 - allocated_bandwidth) / 10000) * 0.1  
    elif slice_type == 'mMTC' and allocated_bandwidth < 100:
        allocated_bandwidth *= 1 + ((100 - allocated_bandwidth) / 100) * 0.1  
    return allocated_bandwidth

def log_to_blockchain(user_id, allocated_bandwidth):
    """Log prediction to Ethereum blockchain."""
    if not blockchain_enabled:
        return None
        
    try:
        # Use account based on container ID
        account_index = min(CONTAINER_ID, len(accounts) - 1)
        web3.eth.default_account = accounts[account_index]
        
        tx_hash = contract.functions.addAllocation(
            int(user_id),
            int(float(allocated_bandwidth))
        ).transact()
        
        # Wait for transaction to be mined
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.transactionHash.hex()
    except Exception as e:
        app.logger.error(f"Blockchain error in container {CONTAINER_ID}: {str(e)}")
        return None

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
    user_id = input_data.get('user_id', '756')

    # Step 1: Predict new allocated bandwidth using Model 1
    prediction1 = model1.predict(user_data_1)
    new_allocated_bandwidth = prediction1[0] if len(prediction1) > 0 else 0

    # Step 2: Boost allocated bandwidth by 10% based on QoS
    new_allocated_bandwidth = enforce_qos(slice_type, new_allocated_bandwidth, updated_latency)

    # Step 3: Log to blockchain
    tx_hash = log_to_blockchain(user_id, new_allocated_bandwidth)

    # Step 4: Add new allocated bandwidth as a feature for Model 2
    user_data_2['Allocated_Bandwidth'] = new_allocated_bandwidth

    # Step 5: Predict new resource allocation using Model 2
    prediction2 = model2.predict(user_data_2)
    new_resource_allocation = prediction2[0] if len(prediction2) > 0 else 0

    # Calculate latency and response time
    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)
    REQUEST_RESPONSE_TIME.set(latency)

    # Increment throughput
    REQUEST_THROUGHPUT.inc()

    # Return both predictions
    return jsonify({
        'new_allocated_bandwidth': new_allocated_bandwidth,
        'new_resource_allocation': new_resource_allocation,
        'blockchain_tx_hash': tx_hash if blockchain_enabled else "blockchain_disabled",
        'container_id': CONTAINER_ID if blockchain_enabled else "none"
    })

@app.route('/reset', methods=['POST'])
def reset_metrics():
    """Reset all Prometheus metrics."""
    REGISTRY.reset()  # Reset all metrics
    return jsonify({'message': 'Metrics reset successfully'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)