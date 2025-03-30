# **Optimizing Resource Allocation in Multi-Access Edge Computing for 5G Networks Using Blockchain-Based Validation**

## **Introduction**
This project presents a novel **ML-driven resource allocation framework** optimized for **Multi-Access Edge Computing (MEC) in 5G networks**. It integrates **LightGBM-based predictive modeling**, **adaptive network slicing**, and **blockchain-secured logging** to optimize resource allocation and enhance network efficiency. The key features include:

- **Intelligent Resource Allocation:** Uses **LightGBM** with **Bayesian Optimization** to enhance predictive accuracy for **bandwidth and resource allocation**.
- **Optimized Clustering for Edge Computing:** Implements **DBSCAN, K-Means, GMM, Hierarchical, MeanShift, OPTICS, and Divisive clustering** for **efficient edge node placement and user grouping**.
- **Adaptive QoS Slicing:** Dynamically **adjusts bandwidth** for **eMBB, URLLC, and mMTC network slices** to ensure efficient resource distribution.
- **Blockchain-Backed Security:** Implements **smart contracts** to **ensure transparency and immutability** in resource allocation decisions.

---

## **Prerequisites**
Before running the project, ensure you have the following tools installed:

- **Python 3.8+**
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **VS Code** or any preferred IDE
- **Docker & Docker Compose** (for MEC deployment)
- **Ganache** (for local blockchain simulation)
- **Truffle** (for smart contract deployment)

### **Setup Instructions**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## **1. User Data Clustering**

### **1.1 Generate Spatial Coordinates**

```bash
python simulate_distance.py
```

- **Input:** `augmented_dataset.csv` (Signal_Strength, Latency)  
- **Action:** Computes user distances using signal + latency metrics, generates x_coordinate and y_coordinate
- **Output:** `simulated_dataset.csv`

---

### **1.2 Cluster Users for Edge Placement**

```bash
python clustering_algorithms.py
```

- **Input:** `simulated_dataset.csv`
- **Algorithms Tested:** K-Means, DBSCAN, Hierarchical, MeanShift, OPTICS, GMM, Divisive  
- **Actions:** Clusters users into **2–20 edge nodes**, updates metrics (**distance, signal strength, latency**), assigns network slices (**eMBB/URLLC/mMTC**)
- **Output:** `output/pre/<algorithm>/<n>_cluster.csv` (e.g., `output/pre/kmeans/10_cluster.csv`)

---

## **2. Model Training**

### **2.1 Evaluate ML Models**

```bash
python model_eval.py
```

- **Input:** `augmented_dataset.csv`  
- **Models Compared:** 8 algorithms (**LightGBM, XGBoost, Random Forest, etc.**)
- **Output:** `model_performance_metrics.csv` (**RMSE, R², MAE**), plots in `/graphs/model_output/`

### **2.2 Train LightGBM Models**

```bash
python model.py
```

- **Output:** `model1.txt` (**Allocated_Bandwidth predictor**), `model2.txt` (**Resource_Allocation predictor**)

---

## **3. Blockchain Setup**

### **3.1 Local Blockchain Deployment**

- **Install Ganache:** Download from [trufflesuite.com/ganache](https://trufflesuite.com/ganache), run with `http://localhost:7545`

```bash
truffle migrate --network ganache
```

- **Key Outputs:** Contract Address: `0x123...abc`, ABI extracted from `build/contracts/YourContractName.json`

- **Update Config:**

```json
{
  "ganache_url": "http://host.docker.internal:7545",
  "contract_address": "0x123...abc",
  "abi": [...]
}
```

---

## **4. MEC Setup with Docker**

### **4.1 Build & Deploy Edge Containers**

```bash
cd docker
docker build -t lightgbm-container .
docker-compose up -d
```

- **Verification:** `docker ps` (should show `edge_1` running)
- **Access:** API: `http://localhost:5001`, Metrics: `http://localhost:8001`

---

## **5. Data Processing & Evaluation**

### **5.1 Run MEC Simulation**

```bash
python run_mec.py
```

- **Input:** Cluster files from `output/pre/<algorithm>/`
- **Outputs:**
  - Allocations: `output/post/<algorithm>/<n>_edge.csv`
  - Performance: `output/metrics/<algorithm>/<n>_metrics.txt`
  - Docker Stats: `output/docker_metrics/<algorithm>/<n>_docker_metrics.txt`

### **5.2 Verify Results**

```bash
docker logs edge_1
```

- **Blockchain Verification:** Match transaction hashes in **Ganache UI → Transactions tab**

---

## **6. Visualization**

### **6.1 Generate Plots**

```bash
python cluster_compare.py
```
- Generates scatter plots in `graphs/results/cluster_plots/`

```bash
python cluster_visualize.py
```
- Produces line charts in `graphs/results/cluster_comparison/`

```bash
python visualize.py
```
- Saves bar charts in `graphs/results/decentralization/`

```bash
python metrics_visualize.py
```
- Outputs request-response time graphs in `graphs/results/metrics/request_response_time.png`

```bash
python docker_metrics.py
```
- Generates CPU and RAM plots in `graphs/results/metrics/`

```bash
python blockchain_visualize.py
```
- Produces transaction-related graphs stored in `graphs/blockchain/`

---