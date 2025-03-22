import os
import subprocess
import re
import time
import threading

# Define directories
INPUT_DIR = "output/pre/divisive"
OUTPUT_DIR = "output/post/divisive"
METRICS_DIR = "output/metrics/divisive"
DOCKER_METRICS_DIR = "output/docker_metrics/divisive"

# Ensure output and metrics directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(DOCKER_METRICS_DIR, exist_ok=True)

def get_edge_count(input_file):
    """Extract the number of edges from the filename (e.g., 3_cluster.csv -> 3)."""
    base_name = os.path.basename(input_file)
    return int(re.sub(r"[^0-9]", "", base_name.split("_")[0]))

def collect_docker_metrics(container_names, interval=1):
    """Collect Docker container metrics periodically."""
    metrics = {name: {"cpu": [], "memory": []} for name in container_names}
    while getattr(threading.current_thread(), "do_run", True):
        for name in container_names:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}\t{{.MemUsage}}", name],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                cpu, memory = result.stdout.strip().split("\t")
                # Extract numeric CPU value (remove '%')
                cpu_value = float(cpu.replace("%", ""))
                # Extract numeric memory value (e.g., "129MiB" -> 129)
                memory_value = float(memory.split(" ")[0].replace("MiB", "").replace("GiB", ""))
                metrics[name]["cpu"].append(cpu_value)
                metrics[name]["memory"].append(memory_value)
        time.sleep(interval)
    return metrics  # Return the collected metrics

class DockerMetricsThread(threading.Thread):
    """Custom thread class to collect Docker metrics and store the result."""
    def __init__(self, container_names, interval=1):
        super().__init__()
        self.container_names = container_names
        self.interval = interval
        self.result = None  # Store the result here

    def run(self):
        """Override the run method to store the result."""
        self.result = collect_docker_metrics(self.container_names, self.interval)

def reset_metrics(edge_count):
    """Reset metrics on all edge nodes."""
    print("Resetting metrics on edge nodes...")
    for i in range(1, edge_count + 1):
        reset_url = f"http://localhost:{8000 + i}/reset"  # Correct URL formatting
        try:
            subprocess.run(
                ["curl", "-X", "POST", reset_url],
                capture_output=True, text=True
            )
            print(f"✅ Metrics reset on edge_{i}")
        except Exception as e:
            print(f"❌ Failed to reset metrics on edge_{i}: {e}")

def fetch_metrics_with_retry(metrics_url, retries=3, delay=2):
    """Fetch metrics with retries."""
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["curl", "-s", metrics_url],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
            else:
                print(f"Attempt {attempt + 1} failed: {result.stderr}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(delay)  # Wait before retrying
    return None  # Return None if all attempts fail

def main():
    # Loop through all cluster CSV files in the input directory
    for input_file in os.listdir(INPUT_DIR):
        if input_file.endswith("_cluster.csv"):
            input_path = os.path.join(INPUT_DIR, input_file)
            
            # Extract the base name (e.g., 2_cluster.csv -> 2)
            base_name = os.path.splitext(input_file)[0].replace("_cluster", "")
            
            # Get the number of edge nodes required for this file
            edge_count = get_edge_count(input_file)
            
            # Define the output file name (e.g., 2_edge.csv)
            output_file = os.path.join(OUTPUT_DIR, f"{base_name}_edge.csv")
            
            # Define the metrics file name (e.g., 2_metrics.txt)
            metrics_file = os.path.join(METRICS_DIR, f"{base_name}_metrics.txt")
            
            # Define the Docker metrics file name (e.g., 2_docker_metrics.txt)
            docker_metrics_file = os.path.join(DOCKER_METRICS_DIR, f"{base_name}_docker_metrics.txt")
            
            # Reset metrics on all edge nodes before processing the new input file
            reset_metrics(edge_count)
            
            # Set environment variables for the MEC script
            os.environ["INPUT_FILE"] = input_path
            os.environ["OUTPUT_FOLDER"] = OUTPUT_DIR
            os.environ["OUTPUT_FILE"] = os.path.basename(output_file)
            
            # Start Docker metrics collection in a separate thread
            container_names = [f"edge_{i}" for i in range(1, edge_count + 1)]
            docker_metrics_thread = DockerMetricsThread(container_names)
            docker_metrics_thread.start()
            
            # Run the MEC script
            print(f"Processing {input_path} -> {output_file} (using {edge_count} edge nodes)")
            subprocess.run(["python3", "mec.py"])
            
            # Stop Docker metrics collection
            docker_metrics_thread.do_run = False
            docker_metrics_thread.join()
            
            # Capture the Docker metrics
            docker_metrics = docker_metrics_thread.result  # Store the result of the thread
            
            # Check if the output file was created
            if os.path.exists(output_file):
                print(f"✅ Successfully created {output_file}")
                
                # Wait for edge nodes to become ready
                print("Waiting for edge nodes to become ready...")
                time.sleep(10)  # Adjust delay as needed
                
                # Fetch metrics from all edge nodes and save to the metrics file
                print(f"Collecting metrics for {input_path}...")
                with open(metrics_file, "w") as f:
                    for i in range(1, edge_count + 1):
                        metrics_url = f"http://localhost:{8000 + i}/metrics"  # Correct URL formatting
                        print(f"Fetching metrics from {metrics_url}...")
                        metrics_data = fetch_metrics_with_retry(metrics_url)
                        if metrics_data:
                            print(f"Metrics from edge_{i}:", file=f)
                            print(metrics_data, file=f)
                        else:
                            print(f"Failed to fetch metrics from edge_{i}", file=f)
                        print("", file=f)  # Add a newline for readability
                print(f"✅ Metrics saved to {metrics_file}")
                
                # Compute and save average Docker metrics
                print("Computing average Docker metrics...")
                with open(docker_metrics_file, "w") as f:
                    for name, data in docker_metrics.items():
                        avg_cpu = sum(data["cpu"]) / len(data["cpu"]) if data["cpu"] else 0
                        avg_memory = sum(data["memory"]) / len(data["memory"]) if data["memory"] else 0
                        print(f"Average CPU usage for {name}: {avg_cpu:.2f}%", file=f)
                        print(f"Average memory usage for {name}: {avg_memory:.2f} MiB", file=f)
                print(f"✅ Docker metrics saved to {docker_metrics_file}")
            else:
                print(f"❌ Failed to create {output_file}")
    
    print("All files processed.")

if __name__ == "__main__":
    main()