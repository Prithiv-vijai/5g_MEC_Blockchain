#!/bin/bash

export OUTPUT_SIZE=10
echo "========================================================"
echo "STARTING SAMPLING PROCESS ($OUTPUT_SIZE data)"
echo "========================================================"

python sample.py


# Define output directory for the processed files
OUTPUT_FOLDER="output/pre/"
POST_OUTPUT_FOLDER="output/post"

# User input for cluster sizes (can be a comma-separated list)
CLUSTER_SIZES="3,7"  # You can provide any comma-separated list here

# Convert the CLUSTER_SIZES string into an array
IFS=',' read -r -a CLUSTER_LIST <<< "$CLUSTER_SIZES"

# ========================================================
# PROCESS 1: Running clustering script (cluster.py)
# ========================================================

echo "========================================================"
echo "STARTING CLUSTERING PROCESS"
echo "========================================================"

# Ensure the output directory exists (if it doesn't, it will be created)
mkdir -p "$OUTPUT_FOLDER"
echo "✔ Output folder '$OUTPUT_FOLDER' ensured."

# Dynamically create cluster_files and input_output_files arrays based on the cluster sizes
declare -A cluster_files
declare -A input_output_files

# Loop through each cluster size and dynamically create filenames
for num_clusters in "${CLUSTER_LIST[@]}"; do
    # Dynamically generate the cluster name (e.g., 3 -> three)
    case $num_clusters in
        2) cluster_name="two" ;;
        3) cluster_name="three" ;;
        4) cluster_name="four" ;;
        5) cluster_name="five" ;;
        6) cluster_name="six" ;;
        7) cluster_name="seven" ;;
        8) cluster_name="eight" ;;
        9) cluster_name="nine" ;;
        10) cluster_name="ten" ;;
        *) cluster_name="$num_clusters" ;;  # default to the number itself for other cases
    esac
    
    # Populate the cluster_files and input_output_files arrays dynamically
    cluster_files[$num_clusters]="${cluster_name}_cluster.csv"
    input_output_files["$OUTPUT_FOLDER${cluster_name}_cluster.csv"]="${cluster_name}_edges.csv"
done

echo "Cluster files and mappings generated: ${!cluster_files[@]}"

# Loop through each cluster size and process the corresponding CSV file using cluster.py
for num_clusters in "${CLUSTER_LIST[@]}"; do
    # Export the number of clusters and the corresponding file
    export NUM_CONTAINERS=$num_clusters
    export OUTPUT_FOLDER=$OUTPUT_FOLDER
    export OUTPUT_FILE="${cluster_files[$num_clusters]}"

    echo "--------------------------------------------------------"
    echo "Running clustering script for $NUM_CONTAINERS clusters..."
    python cluster.py

    if [[ $? -eq 0 ]]; then
        echo "✔ Clustering for $NUM_CONTAINERS clusters completed successfully."
    else
        echo "❌ Error occurred during clustering for $NUM_CONTAINERS clusters."
    fi
    echo "--------------------------------------------------------"
done

echo "========================================================"
echo "CLUSTERING PROCESS COMPLETED"
echo "========================================================"

# ========================================================
# PROCESS 2: Running mec.py for post-clustering processing
# ========================================================

echo "========================================================"
echo "STARTING MEC PROCESS"
echo "========================================================"

# Ensure the 'output/post' directory exists (if it doesn't, it will be created)
mkdir -p "$POST_OUTPUT_FOLDER"
echo "✔ Output folder '$POST_OUTPUT_FOLDER' ensured."

# Iterate over the input/output file pairs and run mec.py
for input_file in "${!input_output_files[@]}"; do
    export INPUT_FILE=$input_file
    export OUTPUT_FOLDER=$POST_OUTPUT_FOLDER
    export OUTPUT_FILE="${input_output_files[$input_file]}"

    echo "--------------------------------------------------------"
    echo "Running mec.py for $INPUT_FILE and output file $OUTPUT_FILE..."
    python mec.py
    echo "✔ Processing for $INPUT_FILE to $OUTPUT_FILE completed successfully."
    echo "--------------------------------------------------------"
done

echo "========================================================"
echo "MEC PROCESS COMPLETED"
echo "========================================================"


echo "========================================================"
echo "STARTING VISUALIZATION PROCESS"
echo "========================================================"

python visualize.py

echo "All processes completed"