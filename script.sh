#!/bin/bash

# Define color codes for better readability
RED='\033[0;31m'       # Red for errors
GREEN='\033[0;32m'     # Green for success
YELLOW='\033[0;33m'    # Yellow for process steps
CYAN='\033[0;36m'      # Cyan for section headers
NC='\033[0m'           # Reset color

export OUTPUT_SIZE=500

# ================================
# STEP 1: Sampling Process
# ================================
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"
echo -e "▶ ${CYAN}STARTING SAMPLING PROCESS ($OUTPUT_SIZE records)${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"

python sample.py || { echo -e "✖ ${RED}Sampling process failed.${NC}"; exit 1; }

# Define output directories
OUTPUT_FOLDER="output/pre/"
POST_OUTPUT_FOLDER="output/post"

# Define cluster sizes (modifiable)
CLUSTER_SIZES="3,5,7,10"

# Convert cluster sizes string into an array
IFS=',' read -r -a CLUSTER_LIST <<< "$CLUSTER_SIZES"

# ================================
# STEP 2: Clustering Process
# ================================
echo -e "\n${CYAN}═════════════════════════════════════════════════════════${NC}"
echo -e "▶ ${CYAN}STARTING CLUSTERING PROCESS${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"

# Ensure output directory exists
mkdir -p "$OUTPUT_FOLDER"
echo -e "✔ ${GREEN}Output directory ensured: $OUTPUT_FOLDER${NC}"

# Declare associative arrays
declare -A cluster_files
declare -A input_output_files

# Generate cluster filenames dynamically
for num_clusters in "${CLUSTER_LIST[@]}"; do
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
        *) cluster_name="$num_clusters" ;;
    esac

    cluster_files[$num_clusters]="${cluster_name}_cluster.csv"
    input_output_files["$OUTPUT_FOLDER${cluster_name}_cluster.csv"]="${cluster_name}_edges.csv"
done

echo -e "▪ ${YELLOW}Cluster files initialized: ${!cluster_files[@]}${NC}"

# Execute clustering for each cluster size
for num_clusters in "${CLUSTER_LIST[@]}"; do
    export NUM_CONTAINERS=$num_clusters
    export OUTPUT_FOLDER=$OUTPUT_FOLDER
    export OUTPUT_FILE="${cluster_files[$num_clusters]}"

    echo -e "\n${YELLOW}-------------------------------------------------${NC}"
    echo -e "▶ ${YELLOW}Running clustering for $NUM_CONTAINERS clusters...${NC}"
    
    python cluster.py
    if [[ $? -eq 0 ]]; then
        echo -e "✔ ${GREEN}Clustering completed: $NUM_CONTAINERS clusters.${NC}"
    else
        echo -e "✖ ${RED}Clustering failed for $NUM_CONTAINERS clusters.${NC}"
        exit 1
    fi
    echo -e "${YELLOW}-------------------------------------------------${NC}"
done

echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"
echo -e "✔ ${CYAN}CLUSTERING PROCESS COMPLETED${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"

# ================================
# STEP 3: MEC Processing
# ================================
echo -e "\n${CYAN}═════════════════════════════════════════════════════════${NC}"
echo -e "▶ ${CYAN}STARTING MEC PROCESS${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"

# Ensure post-processing directory exists
mkdir -p "$POST_OUTPUT_FOLDER"
echo -e "✔ ${GREEN}Output directory ensured: $POST_OUTPUT_FOLDER${NC}"

# Run MEC processing for each cluster
for input_file in "${!input_output_files[@]}"; do
    export INPUT_FILE=$input_file
    export OUTPUT_FOLDER=$POST_OUTPUT_FOLDER
    export OUTPUT_FILE="${input_output_files[$input_file]}"

    echo -e "\n${YELLOW}-------------------------------------------------${NC}"
    echo -e "▶ ${YELLOW}Processing MEC for: $INPUT_FILE → $OUTPUT_FILE${NC}"
    
    python mec.py
    if [[ $? -eq 0 ]]; then
        echo -e "✔ ${GREEN}MEC processing completed: $INPUT_FILE${NC}"
    else
        echo -e "✖ ${RED}MEC processing failed: $INPUT_FILE${NC}"
        exit 1
    fi
    echo -e "${YELLOW}-------------------------------------------------${NC}"
done

echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"
echo -e "✔ ${CYAN}MEC PROCESS COMPLETED${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"

# ================================
# STEP 4: Visualization
# ================================
echo -e "\n${CYAN}═════════════════════════════════════════════════════════${NC}"
echo -e "▶ ${CYAN}STARTING VISUALIZATION PROCESS${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"

python visualize.py
if [[ $? -eq 0 ]]; then
    echo -e "✔ ${GREEN}Visualization completed successfully.${NC}"
else
    echo -e "✖ ${RED}Visualization failed.${NC}"
    exit 1
fi

echo -e "\n✔ ${GREEN}All processes completed successfully.${NC}\n"
