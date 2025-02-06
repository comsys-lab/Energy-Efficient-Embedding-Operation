#!/bin/bash

# Check if correct number of arguments is provided
# if [ "$#" -ne 10 ]; then
#     echo "Usage: $0 <dataset_name> <num_emb> <num_batch> <num_table> <batch_sz> <lookup_per_table> <emb_dim> <mem_gran> <n_format> <rows_per_table>"
#     exit 1
# fi

# Set variables with absolute path
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# argument
# DATASET_NAME=$1
# NUM_EMB=$2
# NUM_BATCH=$3
# NUM_TABLE=$4
# BATCH_SZ=$5
# LOOKUP_PER_TABLE=$6
# EMB_DIM=$7
# MEM_GRAN=$8
# N_FORMAT=$9
# ROWS_PER_TABLE=${10}

# DLRM
DATASET_NAME="dlrm/jain_train"
NUM_EMB=400000
NUM_BATCH=1
NUM_TABLE=512
BATCH_SZ=128
LOOKUP_PER_TABLE=170
EMB_DIM=256
MEM_GRAN=128
N_FORMAT=1
ROWS_PER_TABLE=1000000


# # VDB
# DATASET_NAME="vectordb/spacev250m_train_1B"
# NUM_EMB=1000000
# NUM_BATCH=1
# NUM_TABLE=1
# BATCH_SZ=1
# LOOKUP_PER_TABLE=10000000
# EMB_DIM=100
# MEM_GRAN=128
# N_FORMAT=4
# ROWS_PER_TABLE=250000000

# Convert input filename format (replace '-' with '/')
PROCESSED_FILENAME=$(echo $DATASET_NAME | tr '-' '/')
DATASET_PATH="${SCRIPT_DIR}/datasets/${PROCESSED_FILENAME}.txt"

# Remove existing executable if it exists
if [ -f "static_profile" ]; then
    rm static_profile
fi

# Compile with optimizations
g++ -O3 -march=native -fopenmp -flto -funroll-loops -std=c++17 src/static_profile_addr.cpp -o static_profile

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Run the program with absolute dataset path and all parameters
./static_profile "${DATASET_PATH}" ${NUM_EMB} ${NUM_BATCH} ${NUM_TABLE} ${BATCH_SZ} ${LOOKUP_PER_TABLE} ${EMB_DIM} ${MEM_GRAN} ${N_FORMAT} ${ROWS_PER_TABLE}

