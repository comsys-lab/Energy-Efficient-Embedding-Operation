#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name> <num_emb>"
    exit 1
fi

# Set variables with absolute path
DATASET_NAME=$1
NUM_EMB=$2
SCRIPT_DIR=$(dirname $(readlink -f $0))
DATASET_PATH="${SCRIPT_DIR}/datasets/profile_target/${DATASET_NAME}.txt"

# Remove existing executable if it exists
if [ -f "static_profile" ]; then
    rm static_profile
fi

# Compile with optimizations
g++ -O3 -march=native -fopenmp -flto -funroll-loops src/static_profile.cpp -o static_profile

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Run the program with absolute dataset path
./static_profile "${DATASET_PATH}" ${NUM_EMB}

