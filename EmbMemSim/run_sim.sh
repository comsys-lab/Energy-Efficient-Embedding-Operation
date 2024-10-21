#!/bin/bash
# Original code: https://github.com/rishucoding/reproduce_MICRO24_GPU_DLRM_inference
PyGenTbl='import sys; rows,tables=sys.argv[1:3]; print("-".join([rows]*int(tables)))'

data_path_dir="/home/choi/2nd/EmbMemSim/datasets"
dataset_list=("/reuse_high/table_1M.txt")

EMBS='128,500000,12,150' # EMBS= emb_dim, rows/table, num_tables, pooling_factor
# BOT_MLP=256-128-128
# TOP_MLP=128-64-1
NUM_BATCH=10 #8 batches are used for warmup and not accounted in the average ET.
BS=1024

for dataset in "${dataset_list[@]}"; do
    DATA_GEN_PATH=$data_path_dir$dataset
    for e in $EMBS; do
        IFS=','; set -- $e; EMB_DIM=$1; EMB_ROW=$2; EMB_TBL=$3; EMB_LS=$4; unset IFS;
        EMB_TBL=$(python -c "$PyGenTbl" "$EMB_ROW" "$EMB_TBL")
        python3 simulator.py --num-batches $NUM_BATCH --batch-size $BS --lookups-per-sample $EMB_LS --arch-sparse-feature-size $EMB_DIM --arch-embedding-size $EMB_TBL --data-generation=$DATA_GEN_PATH
    done
done
