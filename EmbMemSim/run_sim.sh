#!/bin/bash
# Original code: https://github.com/rishucoding/reproduce_MICRO24_GPU_DLRM_inference

###################

# OUT="results_test" # "results_ed_r_nt_lk_nb_bs"
OUT="results_ed_r_nt_lk_nb_bs" # "results_ed_r_nt_lk_nb_bs"
data_path_dir="/home/choi/2nd/EmbMemSim/datasets/"

###################

mkdir -p $OUT
PyGenTbl='import sys; rows,tables=sys.argv[1:3]; print("-".join([rows]*int(tables)))'
# dataset_list=("reuse_high/table_1M.txt" "reuse_medium/table_1M.txt" "reuse_low/table_1M.txt")
dataset_list=("reuse_medium/table_1M.txt")

MEM_CFG=$1 # spad_naive

EMBS='128,1000000,10,150' # EMBS= emb_dim, rows/table, num_tables, pooling_factor 128, 1000000, 170, 180
NUM_BATCH=1 #8 batches are used for warmup and not accounted in the average ET.
BS=128 # 128

OUTDIR="$(echo "$EMBS" | sed 's/,/_/g')"
echo $OUTDIR
OUTDIR="${OUT}/${OUTDIR}_${NUM_BATCH}_${BS}"
echo $OUTDIR
mkdir -p $OUTDIR

for dataset in "${dataset_list[@]}"; do
    DATA_GEN_PATH=$data_path_dir$dataset
    # OUTFILE="$OUT$dataset"
    OUTFILE=$(echo "$dataset" | sed 's/\//_/g')
    OUTFILE="$OUTDIR/$OUTFILE"
    for e in $EMBS; do
        IFS=','; set -- $e; EMB_DIM=$1; EMB_ROW=$2; EMB_TBL=$3; EMB_LS=$4; unset IFS;
        EMB_TBL=$(python -c "$PyGenTbl" "$EMB_ROW" "$EMB_TBL")
        python3 simulator.py --num-batches $NUM_BATCH --batch-size $BS\
            --lookups-per-sample $EMB_LS --arch-sparse-feature-size $EMB_DIM\
            --arch-embedding-size $EMB_TBL --data-generation=$DATA_GEN_PATH --memory-config=$MEM_CFG | tee $(pwd)/${OUTFILE}_${MEM_CFG}.log
    done
done
