#!/bin/bash
# Original code: https://github.com/rishucoding/reproduce_MICRO24_GPU_DLRM_inference

### outdir ### 
# OUT="results_ed_r_nt_lk_nb_bs"
OUT="results_Batchperiod0204"
mkdir -p $OUT
##############

### required params ###
workload_type="vectordb"
num_tables=1
num_indices_per_lookup=30000000
access_per_batch=7500000
bsz=1
mem_gran=128
n_format_byte=4
on_mem_access=36325
off_mem_access=7463675

# ### required params ###
# workload_type="dlrm"
# num_tables=512
# # num_indices_per_lookup=30000000
# access_per_batch=22282240
# bsz=128
# mem_gran=128
# n_format_byte=1
# on_mem_access=25240
# off_mem_access=22257000

### required params ###
# workload_type="dlrm"
# num_tables=1
# # num_indices_per_lookup=30000000
# access_per_batch=8388608
# bsz=1
# mem_gran=128
# n_format_byte=1
# on_mem_access=8388608
# off_mem_access=0

### energy estimation ###
python3 src/simulator_only_energy.py $workload_type $num_tables $access_per_batch $bsz $mem_gran $n_format_byte $on_mem_access $off_mem_access