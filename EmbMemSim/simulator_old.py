import numpy as np
import argparse
import time
import torch

## RJ: reading  the trace files to generate the datasets
## Mechansim: for a given trace file, we sequentially read the indices to assign for a
## batch for each table, and for all batches. We do a circular read if we reach the end
## of the trace file.


## Assisting the args parser
def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value

## Rishabh: helper function
def print_model_config(device, nbatches, n, table_config, emb_dim, lookups_per_sample, fname):
    print("Device: " + device)
    print("Hotness: " + fname)
    print("Num batches: " + str(nbatches))
    print("Batch Size: " + str(n))
    print("Table config: " + str(table_config))
    print("Embedding Dimension " + str(emb_dim))
    print("Lookups per sample: " + str(lookups_per_sample))

def open_gen(name, rows):
    with open(name) as f:
        idx = list(filter(lambda x: x < rows, map(int, f.readlines())))
    while True:
        for x in idx:
            yield x

dataset_gen = None

def get_gen(fname, rows):
    global dataset_gen
    if dataset_gen is None:
        dataset_gen = open_gen(fname, int(rows))
    return dataset_gen


def trace_read_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    fname,
):
    # dense feature <- for BMLP
    # Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32), dtype=torch.float32)
    #Xt = (ra.rand(n, m_den).astype(np.float32))
    cur_gen = get_gen(fname, ln_emb[0])

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    #RJ: for each table
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        #RJ: goto each sample
        for _ in range(n):
            #pooling factor for each sample
            sparse_group_size = np.int64(num_indices_per_lookup)
            # sparse indices to be used per embedding
            # r = ra.random(sparse_group_size)
            #sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
            # reset sparse_group_size in case some index duplicates were removed
            #sparse_group_size = np.int32(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += [x for _, x in zip(range(sparse_group_size), cur_gen)]
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets, dtype=torch.int64))
        lS_emb_indices.append(torch.tensor(lS_batch_indices, dtype=torch.int64))

    return (lS_emb_offsets, lS_emb_indices)




## --------------------------------------------------------------------


def data_gen():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="DLRM Inference on CPU and GPUs"
    )

    # emb related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=128)
    # parser.add_argument("--arch-embedding-size", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-embedding-size", type=dash_separated_ints, default="500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000")

    # MLP related parameters
    #parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    #parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")

    # execution and dataset related parameters
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--data-generation", type=str, default="/home/choi/2nd/EmbMemSim/datasets/reuse_high/table_1M.txt")
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--output-name", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lookups-per-sample", type=int, default=150)

    global args
    args = parser.parse_args()

    device = args.device
    nbatches = args.num_batches

    #ln_emb = [500000]*12;
    ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
    ln_emb = np.asarray(ln_emb, dtype=np.int32)
    m_spa = args.arch_sparse_feature_size #embedding dim
    n = args.batch_size # batch size
    fname = args.data_generation
    num_indices_per_lookup = args.lookups_per_sample # pooling factor or lookups per sample
    ln_bot = np.fromstring('256-128-' + str(m_spa), dtype=int, sep="-")
    lS_o = []
    lS_i = []

    print_model_config(device, nbatches, n, args.arch_embedding_size, m_spa, num_indices_per_lookup, fname)

    start = time.perf_counter()
    for j in range(0, nbatches):
        # Xt, lS_emb_offsets, lS_emb_indices = trace_read_input_batch(ln_bot[0], ln_emb, n, num_indices_per_lookup, fname)
        lS_emb_offsets, lS_emb_indices = trace_read_input_batch(ln_bot[0], ln_emb, n, num_indices_per_lookup, fname)
        lS_o.append(lS_emb_offsets)
        lS_i.append(lS_emb_indices)

    #emb_l, w_list = create_emb(m_spa, ln_emb)
    print("len(lS_i): {}".format(len(lS_i[0][0]))) # lS_i[numbatch][table][batchsz*embdim]
    print("lS_i[0][0]: {}".format(lS_i[0][0]))
    print("lS_i[1][0]: {}".format(lS_i[1][0]))
    print("lS_i[5][0]: {}".format(lS_i[5][0]))
    print("lS_i[0][0].shape: {}".format(lS_i[0][0].shape))

    end = time.perf_counter()
    print('Time elapsed(s) in model and data gen: {:10.6f}'.format(end-start))



if __name__ == "__main__":
    data_gen()
