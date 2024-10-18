from ReqGenerator import ReqGenerator
import argparse

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
def print_model_config(nbatches, bsz, table_config, emb_dim, lookups_per_sample, fname):
    print("Hotness: " + fname)
    print("Num batches: " + str(nbatches))
    print("Batch Size: " + str(bsz))
    print("Table config: " + str(table_config))
    print("Embedding Dimension " + str(emb_dim))
    print("Lookups per sample: " + str(lookups_per_sample))

if __name__ == "__main__":
    #-------------------------------------------------------------------
    
    #######################
    ### parse arguments ###
    #######################
    
    parser = argparse.ArgumentParser(description="DLRM Inference on CPU and GPUs")
    # emb related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=128)
    parser.add_argument("--arch-embedding-size", type=dash_separated_ints, default="500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000")

    # MLP related parameters
    #parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    #parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")

    # execution and dataset related parameters
    parser.add_argument("--data-generation", type=str, default="/home/choi/2nd/EmbMemSim/datasets/reuse_high/table_1M.txt")
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--output-name", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lookups-per-sample", type=int, default=150)

    args = parser.parse_args()

    nbatches = args.num_batches
    embsize = args.arch_embedding_size
    m_spa = args.arch_sparse_feature_size #embedding dim
    bsz = args.batch_size # batch size
    fname = args.data_generation
    num_indices_per_lookup = args.lookups_per_sample # pooling factor or lookups per sample
    
    #-------------------------------------------------------------------
    
    ################################
    ### Create request generator ###
    ################################
    
    reqgen = ReqGenerator(nbatches, embsize, m_spa, bsz, fname, num_indices_per_lookup)
    reqgen.data_gen()
    
    print("len(reqgen.lS_i): {}".format(len(reqgen.lS_i[0][0]))) # reqgen.lS_i[numbatch][table][batchsz*embdim]
    print("reqgen.lS_i[0][0]: {}".format(reqgen.lS_i[0][0]))
    print("reqgen.lS_i[1][0]: {}".format(reqgen.lS_i[1][0]))
    print("reqgen.lS_i[5][0]: {}".format(reqgen.lS_i[5][0]))
    print("reqgen.lS_i[0][0].shape: {}".format(reqgen.lS_i[0][0].shape))
    
    #-------------------------------------------------------------------
    
    ###############################
    ### Create memory structure ###
    ###############################
    
    #-------------------------------------------------------------------