from Helper import Helper
from ReqGenerator import ReqGenerator
from MemSpad import MemSpad
from MemCache import MemCache
import argparse
import sys
import numpy as np

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
def print_general_config(nbatches, bsz, table_config, emb_dim, lookups_per_sample, fname):
    emb_config = np.fromstring(table_config, dtype=int, sep="-")
    emb_config = np.asarray(emb_config, dtype=np.int32)
    print("\n************************************")
    print("* General Simulation Configuration *")
    print("************************************")
    print("Dataset: {}".format(fname))
    print("Num batches: {}".format(str(nbatches)))
    print("Num tables: {}".format(str(len(emb_config))))
    print("Batch Size (samples per batch): {}".format(str(bsz)))
    print("Vectors per table: {}".format(str(emb_config[0])))
    print("Lookups per sample: {}".format(str(lookups_per_sample)))
    print("Embedding Dimension {}".format(str(emb_dim)))
    print("************************************")

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
    
    # memory config
    parser.add_argument("--memory-config", type=str, default="spad_naive")

    args = parser.parse_args()

    nbatches = args.num_batches
    embsize = args.arch_embedding_size
    emb_dim = args.arch_sparse_feature_size #embedding dim
    bsz = args.batch_size # batch size
    fname = args.data_generation
    num_indices_per_lookup = args.lookups_per_sample # pooling factor or lookups per sample
    
    # these are for convenience...
    emb_config = np.fromstring(embsize, dtype=int, sep="-")
    emb_config = np.asarray(emb_config, dtype=np.int32)
    num_tables = len(emb_config)
    vectors_per_tables = emb_config[0]
    
    helper = Helper()
    
    #-------------------------------------------------------------------
    
    ################################
    ### Create request generator ###
    ################################
    
    helper.set_timer()
    reqgen = ReqGenerator(nbatches, embsize, emb_dim, bsz, fname, num_indices_per_lookup)
    reqgen.data_gen()
    
    print_general_config(reqgen.nbatches, reqgen.bsz, reqgen.embsize, reqgen.emb_dim, reqgen.num_indices_per_lookup, reqgen.fname)
    
    # print("len(reqgen.lS_i): {}".format(len(reqgen.lS_i[0][0]))) # reqgen.lS_i[numbatch][table][batchsz*lookuppersample]
    # print("reqgen.lS_i[0][0]: {}".format(reqgen.lS_i[0][0]))
    # print("reqgen.lS_i[1][0]: {}".format(reqgen.lS_i[1][0]))
    # print("reqgen.lS_i[5][0]: {}".format(reqgen.lS_i[5][0]))
    # print("reqgen.lS_i[0][0].shape: {}".format(reqgen.lS_i[0][0].shape))
    helper.end_timer("model and data gen")
    
    #-------------------------------------------------------------------
    
    ######################################
    ### Convert indices to memory addr ###
    ######################################
    
    helper.set_timer()
    reqgen.index_to_addr()
    helper.end_timer("address generation")
    
    #-------------------------------------------------------------------
    
    ###############################
    ### Create memory structure ###
    ###############################
    
    helper.set_timer()
    
    # Parse the memory config file and set the variables
    mem_config_dir = '/home/choi/2nd/EmbMemSim/configs/'
    mem_config_file = args.memory_config
    mem_config_path = mem_config_dir+mem_config_file+'.config'
    
    mem_type = None
    cache_way = 0
    cache_line_size = 0
    with open(mem_config_path, 'r') as mem_cfg:
        for cfg_line in mem_cfg:
            key, value = cfg_line.split(':')
            if key.strip() == 'mem_size':
                mem_size = int(value.strip()) # KB
            elif key.strip() == 'mem_type':
                mem_type = str(value.strip())
            elif key.strip() == 'policy':
                mem_policy = mem_type+'_'+str(value.strip())
            if mem_type == "cache":
                if key.strip() == 'cache_way':
                    cache_way = int(value.strip())
                elif key.strip() == 'cache_line_size':
                    cache_line_size = int(value.strip())
        cache_config = [cache_way, cache_line_size]
    emb_dataset = reqgen.lS_i
    
    # Create mem_struct
    if mem_type == "spad":
        mem_struct = MemSpad(mem_size, mem_type, emb_dim, emb_dataset, vectors_per_tables)
    elif mem_type == "cache":
        mem_struct = MemCache(mem_size, mem_type, cache_config, emb_dim, emb_dataset)
    mem_struct.set_policy(mem_policy)
    mem_struct.print_config()
    mem_struct.create_on_mem() # num_tables, num_rows_per_table
    # print("on_mem: {}, data structure size: {:.2f} KB".format(mem_struct.on_mem, sys.getsizeof(mem_struct.on_mem)/1024))
    print("on mem data structure size: {:.2f} KB".format(sys.getsizeof(mem_struct.on_mem)/1024))
    
    helper.end_timer("create memory structure")

    #-------------------------------------------------------------------
    
    ##########################
    ### Run the simulation ###
    ##########################
    
    helper.set_timer()
    mem_struct.do_simulation()
    helper.end_timer("do simulation")
    
    #-------------------------------------------------------------------