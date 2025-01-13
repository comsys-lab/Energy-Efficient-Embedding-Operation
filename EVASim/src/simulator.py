from Helper import Helper
from ReqGenerator import ReqGenerator
from MemSpad import MemSpad
from MemCache import MemCache
import argparse
import sys
import numpy as np
import os

## Credit: Original code from Rishabh; Assisting the args parser
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

## Credit: Original code from Rishabh
def print_general_config(nbatches, n_format, bsz, table_config, emb_dim, lookups_per_sample, fname):
    emb_config = np.fromstring(table_config, dtype=int, sep="-")
    emb_config = np.asarray(emb_config, dtype=np.int32)
    print("\n************************************")
    print("* General Simulation Configuration *")
    print("************************************")
    print("Dataset: {}".format(fname))
    print("Numeric format: {} bits".format(str(n_format*8)))
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
    
    parser = argparse.ArgumentParser(description="EVASim")
    # memory config
    parser.add_argument("--memory-config", type=str, default="spad_naive")
    
    # emb related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=128)
    parser.add_argument("--arch-embedding-size", type=dash_separated_ints, default="500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000")

    # execution and dataset related parameters
    parser.add_argument("--data-generation", type=str, default="./EVASim/datasets/reuse_high/table_1M.txt")
    parser.add_argument("--numeric-format-bits", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--output-name", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lookups-per-sample", type=int, default=150)
    
    # argparses
    args = parser.parse_args()
    mem_config_file = args.memory_config
    n_format_bits = args.numeric_format_bits
    n_format = int(np.ceil(n_format_bits / 8))
    nbatches = args.num_batches
    embsize = args.arch_embedding_size
    emb_dim = args.arch_sparse_feature_size #embedding dim
    bsz = args.batch_size # batch size
    fname = args.data_generation
    num_indices_per_lookup = args.lookups_per_sample # pooling factor or lookups per sample
    # Parse the memory config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mem_config_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 
                                  'EVASim', 'configs', 
                                  f'{mem_config_file}.config')
    
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
            elif key.strip() == 'access_granularity':
                mem_gran = int(value.strip()) # B
            if mem_type == "cache":
                if key.strip() == 'cache_way':
                    cache_way = int(value.strip())
                elif key.strip() == 'cache_line_size':
                    # cache_line_size = int(value.strip())
                    cache_line_size = mem_gran
        cache_config = [cache_way, cache_line_size]    

    # these are for convenience...
    emb_config = np.fromstring(embsize, dtype=int, sep="-")
    emb_config = np.asarray(emb_config, dtype=np.int32)
    num_tables = len(emb_config)
    vectors_per_table = emb_config[0]
    
    helper = Helper()
    
    #-------------------------------------------------------------------
    
    ################################
    ### Create request generator ###
    ################################
    
    helper.set_timer()
    reqgen = ReqGenerator(nbatches, n_format, embsize, emb_dim, bsz, fname, num_indices_per_lookup, mem_gran)
    reqgen.data_gen()
    
    print_general_config(reqgen.nbatches, reqgen.n_format, reqgen.bsz, reqgen.embsize, reqgen.emb_dim, reqgen.num_indices_per_lookup, reqgen.fname)

    helper.end_timer("model and data gen")
    
    #-------------------------------------------------------------------
    
    ######################################
    ### Convert indices to memory addr ###
    ######################################
    
    helper.set_timer()
    reqgen.index_to_addr()
    emb_dataset = reqgen.addr_trace
    print("len(emb_dataset): {}".format(len(emb_dataset)))
    print("len(emb_dataset[0]): {}".format(len(emb_dataset[0])))
    print("emb_dataset[0][0].shape: {}".format(emb_dataset[0][0].shape))
    helper.end_timer("address generation")
    
    #-------------------------------------------------------------------
    
    ###############################
    ### Create memory structure ###
    ###############################
    
    helper.set_timer()    
    
    # Create mem_struct
    if mem_type == "spad":
        mem_struct = MemSpad(mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran)
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