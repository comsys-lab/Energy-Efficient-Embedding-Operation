import numpy as np
import time
import torch
from tqdm import tqdm

## We implement this module based on this code: https://github.com/rishucoding/reproduce_MICRO24_GPU_DLRM_inference by RJ

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

class ReqGenerator:
    def __init__(self, nbatches, n_format_byte, embsize, emb_dim, bsz, fname, num_indices_per_lookup, mem_gran):
        self.dataset_gen = None
        # sparse feature (sparse indices)
        self.lS_emb_offsets = []
        self.lS_emb_indices = []
        self.lS_o = []
        self.lS_i = []
        self.addr_trace = []
        
        self.nbatches = 0
        self.n_format_byte = 0
        self.embsize = 0
        self.emb_dim = 0
        self.bsz = 0
        self.fname = ""
        self.num_indices_per_lookup = 0
        
        self.mem_gran = 0
        
        self.set_params(nbatches, n_format_byte, embsize, emb_dim, bsz, fname, num_indices_per_lookup, mem_gran)
        
    def set_params(self, nbatches, n_format_byte, embsize, emb_dim, bsz, fname, num_indices_per_lookup, mem_gran):
        self.nbatches = nbatches
        self.n_format_byte = n_format_byte # numeric format bits -> bytes
        self.embsize = embsize
        self.emb_dim = emb_dim
        self.bsz = bsz
        self.fname = fname
        self.num_indices_per_lookup = num_indices_per_lookup
        self.mem_gran = mem_gran
        
        self.access_per_vector = np.ceil(self.emb_dim * self.n_format_byte / self.mem_gran).astype(np.int32)
        
    def open_gen(self, name, rows):
        with open(name) as f:
            idx = list(filter(lambda x: x < rows, map(int, f.readlines())))
        while True:
            for x in idx:
                yield x

    def get_gen(self, rows):
        if self.dataset_gen is None:
            self.dataset_gen = self.open_gen(self.fname, int(rows))
        return self.dataset_gen

    def trace_read_input_batch(
        self,
        m_den,
        ln_emb,
    ):
        cur_gen = self.get_gen(ln_emb[0])

        self.lS_emb_offsets = []
        self.lS_emb_indices = []
        #RJ: for each table
        for size in ln_emb:
            lS_batch_offsets = []
            lS_batch_indices = []
            offset = 0
            #RJ: goto each sample
            for _ in range(self.bsz):
                #pooling factor for each sample
                sparse_group_size = np.int64(self.num_indices_per_lookup)
                # store lengths and indices
                lS_batch_offsets += [offset]
                lS_batch_indices += [x for _, x in zip(range(sparse_group_size), cur_gen)]
                # update offset for next iteration
                offset += sparse_group_size
            self.lS_emb_offsets.append(np.array(lS_batch_offsets, dtype=np.int64))
            self.lS_emb_indices.append(np.array(lS_batch_indices, dtype=np.int64))

        return (self.lS_emb_offsets, self.lS_emb_indices)

    def data_gen(self):
        ln_emb = np.fromstring(self.embsize, dtype=int, sep="-") # memo ln_emb represents the number of tables
        ln_emb = np.asarray(ln_emb, dtype=np.int32) # shape of ln_emb is (num_tables,)
        ln_bot = np.fromstring('256-128-' + str(self.emb_dim), dtype=int, sep="-")

        for j in range(0, self.nbatches):
            self.lS_emb_offsets, self.lS_emb_indices = self.trace_read_input_batch(ln_bot[0], ln_emb)
            self.lS_o.append(self.lS_emb_offsets)
            self.lS_i.append(self.lS_emb_indices)
            
    def index_to_addr(self):
        # init addr_trace array
        self.addr_trace = [
            [np.ones(int(len(self.lS_i[0][0]) * self.access_per_vector), dtype=np.int64) 
             for _ in range(len(self.lS_i[0]))] 
            for _ in range(self.nbatches)
        ]
        print("[DEBUG] lS_i shape: {}".format(np.array(self.lS_i).shape))
        print("[DEBUG] addr_trace shape: {}".format(np.array(self.addr_trace).shape))
        
        # convert indices in self.lS_i to memory address...
        ln_emb = np.fromstring(self.embsize, dtype=int, sep="-")
        ln_emb = np.asarray(ln_emb, dtype=np.int32)
        rows_per_table = ln_emb[0]
        for nb in range(len(self.lS_i)): # recall that self.lS_i[numbatch][table][batchsz*lookuppersample]
            print("Converting vector indices into virtual memory addresses for batch {}...".format(nb))
            with tqdm(total=len(self.lS_i[nb])*len(self.lS_i[nb][0])*self.access_per_vector, desc="Processing") as pbar:
                for nt in range(len(self.lS_i[nb])):
                    for vec in range(len(self.lS_i[nb][nt])):
                        for dim in range(self.access_per_vector):
                            bytes_per_vec = (self.emb_dim * self.n_format_byte - 1).bit_length()
                            tbl_bits = nt << int(np.log2(rows_per_table-1)+1 + bytes_per_vec)                            
                            vec_idx = self.lS_i[nb][nt][vec] << bytes_per_vec
                            dim_bits = self.mem_gran * dim
                            this_addr = tbl_bits + vec_idx + dim_bits
                            
                            self.addr_trace[nb][nt][vec * self.access_per_vector + dim] = this_addr
                            # print(this_addr)
                        
                            pbar.update(1)

    def do_batch_access_pattern_analysis(self):
        # Convert first batch (batch 0) addresses into a list to maintain duplicates
        first_batch_addrs = []
        for table in self.addr_trace[0]:
            first_batch_addrs.extend(table)
        
        total_addrs = len(first_batch_addrs)  # Total number of addresses including duplicates
        first_batch_set = set(first_batch_addrs)  # Set for intersection
        
        print("\nBatch Access Pattern Analysis:")
        print("--------------------------------")
        print(f"Total addresses in each batch: {total_addrs}")
        print("Overlap percentages with first batch (Batch 0):")
        
        # Compare with all batches (including batch 0)
        for batch_idx in range(0, len(self.addr_trace)):
            current_batch_addrs = []
            for table in self.addr_trace[batch_idx]:
                current_batch_addrs.extend(table)
            
            # Calculate overlap using sets for unique addresses
            current_batch_set = set(current_batch_addrs)
            overlap_addrs = first_batch_set.intersection(current_batch_set)
            
            # Calculate percentage based on total addresses (including duplicates)
            overlap_count = sum(1 for addr in first_batch_addrs if addr in current_batch_set)
            overlap_percentage = (overlap_count / total_addrs) * 100
            
            print(f"Batch {batch_idx}: {overlap_percentage:.2f}%")

if __name__ == "__main__":
    reqgen = ReqGenerator()
    reqgen.data_gen()
