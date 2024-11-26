import numpy as np
import time
import torch
from tqdm import tqdm

## Original code: https://github.com/rishucoding/reproduce_MICRO24_GPU_DLRM_inference by RJ

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
    def __init__(self, nbatches, embsize, emb_dim, bsz, fname, num_indices_per_lookup):
        self.dataset_gen = None
        # sparse feature (sparse indices)
        self.lS_emb_offsets = []
        self.lS_emb_indices = []
        self.lS_o = []
        self.lS_i = []
        
        self.nbatches = 0
        self.embsize = 0
        self.emb_dim = 0
        self.bsz = 0
        self.fname = ""
        self.num_indices_per_lookup = 0
        
        self.set_params(nbatches, embsize, emb_dim, bsz, fname, num_indices_per_lookup)
        
    def set_params(self, nbatches, embsize, emb_dim, bsz, fname, num_indices_per_lookup):
        self.nbatches = nbatches
        self.embsize = embsize
        self.emb_dim = emb_dim
        self.bsz = bsz
        self.fname = fname
        self.num_indices_per_lookup = num_indices_per_lookup
        
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
            # self.lS_emb_offsets.append(torch.tensor(lS_batch_offsets, dtype=torch.int64))
            # self.lS_emb_indices.append(torch.tensor(lS_batch_indices, dtype=torch.int64))
            self.lS_emb_offsets.append(np.array(lS_batch_offsets, dtype=np.int64))
            self.lS_emb_indices.append(np.array(lS_batch_indices, dtype=np.int64))

        return (self.lS_emb_offsets, self.lS_emb_indices)

    def data_gen(self):
        ln_emb = np.fromstring(self.embsize, dtype=int, sep="-")
        ln_emb = np.asarray(ln_emb, dtype=np.int32)
        ln_bot = np.fromstring('256-128-' + str(self.emb_dim), dtype=int, sep="-")

        for j in range(0, self.nbatches):
            self.lS_emb_offsets, self.lS_emb_indices = self.trace_read_input_batch(ln_bot[0], ln_emb)
            self.lS_o.append(self.lS_emb_offsets)
            self.lS_i.append(self.lS_emb_indices)
            
    def index_to_addr(self):
        # convert indices in self.lS_i to memory address...
        ln_emb = np.fromstring(self.embsize, dtype=int, sep="-")
        ln_emb = np.asarray(ln_emb, dtype=np.int32)
        rows_per_table = ln_emb[0]
        for nb in range(len(self.lS_i)): # recall that self.lS_i[numbatch][table][batchsz*lookuppersample]
            print("Processing batch {}...".format(nb))
            with tqdm(total=len(self.lS_i[nb])*len(self.lS_i[nb][0]), desc="Processing") as pbar:
                for nt in range(len(self.lS_i[nb])):
                    for vec in range(len(self.lS_i[nb][nt])):
                        tbl_bits = nt << int(np.log2(rows_per_table) + np.log2(self.emb_dim))
                        vec_idx = self.lS_i[nb][nt][vec] << int(np.log2(self.emb_dim))
                        this_addr = tbl_bits + vec_idx
                        self.lS_i[nb][nt][vec] = this_addr
                        
                        pbar.update(1)

if __name__ == "__main__":
    reqgen = ReqGenerator()
    reqgen.data_gen()
