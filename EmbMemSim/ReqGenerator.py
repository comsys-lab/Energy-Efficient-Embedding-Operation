import numpy as np
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
def print_model_config(nbatches, bsz, table_config, emb_dim, lookups_per_sample, fname):
    print("Hotness: " + fname)
    print("Num batches: " + str(nbatches))
    print("Batch Size: " + str(bsz))
    print("Table config: " + str(table_config))
    print("Embedding Dimension " + str(emb_dim))
    print("Lookups per sample: " + str(lookups_per_sample))



class ReqGenerator:
    def __init__(self, nbatches, embsize, m_spa, bsz, fname, num_indices_per_lookup):
        self.dataset_gen = None
        # sparse feature (sparse indices)
        self.lS_emb_offsets = []
        self.lS_emb_indices = []
        self.lS_o = []
        self.lS_i = []
        
        self.nbatches = 0
        self.embsize = 0
        self.m_spa = 0
        self.bsz = 0
        self.fname = ""
        self.num_indices_per_lookup = 0
        
        self.set_params(nbatches, embsize, m_spa, bsz, fname, num_indices_per_lookup)
        
    def set_params(self, nbatches, embsize, m_spa, bsz, fname, num_indices_per_lookup):
        self.nbatches = nbatches
        self.embsize = embsize
        self.m_spa = m_spa
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
        # self.bsz,
        # self.num_indices_per_lookup,
        # self.fname,
    ):
        # dense feature <- for BMLP
        # Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32), dtype=torch.float32)
        #Xt = (ra.rand(n, m_den).astype(np.float32))
        # cur_gen = self.get_gen(self.fname, ln_emb[0])
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
            self.lS_emb_offsets.append(torch.tensor(lS_batch_offsets, dtype=torch.int64))
            self.lS_emb_indices.append(torch.tensor(lS_batch_indices, dtype=torch.int64))

        return (self.lS_emb_offsets, self.lS_emb_indices)




    ## --------------------------------------------------------------------


    def data_gen(self):
        ln_emb = np.fromstring(self.embsize, dtype=int, sep="-")
        ln_emb = np.asarray(ln_emb, dtype=np.int32)
        ln_bot = np.fromstring('256-128-' + str(self.m_spa), dtype=int, sep="-")

        print_model_config(self.nbatches, self.bsz, self.embsize, self.m_spa, self.num_indices_per_lookup, self.fname)

        start = time.perf_counter()
        for j in range(0, self.nbatches):
            # Xt, self.lS_emb_offsets, self.lS_emb_indices = trace_read_input_batch(ln_bot[0], ln_emb, self.bsz, self.num_indices_per_lookup, self.fname)
            # self.lS_emb_offsets, self.lS_emb_indices = self.trace_read_input_batch(ln_bot[0], ln_emb, self.bsz, self.num_indices_per_lookup, self.fname)
            self.lS_emb_offsets, self.lS_emb_indices = self.trace_read_input_batch(ln_bot[0], ln_emb)
            self.lS_o.append(self.lS_emb_offsets)
            self.lS_i.append(self.lS_emb_indices)

        end = time.perf_counter()
        print('Time elapsed(s) in model and data gen: {:10.6f}'.format(end-start))



if __name__ == "__main__":
    reqgen = ReqGenerator()
    reqgen.data_gen()
