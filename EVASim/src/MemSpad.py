import numpy as np
import time
import torch
import itertools
import random
from collections import OrderedDict, Counter
from tqdm import tqdm
from Helper import print_styled_header, print_styled_box

class MemSpad:
    def __init__(self, mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran):
        self.mem_size = 0 ### KB
        self.mem_type = "init"
        self.mem_gran = 0
        self.on_mem = np.ones(1)
        self.spad_size = 0
        self.batch_counter = 0 ### this is only for spad_oracle
        self.table_counter = 0 ### this is only for spad_oracle
        
        ### below configs are related to the dataset
        self.emb_dim = 0 # this is for spad
        self.emb_dataset = np.ones(1)
        self.num_tables = 0
        self.vectors_per_table = 0
        
        self.access_results = []
               
        self.set_params(mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran)
        
    def set_params(self, mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran):
        self.mem_size = mem_size * 1024 # KB -> Byte
        self.mem_type = mem_type # spad or cache
        self.mem_gran = mem_gran
                
        ### below configs are related to the dataset
        self.emb_dim = emb_dim # this is for spad
        self.emb_dataset = emb_dataset # emb_dataset[numbatch][table][batchsz*lookuppersample]
        self.num_tables = len(self.emb_dataset[0])
        self.vectors_per_table = vectors_per_table
        self.elem_per_vector = int(self.emb_dim / self.mem_gran)
        
        self.spad_size = int(self.mem_size / self.emb_dim * self.elem_per_vector)
        
    def set_policy(self, policy):
        if (self.mem_type == "spad" and not policy.startswith("spad_")):
            assert False, f"Invalid policy: '{policy}' for mem_type: '{self.mem_type}'"
        self.mem_policy = policy        
        
    def print_config(self):
        content = [
            f"Memory size: {self.mem_size} B ({int(self.mem_size/1024/1024)} MB)",
            f"Memory type: {self.mem_type}",
            f"Memory policy: {self.mem_policy}"
        ]
        print_styled_box("On-Chip Memory Configuration", content)
        
    def print_sim(self):
        print_styled_header("Simulation Start")
        
    def create_on_mem(self):
        ### create on-chip memory data structure (spad or cache)
        self.on_mem = self.set_spad()
    
    def set_spad(self):
        if self.mem_policy == "spad_naive":
            on_mem_set = []
            counter = 0
            break_flag = False
            
            with tqdm(total=self.spad_size, desc="Setting spad") as pbar:
                for t_i in range(self.num_tables):
                    for v_i in range(self.vectors_per_table):
                        for d_i in range(self.elem_per_vector):
                            tbl_bits = t_i << int(np.log2(self.vectors_per_table) + np.log2(self.emb_dim))
                            vec_idx = v_i << int(np.log2(self.emb_dim))
                            dim_bits = self.mem_gran * d_i
                            this_addr = tbl_bits + vec_idx + dim_bits
                            on_mem_set.append(this_addr)
                            counter = counter + 1
                            if counter==self.spad_size:
                                break_flag = True
                                break
                            pbar.update(1)
                        if break_flag:
                            break
                    if break_flag:
                        break
            print("[DEBUG] on_mem has {} elements.".format(counter))
            on_mem_set = np.asarray(on_mem_set, dtype=np.int64)
        
        elif self.mem_policy == "spad_random":
            on_mem_set = []
            ### Randomly store the data from the available address space until the on-chip memory becomes full.            
            avail_space = list(itertools.product(range(self.num_tables), range(self.vectors_per_table)))
            random.shuffle(avail_space)
            ### avail_space = avail_space[:self.spad_size]
            avail_space = avail_space[:int(self.spad_size/self.elem_per_vector)]
            with tqdm(total=self.spad_size, desc="Setting spad") as pbar:
                for pair in avail_space:
                    for d_i in range(self.elem_per_vector):
                        # address generation
                        tbl_bits = pair[0] << int(np.log2(self.vectors_per_table) + np.log2(self.emb_dim))
                        vec_idx = pair[1] << int(np.log2(self.emb_dim))
                        dim_bits = self.mem_gran * d_i
                        this_addr = tbl_bits + vec_idx + dim_bits
                        on_mem_set.append(this_addr)
                        pbar.update(1)
            print("[DEBUG] on_mem has {} elements.".format(len(avail_space)))
            on_mem_set = np.asarray(on_mem_set, dtype=np.int64)
        
        elif self.mem_policy == "spad_oracle":
            ### flatten the dataset -> count and sort the access frequency of each memory address
            flat_dataset = itertools.chain.from_iterable(itertools.chain.from_iterable([self.emb_dataset[self.batch_counter]]))
            
            access_freq = Counter(flat_dataset)
            access_freq = access_freq.most_common()
            ### store the memory addresses in the spad
            access_freq = access_freq[:min(self.spad_size, len(access_freq))]
            on_mem_set = np.array([x[0] for x in access_freq], dtype = np.int64)
            # print(len(access_freq))
            # print(access_freq[0])
            # print(access_freq[-1])
            # print(on_mem_set.shape)
            # exit()
        
        return on_mem_set
    
    def do_simulation(self):
        # Simulation
        self.print_sim()
        for nb in range(len(self.emb_dataset)): # recall that self.emb_dataset[numbatch][table][batchsz*lookuppersample]
            num_hit = 0
            num_miss = 0
            
            print("Simulation for batch {}...".format(nb))
            with tqdm(total=len(self.emb_dataset[nb]), desc="Simulation") as pbar:
                for nt in range(len(self.emb_dataset[nb])):                           
                    hit_mask = np.isin(self.emb_dataset[nb][nt], self.on_mem)  # hit_mask is a boolean array between table_data and self.on_mem
                    num_hit += np.sum(hit_mask) 
                    num_miss += np.sum(~hit_mask)
                    
                    # if self.mem_policy == "spad_oracle":
                        ### Table-wise oracular profiling
                        # self.table_counter = min(self.table_counter + 1, len(self.emb_dataset[nb])-1)
                        # self.on_mem = self.set_spad()
                    
                    pbar.update(1)
                    
                ### Batch-wise oracular profiling
                if self.mem_policy == "spad_oracle":
                    self.batch_counter = min(self.batch_counter + 1, len(self.emb_dataset)-1)
                    self.on_mem = self.set_spad()
            
            self.access_results.append([num_hit, num_miss]) # add the results for each batch
            
        print("Simulation Done")
        self.print_stats()
        
    def print_stats(self):
        total_hits = 0
        total_miss = 0
        for i in range(len(self.access_results)):
            total_hits += self.access_results[i][0]
            total_miss += self.access_results[i][1]
        total_hit_ratio = total_hits / (total_hits + total_miss)
        
        content = [
            f"Total hit ratio: {total_hit_ratio:.4f}",
            f"Total accesses: {total_hits+total_miss}",
            f"Total hits: {total_hits}",
            f"Total misses: {total_miss}",
            "",
            "Per batch results:"
        ]
        
        for i in range(len(self.access_results)):
            batch_hit_ratio = self.access_results[i][0] / (self.access_results[i][0] + self.access_results[i][1])
            content.append(
                f"[Batch {i}] hit ratio: {batch_hit_ratio:.4f} " +
                f"accesses: {self.access_results[i][0]+self.access_results[i][1]} " +
                f"hits: {self.access_results[i][0]} " +
                f"misses: {self.access_results[i][1]}"
            )
        
        print_styled_box("Simulation Results", content)