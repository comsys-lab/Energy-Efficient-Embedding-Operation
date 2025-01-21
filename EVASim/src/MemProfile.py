import numpy as np
import time
import torch
import itertools
import random
from collections import OrderedDict, Counter
from tqdm import tqdm
from Helper import print_styled_header, print_styled_box

class MemProfile:
    def __init__(self, mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran, profiled_path):
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
        self.profiled_path = ""
        
        self.access_results = []
               
        self.set_params(mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran, profiled_path)
        
    def set_params(self, mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran, profiled_path):
        self.mem_size = mem_size * 1024 # KB -> Byte
        self.mem_type = mem_type # spad or cache
        self.mem_gran = mem_gran
                
        ### below configs are related to the dataset
        self.emb_dim = emb_dim # this is for spad
        self.emb_dataset = emb_dataset # emb_dataset[numbatch][table][batchsz*lookuppersample]
        self.num_tables = len(self.emb_dataset[0])
        self.vectors_per_table = vectors_per_table
        self.elem_per_vector = int(self.emb_dim / self.mem_gran)
        self.profiled_path = profiled_path
        
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
        if self.mem_policy == "profile_static":
            on_mem_set = []
            
            # open the profiled dataset file using the self.profiled_path
            with open(self.profiled_path, 'r') as f:
            # load the profiled dataset in to the on_mem_set array (load "self.spad_size" of elements)
                for i in range(self.spad_size):
                    line = f.readline()
                    on_mem_set.append(int(line))
                    # break if the end of the file is reached
                    if not line:
                        break                    
            on_mem_set = np.asarray(on_mem_set, dtype=np.int64)
            f.close()
            # [DEBUG] print the (number of entries in the on_mem_set) and (the first and the last element of the on_mem_set)
            print("[DEBUG] on_mem has {} elements.".format(len(on_mem_set)))
            print("[DEBUG] on_mem[0]: {}".format(on_mem_set[0]))
            print("[DEBUG] on_mem[-1]: {}".format(on_mem_set[-1]))                    
        
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