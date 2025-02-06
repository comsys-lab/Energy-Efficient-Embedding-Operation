import numpy as np
import time
import torch
import itertools
import random
from collections import OrderedDict, Counter
from LRUlist import LRUlist
from tqdm import tqdm
from Helper import print_styled_header, print_styled_box
from itertools import chain
from lru_cache import LRUCache

class MemProfile:
    def __init__(self, mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran, n_format_byte, profiled_path):
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
        self.profiled_path = "" # this is for profile_static
        
        ### this is for profile_dynamic_cache
        self.n_format_byte = 0
        
        self.access_results = []
        self.spad_load_results = []
               
        self.set_params(mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran, n_format_byte, profiled_path)
        
    def set_params(self, mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran, n_format_byte, profiled_path):
        self.mem_size = mem_size * 1024 # KB -> Byte
        self.mem_type = mem_type # spad or cache
        self.mem_gran = mem_gran
        
        ### this is for profile_dynamic_cache
        self.n_format_byte = n_format_byte
                
        ### below configs are related to the dataset
        self.emb_dim = emb_dim # this is for spad
        self.emb_dataset = emb_dataset # emb_dataset[numbatch][table][batchsz*lookuppersample]
        self.num_tables = len(self.emb_dataset[0])
        self.vectors_per_table = vectors_per_table
        # self.access_per_vector = np.ceil(self.emb_dim / self.mem_gran).astype(np.int32)  # Convert to 32-bit integer
        self.access_per_vector = np.ceil(self.emb_dim * self.n_format_byte / self.mem_gran).astype(np.int32)
        # print("[DEBUG] emb_dim: {} n_format_byte: {} mem_gran: {}".format(self.emb_dim, self.n_format_byte, self.mem_gran))
        # print("[DEBUG] access_per_vector: {} type of access_per_vector: {} ".format(self.access_per_vector, type(self.access_per_vector)))
        self.profiled_path = profiled_path
        
        self.spad_size = np.floor(self.mem_size / self.mem_gran).astype(np.int32)
    
    def set_index_trace(self, index_trace):
        self.index_trace = index_trace # lS_i
    
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
        if self.mem_policy == "profile_dynamic_cache":            
            # self.logger_size = int((self.mem_size / self.emb_dim) / self.n_format_byte) * self.access_per_vector # multiply access_per_vector to enable the vector-level LRU cache simulation
            self.logger_size = self.spad_size # access-level logging -> after all, the logger should be able to contain all the entries in the spad (vector-level logging is meaningless)
            self.logger = LRUCache(self.logger_size) # it simulates fully associative LRU cache
            # print the number of vectors that the logger can contain assuming that logger performs vector-level logging in real implementation (not in this simulation)
            print("[DEBUG] logger can contain {} vectors".format(int(self.logger_size / self.access_per_vector)))
        elif self.mem_policy == "profile_dynamic_count":
            # create a counter array, which has the same dimension as the emb_dataset
            self.counter_arr = np.zeros((len(self.index_trace), len(self.index_trace[0]), self.vectors_per_table), dtype=np.int64)
            self.counter_set = 0
        self.on_mem = self.set_spad()
    
    def set_spad(self):
        on_mem_set = []
        if self.mem_policy == "profile_static":
            try:
                with open(self.profiled_path, 'r') as f:
                    for i in range(self.spad_size):
                        line = f.readline().strip()
                        if not line:  # Check for empty line
                            break
                        on_mem_set.append(int(line))
                        
                on_mem_set = np.asarray(on_mem_set, dtype=np.int64)
                
                # Debug prints
                print("[DEBUG] on_mem has {} elements.".format(len(on_mem_set)))
                if len(on_mem_set) > 0:
                    print("[DEBUG] on_mem[0]: {}".format(on_mem_set[0]))
                    print("[DEBUG] on_mem[-1]: {}".format(on_mem_set[-1]))
                
            except FileNotFoundError:
                print(f"Error: Could not find profile file at {self.profiled_path}")
                raise
            except ValueError as e:
                print(f"Error: Invalid data in profile file - {e}")
                raise
                
        elif self.mem_policy == "profile_dynamic_cache":
            if self.logger.is_empty(): 
                print("[DEBUG] logger is empty. Set the spad with the naive method.")
                counter = 0
                break_flag = False
                
                with tqdm(total=self.spad_size, desc="Setting spad") as pbar:
                    for t_i in range(self.num_tables):
                        for v_i in range(self.vectors_per_table):
                            for d_i in range(self.access_per_vector):
                                bytes_per_vec = (self.emb_dim * self.n_format_byte - 1).bit_length()
                                tbl_bits = t_i << int(np.log2(self.vectors_per_table-1)+1 + bytes_per_vec)
                                vec_idx = v_i << bytes_per_vec
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
                on_mem_set = np.asarray(on_mem_set, dtype=np.int64)
            else: # if the logger is not empty, set the spad with the entries in the logger
                print("[DEBUG] logger is not empty. Set the spad with the logger entries.")
                # slice the logger to the size of self.spad_size
                on_mem_set = self.logger.return_as_array()[:self.spad_size]
                print("[DEBUG] on_mem_set type: {}, shape: {}, dtype: {}".format(type(on_mem_set), on_mem_set.shape, on_mem_set.dtype))
                print("[DEBUG] on_mem_set[0]: {}, on_mem_set[-1]: {}".format(on_mem_set[0], on_mem_set[-1]))
        
        elif self.mem_policy == "profile_dynamic_count":
            if self.counter_set == 0: 
                print("[DEBUG] counter array is empty. Set the spad with the naive method.")
                counter = 0
                break_flag = False
                
                with tqdm(total=self.spad_size, desc="Setting spad") as pbar:
                    for t_i in range(self.num_tables):
                        for v_i in range(self.vectors_per_table):
                            for d_i in range(self.access_per_vector):
                                bytes_per_vec = (self.emb_dim * self.n_format_byte - 1).bit_length()
                                tbl_bits = t_i << int(np.log2(self.vectors_per_table-1)+1 + bytes_per_vec)
                                vec_idx = v_i << bytes_per_vec
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
                
                self.counter_set = 1
                on_mem_set = np.asarray(on_mem_set, dtype=np.int64)
                
            else:
                # Pre-allocate the final array
                on_mem_set = np.empty(self.spad_size, dtype=np.int64)
                
                # Get the top indices in one go
                vectors_needed = self.spad_size // self.access_per_vector
                flat_indices = np.argpartition(self.counter_arr.ravel(), -vectors_needed)[-vectors_needed:]
                
                # Calculate the actual addresses directly
                batch_indices = flat_indices // (self.vectors_per_table * len(self.index_trace[0]))
                temp = flat_indices % (self.vectors_per_table * len(self.index_trace[0]))
                table_indices = temp // self.vectors_per_table
                vector_indices = temp % self.vectors_per_table
                
                # Generate all addresses at once using broadcasting
                bytes_per_vec = (self.emb_dim * self.n_format_byte - 1).bit_length()
                dim_offsets = np.arange(self.access_per_vector) * self.mem_gran
                tbl_bits = (table_indices[:, None] << int(np.log2(self.vectors_per_table-1)+1 + bytes_per_vec))
                vec_idx = (vector_indices[:, None] << bytes_per_vec)
                
                # Final address calculation using broadcasting
                addresses = tbl_bits + vec_idx + dim_offsets
                print("[DEBUG] shape of addresses during dcount set_spad: {}".format(addresses.shape))
                on_mem_set = addresses.ravel()[:self.spad_size]
                print("[DEBUG] shape of on_mem_set during dcount set_spad: {}".format(on_mem_set.shape))
                
                return on_mem_set
        
        return on_mem_set
    
    def do_simulation(self):
        # Simulation
        self.print_sim()
        if self.mem_policy == "profile_dynamic_cache":
            self.do_simulation_dcache()
        elif self.mem_policy == "profile_dynamic_count":
            self.do_simulation_dcount()
        else:
            for nb in range(len(self.emb_dataset)): # recall that self.emb_dataset[numbatch][table][batchsz*lookuppersample]
                num_hit = 0
                num_miss = 0
                num_spad_load = 0
                
                print("Simulation for batch {}...".format(nb))
                with tqdm(total=len(self.emb_dataset[nb]), desc="Simulation") as pbar:
                    for nt in range(len(self.emb_dataset[nb])):                           
                        hit_mask = np.isin(self.emb_dataset[nb][nt], self.on_mem)  # hit_mask is a boolean array between table_data and self.on_mem
                        num_hit += np.sum(hit_mask) 
                        num_miss += np.sum(~hit_mask)
                        
                        pbar.update(1)
                                
                self.access_results.append([num_hit, num_miss]) # add the results for each batch
                self.spad_load_results.append(num_spad_load)
                
            print("Simulation Done")
            self.print_stats()
        
    def do_simulation_dcache(self):
        dynamic_counter = 0
        dynamic_counter_threshold_init = 10 #* self.spad_size
        vectors_in_batch = list(chain.from_iterable(self.emb_dataset[0]))
        
        dynamic_counter_threshold = max(len(vectors_in_batch), dynamic_counter_threshold_init) # number of vectors in the batch
        
        print("[DEBUG] dynamic_counter_threshold: {}".format(dynamic_counter_threshold))
        
        self.logger_results = [] # for DEBUG
        
        # print("[DEBUG] print the nb, nt, vec of self.emb_dataset {} {} {}".format(len(self.emb_dataset), len(self.emb_dataset[0]), len(self.emb_dataset[0][0])))
        
        for nb in range(len(self.emb_dataset)):
            num_hit = 0
            num_miss = 0
            num_spad_load = 0
            logger_hit = 0
            logger_miss = 0
            
            # nb=0 #DEBUG
            
            print("Simulation for batch {}...".format(nb))
            vectors_in_batch = list(chain.from_iterable(self.emb_dataset[nb]))
            with tqdm(total=len(vectors_in_batch), desc=f"Batch {nb}") as pbar:
                for vec in vectors_in_batch:
                    # Check cache hit or miss
                    is_hit = vec in self.on_mem
                    if is_hit:
                        num_hit += 1
                    else:
                        num_miss += 1
                    
                    # print("[DEBUG] Processing node {}...".format(vec))
                    # Update the logger
                    if not self.logger.search_and_access(vec):
                        # print("[DEBUG] Miss and Inserting node {} to the logger".format(vec))
                        self.logger.insert_node(vec)
                        logger_miss += 1
                    else:
                        # print("[DEBUG] Hit node {} in the logger".format(vec))
                        logger_hit += 1
                    
                    # periodically update the spad
                    dynamic_counter += 1
                    # print("[DEBUG] dynamic_counter: {}".format(dynamic_counter))
                    if dynamic_counter == dynamic_counter_threshold:
                        # print("[DEBUG] update spad / dynamic_counter: {}".format(dynamic_counter))
                        self.on_mem = self.set_spad()
                        num_spad_load += self.spad_size
                        dynamic_counter = 0
                    
                    pbar.update(1)
            
            self.access_results.append([num_hit, num_miss])
            self.spad_load_results.append(num_spad_load)
            self.logger_results.append([logger_hit, logger_miss])
            # print("[DEBUG] result appended for batch {}".format(nb))
        
        print("Simulation Done")
        self.print_stats()
    
    def do_simulation_dcount(self):
        dynamic_counter = 0
        dynamic_counter_threshold_init = 10 # * self.spad_size
        vectors_in_batch = list(chain.from_iterable(self.emb_dataset[0]))
        
        dynamic_counter_threshold = max(len(vectors_in_batch), dynamic_counter_threshold_init) # number of vectors in the batch
        
        print("[DEBUG] dynamic_counter_threshold: {}".format(dynamic_counter_threshold))
        
        for nb in range(len(self.emb_dataset)):
            num_hit = 0
            num_miss = 0
            num_spad_load = 0
            # nb=0 #DEBUG
            print("Simulation for batch {}...".format(nb))
            with tqdm(total=len(self.index_trace[0]) * len(self.index_trace[0][0]), desc=f"Batch {nb}") as pbar:
                for nt in range(len(self.emb_dataset[nb])):
                    for vec_ind in range(len(self.index_trace[nb][nt])):
                        # update the counter array using the index trace
                        this_vec_ind = self.index_trace[nb][nt][vec_ind]
                        self.counter_arr[nb][nt][this_vec_ind] += 1
                        
                        # print("[DEBUG] vector index {} is accessed {} times".format(this_vec_ind, self.counter_arr[nb][nt][this_vec_ind]))
                        
                        vec = vec_ind * self.access_per_vector
                        
                        # simluation is addr based
                        for dim in range(self.access_per_vector):
                            
                            if self.emb_dataset[nb][nt][vec + dim] in self.on_mem:
                                num_hit += 1
                            else:
                                num_miss += 1
                                
                            # periodically update the spad
                            dynamic_counter += 1
                            if dynamic_counter == dynamic_counter_threshold:
                                # print("[DEBUG] update spad / dynamic_counter: {}".format(dynamic_counter))
                                self.on_mem = self.set_spad()
                                num_spad_load += self.spad_size
                                dynamic_counter = 0
                            
                        pbar.update(1)
                
            self.access_results.append([num_hit, num_miss])
            self.spad_load_results.append(num_spad_load)
        
        print("Simulation Done")
        self.print_stats()
        
    def print_stats(self):
        # print("[DEBUG] len access_results2: {}".format(len(self.access_results)))
        total_hits = 0
        total_miss = 0
        for i in range(len(self.access_results)):
            total_hits += self.access_results[i][0]
            total_miss += self.access_results[i][1]
        total_hit_ratio = total_hits / (total_hits + total_miss)
        # print("[DEBUG] len access_results3: {}".format(len(self.access_results)))
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
            
            # print the spad load results per batch
            content.append(
                f"[Batch {i}] spad load: {self.spad_load_results[i]}"
            )
            
            # DEBUG: if the mem policy is dcache, append per-batch logger results and logger hit ratio
            if self.mem_policy == "profile_dynamic_cache":
                content.append(
                    f"[Batch {i}] logger hit ratio: {self.logger_results[i][0]/(self.logger_results[i][0]+self.logger_results[i][1]):.4f} " +
                    f"accesses: {self.logger_results[i][0]+self.logger_results[i][1]} " +
                    f"hits: {self.logger_results[i][0]} " +
                    f"misses: {self.logger_results[i][1]}"
                )
        
        print_styled_box("Simulation Results", content)