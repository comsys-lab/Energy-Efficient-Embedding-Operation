import numpy as np
import time
import torch
from collections import OrderedDict
from LRUlist import LRUlist

class MemStructure:
    def __init__(self, mem_size, mem_type, cache_config, emb_dim, emb_dataset):
        self.mem_size = 0 # KB
        self.mem_type = "init"
        self.on_mem = np.ones(1)
        
        # below configs are related to the dataset
        self.emb_dim = 0 # this is for spad
        self.emb_dataset = np.ones(1)
        
        # below configs are only for spad configurations
        self.spad_size = 0
        
        # below configs are only for cache configurations
        self.cache_way = 0
        self.cache_line_size = 0
        self.cache_set = 0
        self.cache_tag_bits = 0
               
        self.set_params(mem_size, mem_type, cache_config, emb_dim, emb_dataset)
        
    def set_params(self, mem_size, mem_type, cache_config, emb_dim, emb_dataset):
        self.mem_size = mem_size * 1024 # KB -> Byte
        self.mem_type = mem_type # spad or cache
        
        # below configs are related to the dataset
        self.emb_dim = emb_dim # this is for spad
        self.emb_dataset = emb_dataset # emb_dataset[numbatch][table][batchsz*embdim]
        
        # below configs are only for cache configurations
        self.spad_size = int(self.mem_size / self.emb_dim)
        
        # below configs are only for cache configurations
        if self.mem_type == "cache":
            self.cache_way = cache_config[0] # cache_config = [way, line size]
            self.cache_line_size = cache_config[1]
            self.cache_set = int(self.mem_size / self.cache_line_size / self.cache_way)
            self.cache_tag_bits = 48 - int(np.log2(self.cache_set)) - 2 # 48 bits - index bits - byte offset
        
    def set_policy(self, policy):
        if (self.mem_type == "spad" and not policy.startswith("spad_")) or (self.mem_type == "cache" and not policy.startswith("cache_")):
            assert False, f"Invalid policy: '{policy}' for mem_type: '{self.mem_type}'"
        self.mem_policy = policy        
        
    def print_config(self):
        # print current configurations
        print("\n********************************")
        print("* On-Chip Memory Configuration *")
        print("********************************")
        print("Memory size: {} B ({} MB)".format(self.mem_size, int(self.mem_size/1024/1024)))
        print("Memory type: {}".format(self.mem_type))
        print("Memory policy: {}".format(self.mem_policy))
        if self.mem_type == "cache":
            print("Cache way: {}-way".format(self.cache_way))
            print("Cache line size: {} B".format(self.cache_line_size))
            print("Cache set: {} sets".format(self.cache_set))
            print("Cache tag bits: {} bits".format(self.cache_tag_bits))
        print("********************************")
        
    def create_on_mem(self):
        # create on-chip memory data structure (spad or cache)
        if self.mem_type == "spad":
            # on_mem = np.array(int(self.mem_size/self.emb_dim), dtype=np.int64) # This might be unnecessary.
            self.on_mem = self.set_spad()
        elif self.mem_type == "cache":
            # self.on_mem = np.array((self.cache_set, self.cache_way), dtype=np.int32)
            self.on_mem = [LRUlist(self.cache_way) for i in range(self.cache_set)]
    
    def set_spad(self):
        if self.mem_policy == "spad_naive":
            # Store the data from the first element of emb_dataset until the on-chip memory becomes full.
            emb_dataset_set = list(OrderedDict.fromkeys(torch.cat([torch.cat(inner_list) for inner_list in self.emb_dataset]).tolist()))
            on_mem_set = np.array(list(emb_dataset_set)[:self.spad_size], dtype=np.int64)
            return on_mem_set
        elif self.mem_policy == "spad_random":
            # Randomly store the data from the emb_dataset until the on-chip memory becomes full.
            dataset_no_dupl = set(self.emb_dataset.view(-1).tolist())
            on_mem_set = np.array(list(dataset_no_dupl), dtype=np.int64)
            on_mem_set = np.random.choice(on_mem_set, size=(self.mem_size / self.emb_dim), replace=False)
            return on_mem_set
        elif self.mem_policy == "spad_ideal":
            print("Working in progress")