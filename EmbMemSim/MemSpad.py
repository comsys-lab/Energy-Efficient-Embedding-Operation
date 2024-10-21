import numpy as np
import time
import torch
from collections import OrderedDict

class MemSpad:
    def __init__(self, mem_size, mem_type, emb_dim, emb_dataset):
        self.mem_size = 0 # KB
        self.mem_type = "init"
        self.on_mem = np.ones(1)
        self.spad_size = 0
        
        # below configs are related to the dataset
        self.emb_dim = 0 # this is for spad
        self.emb_dataset = np.ones(1)
               
        self.set_params(mem_size, mem_type, emb_dim, emb_dataset)
        
    def set_params(self, mem_size, mem_type, emb_dim, emb_dataset):
        self.mem_size = mem_size * 1024 # KB -> Byte
        self.mem_type = mem_type # spad or cache
                
        # below configs are related to the dataset
        self.emb_dim = emb_dim # this is for spad
        self.emb_dataset = emb_dataset # emb_dataset[numbatch][table][batchsz*embdim]
        
        self.spad_size = int(self.mem_size / self.emb_dim)
        
    def set_policy(self, policy):
        if (self.mem_type == "spad" and not policy.startswith("spad_")):
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
        print("********************************")
        
    def create_on_mem(self):
        # create on-chip memory data structure (spad or cache)
        self.on_mem = self.set_spad()
    
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