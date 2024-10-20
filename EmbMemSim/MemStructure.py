import numpy as np
import time
import torch

class ReqGenerator:
    def __init__(self):
               
        self.set_params()
        
    def set_params(self. mem_size, mem_type, embsize, cache_config):
        self.mem_size = mem_size * 1024 # KB -> Byte
        self.mem_type = mem_type # spad or cache
        self.embsize = embsize # this is for spad
        # below configs are only for cache configurations
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
        print("********************************")
        print("* On-Chip Memory Configuration *")
        print("********************************")
        print("Memory size: {} KB".format(self.mem_size))
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
            on_mem = 
        