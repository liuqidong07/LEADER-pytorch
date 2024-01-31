# here put the import lib
import os
import time
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class Generator(object):

    def __init__(self, args, logger, device):

        self.args = args
        self.num_workers = args.num_workers
        self.bs = args.train_batch_size
        self.logger = logger
        self.device = device

        self.logger.info("Loading dataset ... ")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Dataset is loaded: consume %.3f s" % (end - start))

    
    def _load_dataset(self):

        return NotImplementedError

    
    def _split_dataset(self, data):
        '''Split the datatset based on the ratio 8:1:1'''
        index_list = list(range(len(data)))
        train_num, val_num = int(0.8 * len(data)), int(0.1 * len(data))

        # sequential split
        train_index = index_list[:train_num]
        val_index = index_list[train_num:train_num+val_num]
        test_index = index_list[train_num+val_num:]

        return [data[:train_num], data[train_num:train_num+val_num], data[train_num+val_num:]]


    def make_dataloaders(self):

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        eval_dataloader = DataLoader(self.eval_dataset,
                                     sampler=SequentialSampler(self.eval_dataset),
                                     batch_size=100,
                                     num_workers=self.num_workers)
        test_dataloader = DataLoader(self.test_dataset,
                                     sampler=SequentialSampler(self.test_dataset),
                                     batch_size=100,
                                     num_workers=self.num_workers)

        return train_dataloader, eval_dataloader, test_dataloader
    

    def get_tokenizer(self):

        if self.tokenizer:

            return self.tokenizer

        else:

            raise ValueError("Please initialize the generator firstly")
        
    
    def get_profile_tokenizer(self):

        if self.profile_tokenizer:

            return self.profile_tokenizer

        else:

            raise ValueError("Please initialize the profile tokenizer firstly")

    
    def get_statistics(self):

        return len(self.train_dataset), len(self.eval_dataset), len(self.test_dataset)



