# here put the import lib
import numpy as np
import pandas as pd
import pickle
import copy
import os
import random
import dill

import torch
from torch.utils.data import Dataset


class Voc(object):
    '''Define the vocabulary (token) dict'''

    def __init__(self):

        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        '''add vocabulary to dict via a list of words'''
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)



class EHRTokenizer(object):
    """The tokenization that offers function of converting id and token"""

    def __init__(self, voc_dir, special_tokens=("[PAD]", "[CLS]", "[MASK]")):

        self.vocab = Voc()  # this is a overall Voc

        # special tokens
        self.vocab.add_sentence(special_tokens)

        self.diag_voc, self.med_voc, self.pro_voc = self.read_voc(voc_dir)
        self.vocab.add_sentence(self.med_voc.word2idx.keys())
        self.vocab.add_sentence(self.diag_voc.word2idx.keys())
        self.vocab.add_sentence(self.pro_voc.word2idx.keys())

        self.attri_num = None
        self.hos_num = None
    

    def read_voc(self, voc_dir):

        with open(voc_dir, 'rb') as f:
            
            voc_dict = dill.load(f)
            
        return voc_dict['diag_voc'], voc_dict['med_voc'], voc_dict['pro_voc']


    def add_vocab(self, vocab_file):

        voc = self.vocab
        specific_voc = Voc()

        with open(vocab_file, 'r') as fin:
            for code in fin:
                voc.add_sentence([code.rstrip('\n')])
                specific_voc.add_sentence([code.rstrip('\n')])

        return specific_voc


    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token])
        return ids


    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.vocab.idx2word[i])
        return tokens
    

    def convert_med_tokens_to_ids(self, tokens):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        ids = []
        for i in tokens:
            ids.append(self.med_voc.word2idx[i])
        return ids
    

    def convert_med_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.med_voc.idx2word[i])
        return tokens



class EHRDataset(Dataset):
    '''The dataset for medication recommendation'''

    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len  # the maximum length of a diagnosis/procedure record

        self.sample_counter = 0
        self.records = data_pd

        self.var_name = []


    def __len__(self):

        return NotImplementedError

    def __getitem__(self, item):

        return NotImplementedError



####################################
'''Finetune Dataset'''
####################################

class FinetuneEHRDataset(EHRDataset):

    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        
        super().__init__(data_pd, tokenizer, max_seq_len)
        self.max_seq = 10
        self.var_name = ["diag_seq", "proc_seq", "med_seq", "seq_mask", "labels"]


    def __len__(self):

        return len(self.records)

    
    def __getitem__(self, item):

        # one admission: [diagnosis, procedure, medication]
        adm = copy.deepcopy(self.records[item])

        med_seq = [meta_adm[2] for meta_adm in adm]
        diag_seq = [meta_adm[0] for meta_adm in adm]
        proc_seq = [meta_adm[1] for meta_adm in adm]

        # get the medcation recommendation label -- multi-hot vector
        label_index = self.tokenizer.convert_med_tokens_to_ids(med_seq[-1])
        label = np.zeros(len(self.tokenizer.med_voc.word2idx))
        for index in label_index:
            label[index] = 1

        # get the seq len
        # pad the sequence to longest med / diag / proc sequences
        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l
        # convert raw tokens to unified ids
        med_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in med_seq]
        diag_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in diag_seq]
        proc_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in proc_seq]

        # pad the sequence to max possible records
        pad_seq = ["[PAD]" for _ in range(self.seq_len)]
        pad_seq = self.tokenizer.convert_tokens_to_ids(pad_seq)
        def fill_to_max_seq(l, seq):
            pad_num = 0
            while len(l) < seq:
                l.append(pad_seq)
                pad_num += 1
            if len(l) > seq:
                l = l[:seq]
            return l, pad_num
        med_seq = med_seq[:-1]  # remove the current medication set, which is label
        med_seq, _ = fill_to_max_seq(med_seq, self.max_seq)
        diag_seq, pad_num = fill_to_max_seq(diag_seq, self.max_seq)
        proc_seq, _ = fill_to_max_seq(proc_seq, self.max_seq)

        # get mask
        mask = np.ones(self.max_seq)
        if pad_num != 0:
            mask[-pad_num:] = 0

        return np.array(diag_seq, dtype=int), np.array(proc_seq, dtype=int), \
               np.array(med_seq, dtype=int), mask.astype(int), label.astype(float)



####################################
'''MedRec Dataset'''
####################################

class MedRecEHRDataset(EHRDataset):

    def __init__(self, data_pd, tokenizer: EHRTokenizer, profile_tokenizer, args):
        
        super().__init__(data_pd, tokenizer, args.max_seq_length)

        if args.filter:
            self._filter_data()

        self.max_seq = args.max_record_num
        self.profile_tokenizer = profile_tokenizer
        self.var_name = ["diag_seq", "proc_seq", "med_seq", "seq_mask", "labels", "multi_label", "profile"]


    def __len__(self):

        return len(self.records)

    
    def __getitem__(self, item):

        # one admission: [diagnosis, procedure, medication]
        adm = copy.deepcopy(self.records[item])

        med_seq = adm["records"]["medication"]
        diag_seq = adm["records"]["diagnosis"]
        proc_seq = adm["records"]["procedure"]

        # encode profile, get a vector to organize all feature orderly
        profile = []
        for k, v in adm["profile"].items():
            profile.append(self.profile_tokenizer["word2idx"][k][v])

        # get the medcation recommendation label -- multi-hot vector
        label_index = self.tokenizer.convert_med_tokens_to_ids(med_seq[-1])
        label = np.zeros(len(self.tokenizer.med_voc.word2idx))
        multi_label = np.full(len(self.tokenizer.med_voc.word2idx), -1)
        for i, index in enumerate(label_index):
            label[index] = 1
            multi_label[i] = index

        # get the seq len
        # pad the sequence to longest med / diag / proc sequences
        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l
        # convert raw tokens to unified ids
        med_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in med_seq]
        diag_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in diag_seq]
        proc_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in proc_seq]

        # pad the sequence to max possible records
        pad_seq = ["[PAD]" for _ in range(self.seq_len)]
        pad_seq = self.tokenizer.convert_tokens_to_ids(pad_seq)
        def fill_to_max_seq(l, seq):
            pad_num = 0
            while len(l) < seq:
                l.append(pad_seq)
                pad_num += 1
            if len(l) > seq:
                l = l[:seq]
            return l, pad_num
        med_seq = med_seq[:-1]  # remove the current medication set, which is label
        med_seq, _ = fill_to_max_seq(med_seq, self.max_seq)
        diag_seq, pad_num = fill_to_max_seq(diag_seq, self.max_seq)
        proc_seq, _ = fill_to_max_seq(proc_seq, self.max_seq)

        # get mask
        mask = np.ones(self.max_seq)
        if pad_num != 0:
            mask[-pad_num:] = 0

        return np.array(diag_seq, dtype=int), np.array(proc_seq, dtype=int), \
               np.array(med_seq, dtype=int), mask.astype(int), label.astype(float), \
               multi_label.astype(int), np.array(profile, dtype=int)
    

    def _filter_data(self):

        new_records = []

        for record in self.records:
            if len(record["records"]["medication"]) > 1:
                new_records.append(record)

        self.records = new_records
