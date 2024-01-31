# here put the import lib
import os
import json
import copy
import torch
import numpy as np
from transformers import AutoTokenizer
from generators.generator import Generator
from generators.data import EHRTokenizer, EHRDataset
from utils.utils import read_jsonlines



class DistillEHRDataset(EHRDataset):

    def __init__(self, data_pd, tokenizer: EHRTokenizer, profile_tokenizer, llm_tokenizer, args):
        
        super().__init__(data_pd, tokenizer, args.max_seq_length)
        self.max_seq = args.max_record_num
        self.profile_tokenizer = profile_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.max_source_length = args.max_source_length

        self.var_name = ["diag_seq", "proc_seq", "med_seq", "seq_mask", "labels", "multi_label", "profile", "input_ids"]


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

        # get LLM input
        prompt = adm["input"]
        prompt = self.llm_tokenizer.encode(text=prompt, add_special_tokens=False)
        prompt = prompt[:self.max_source_length-1]
        input_ids = prompt + [self.llm_tokenizer.eos_token_id]
        while len(input_ids) < self.max_source_length:  # pad the input to max_source_length
            input_ids += [self.llm_tokenizer.pad_token_id]

        return np.array(diag_seq, dtype=int), np.array(proc_seq, dtype=int), \
               np.array(med_seq, dtype=int), mask.astype(int), label.astype(float), \
               multi_label.astype(int), np.array(profile, dtype=int), \
               np.array(input_ids, dtype=int)



class OfflineDistillEHRDataset(EHRDataset):

    def __init__(self, data_pd, tokenizer: EHRTokenizer, profile_tokenizer, llm_tokenizer, args):
        
        super().__init__(data_pd, tokenizer, args.max_seq_length)
        self.max_seq = args.max_record_num
        self.profile_tokenizer = profile_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.max_source_length = args.max_source_length

        self.var_name = ["diag_seq", "proc_seq", "med_seq", "seq_mask", \
                         "labels", "multi_label", "profile", "input_ids", \
                         "hidden_states", "logits"]


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

        # get LLM input
        prompt = adm["input"]
        prompt = self.llm_tokenizer.encode(text=prompt, add_special_tokens=False)
        prompt = prompt[:self.max_source_length-1]
        input_ids = prompt + [self.llm_tokenizer.eos_token_id]
        while len(input_ids) < self.max_source_length:  # pad the input to max_source_length
            input_ids += [self.llm_tokenizer.pad_token_id]

        hidden_states = adm["hidden_states"]
        logits = adm["target"]

        return np.array(diag_seq, dtype=int), np.array(proc_seq, dtype=int), \
               np.array(med_seq, dtype=int), mask.astype(int), label.astype(float), \
               multi_label.astype(int), np.array(profile, dtype=int), \
               np.array(input_ids, dtype=int), np.array(hidden_states, dtype=float), \
               np.array(logits, dtype=float)



class DistillGenerator(Generator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)


    def _load_dataset(self):

        data_dir = self.args.data_dir

        voc_dir = os.path.join(data_dir, r'voc_final.pkl')
        if self.args.offline:
            train_file = "offline_" + 'train_{}.json'.format(self.args.train_file)
        else:
            train_file = 'train_{}.json'.format(self.args.train_file)
        train_data = read_jsonlines(os.path.join(data_dir, train_file))
        val_data = read_jsonlines(os.path.join(data_dir, 'val_{}.json'.format(self.args.train_file)))
        test_data = read_jsonlines(os.path.join(data_dir, 'test_{}.json'.format(self.args.train_file)))

        # load tokenizer
        self.tokenizer = EHRTokenizer(voc_dir)
        self.profile_tokenizer = json.load(open(self.args.data_dir+"profile_dict.json", "r"))
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.args.llm_path,
            trust_remote_code=True,
        )
        self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token
        self.llm_tokenizer.padding_side = "right"  # define the padding direction
        
        if self.args.offline:   # use the offline distillation
            TargetDataset = OfflineDistillEHRDataset
        else:
            TargetDataset = DistillEHRDataset
        
        self.train_dataset = TargetDataset(train_data, self.tokenizer, self.profile_tokenizer, self.llm_tokenizer, self.args)
        self.eval_dataset = DistillEHRDataset(val_data, self.tokenizer, self.profile_tokenizer, self.llm_tokenizer, self.args)
        self.test_dataset = DistillEHRDataset(test_data, self.tokenizer, self.profile_tokenizer, self.llm_tokenizer, self.args)


    def get_llm_tokenizer(self):

        return self.llm_tokenizer



