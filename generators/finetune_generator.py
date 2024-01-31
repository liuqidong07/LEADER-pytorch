# here put the import lib
import os
import pickle
import json
import pandas as pd
from generators.generator import Generator
from generators.data import EHRTokenizer, FinetuneEHRDataset, MedRecEHRDataset, Voc
from utils.utils import read_jsonlines


class FinetuneGenerator(Generator):

    def __init__(self, args, logger, device):

        self.hos_index = None
        super().__init__(args, logger, device)
        


    def _load_dataset(self):

        data_dir = self.args.data_dir
        max_seq_len = self.args.max_seq_length

        # whether run demo
        if self.args.demo:
            voc_dir = os.path.join(data_dir, 'vocab.demo.pkl')
            record_dir = os.path.join(data_dir, 'data-single-visit.demo.pkl')
            data = pd.read_pickle(record_dir)
            data_list = self._split_dataset(data)
        else:# full train, full test
            voc_dir = os.path.join(data_dir, r'voc_final.pkl')
            record_dir = os.path.join(data_dir, 'records_final_raw.pkl')
            # load data
            data = pd.read_pickle(record_dir)
            # load data_list = [trian, eval, test data]
            data_list = self._split_dataset(data)

        # load tokenizer
        self.tokenizer = EHRTokenizer(voc_dir)
        
        self.train_dataset = FinetuneEHRDataset(data_list[0], self.tokenizer, max_seq_len)
        self.eval_dataset = FinetuneEHRDataset(data_list[1], self.tokenizer, max_seq_len)
        self.test_dataset = FinetuneEHRDataset(data_list[2], self.tokenizer, max_seq_len)



class MedRecGenerator(Generator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)


    def _load_dataset(self):

        data_dir = self.args.data_dir

        voc_dir = os.path.join(data_dir, r'voc_final.pkl')
        train_data = read_jsonlines(os.path.join(data_dir, 'train_{}.json'.format(self.args.train_file)))
        val_data = read_jsonlines(os.path.join(data_dir, 'val_{}.json'.format(self.args.train_file)))
        test_data = read_jsonlines(os.path.join(data_dir, 'test_{}.json'.format(self.args.train_file)))

        # load tokenizer
        self.tokenizer = EHRTokenizer(voc_dir)
        self.profile_tokenizer = json.load(open(self.args.data_dir+"profile_dict.json", "r"))
        
        self.train_dataset = MedRecEHRDataset(train_data, self.tokenizer, self.profile_tokenizer, self.args)
        self.eval_dataset = MedRecEHRDataset(val_data, self.tokenizer, self.profile_tokenizer, self.args)
        self.test_dataset = MedRecEHRDataset(test_data, self.tokenizer, self.profile_tokenizer, self.args)





