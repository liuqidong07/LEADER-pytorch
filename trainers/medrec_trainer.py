# here put the import lib
import os
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from trainers.trainer import Trainer
from models.LEADER import LEADER
from utils.config import BertConfig
from utils.utils import t2n, metric_report, metric_report_group
import time


class MedRecTrainer(Trainer):

    def __init__(self, args, logger, device, generator):
        
        self.profile_tokenizer = generator.get_profile_tokenizer()
        data_dir = args.data_dir
        self.ehr_adj = pickle.load(open(os.path.join(data_dir, 'ehr_adj_final.pkl'), 'rb'))
        self.ddi_adj = pickle.load(open(os.path.join(data_dir, 'ddi_A_final.pkl'), 'rb'))
        self.ddi_mask_H = pickle.load(open(os.path.join(data_dir, "ddi_mask_H.pkl"), "rb"))
        self.molecule = pickle.load(open(os.path.join(data_dir, "atc3toSMILES.pkl"), "rb"))

        super().__init__(args, logger, device, generator)

        if self.args.freeze:
            self._freeze()


    def _create_model(self):
        '''Load pretrain model or not'''
        config = BertConfig(vocab_size_or_config_json_file=len(self.tokenizer.vocab.word2idx))
        config.hidden_size = self.args.hidden_size

        if self.args.model_name == "leader":
            self.model = LEADER(config, self.args, self.tokenizer, self.device, self.profile_tokenizer)
        else:
            raise ValueError
        
        self.model.to(self.device)


    def _train_one_epoch(self, epoch):
        
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')
        for batch in prog_iter:
            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            inputs = self._prepare_train_inputs(batch) 
            loss = self.model.get_loss(**inputs)
            loss.backward(retain_graph=True)

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        return np.array(train_time).mean()


    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            self.logger.info("  Num examples = %d", self.generator.get_statistics()[2])
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.eval_loader
        
        self.model.eval()

        rx_y_preds = []
        rx_y_trues = []
        seq_len = []
        for batch in tqdm(test_loader, desc=desc):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = self._prepare_eval_inputs(batch) 
                output = self.model(**inputs)
                rx_y_preds.append(t2n(torch.sigmoid(output)))
                rx_y_trues.append(t2n(inputs["labels"]))
                seq_len.append(t2n(torch.sum(inputs["seq_mask"], dim=1)))


        self.logger.info('')
        acc_container = metric_report(self.logger, 
                                      np.concatenate(rx_y_preds, axis=0), 
                                      np.concatenate(rx_y_trues, axis=0),
                                      self.args.therhold,
                                      self.ddi_adj)
        acc_container_group = metric_report_group(self.logger, 
                                                  np.concatenate(rx_y_preds, axis=0), 
                                                  np.concatenate(rx_y_trues, axis=0),
                                                  np.concatenate(seq_len, axis=0),
                                                  self.args.therhold,
                                                  self.ddi_adj)
        acc_container.update(acc_container_group)   # merge two dicts
        return acc_container



    def test(self):

        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running test **********")
        self.logger.info("  Num examples = %d", self.generator.get_statistics()[2])
        desc = 'Testing'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'), map_location=self.device)
        
        # filter out redundant parameters in checkpoint
        new_model_state_dict = {}
        for key, value in model_state_dict.items():
            if key in self.model.state_dict().keys():
                new_model_state_dict[key] = value
        
        self.model.load_state_dict(new_model_state_dict)
        self.model.to(self.device)
        test_loader = self.test_loader
        
        self.model.eval()

        rx_y_preds = []
        rx_y_trues = []
        seq_len = []
        for batch in tqdm(test_loader, desc=desc):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = self._prepare_eval_inputs(batch) 
                output = self.model(**inputs)
                rx_y_preds.append(t2n(torch.sigmoid(output)))
                rx_y_trues.append(t2n(inputs["labels"]))
                seq_len.append(t2n(torch.sum(inputs["seq_mask"], dim=1)))


        self.logger.info('')
        acc_container = metric_report(self.logger, 
                                      np.concatenate(rx_y_preds, axis=0), 
                                      np.concatenate(rx_y_trues, axis=0),
                                      self.args.therhold,
                                      self.ddi_adj)
        acc_container_group = metric_report_group(self.logger, 
                                                  np.concatenate(rx_y_preds, axis=0), 
                                                  np.concatenate(rx_y_trues, axis=0),
                                                  np.concatenate(seq_len, axis=0),
                                                  self.args.therhold,
                                                  self.ddi_adj)
        acc_container.update(acc_container_group)   # merge two dicts
        return acc_container
    

    
    def _freeze(self):
        # freeze all of bert parameters in the model except for prompt 
        for name, param in self.model.named_parameters():
    
            if 'bert' in name:
                if 'prompt' in name:
                    continue
                else:
                    param.requires_grad = False

        return 0












