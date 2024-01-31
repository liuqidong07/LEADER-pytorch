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


class FinetuneTrainer(Trainer):

    def __init__(self, args, logger, device, generator):
        
        super().__init__(args, logger, device, generator)

        if self.args.freeze:
            self._freeze()


    def _create_model(self):
        '''Load pretrain model or not'''
        if self.args.model_name == "leader":
            config = BertConfig(vocab_size_or_config_json_file=len(self.tokenizer.vocab.word2idx))
            config.hidden_size = self.args.hidden_size
            self.model = LEADER(config, self.tokenizer, self.device)
        
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
            # diag_seq / proc_seq / med_seq: (bs, max_seq_len, max_set_len), mask: (bs, max_seq_len), label: (bs, med_voc_size)
            diag_seq, proc_seq, med_seq, seq_mask, labels = batch 
            loss, loss_ddi, output = self.model(diag_seq, proc_seq, med_seq, seq_mask, labels)
            loss.backward()

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
            diag_seq, proc_seq, med_seq, seq_mask, labels = batch 
            with torch.no_grad():
                output = self.model(diag_seq, proc_seq, med_seq, seq_mask, labels)
                rx_y_preds.append(t2n(torch.sigmoid(output)))
                rx_y_trues.append(t2n(labels))
                seq_len.append(t2n(torch.sum(seq_mask, dim=1)))


        self.logger.info('')
        acc_container = metric_report(self.logger, 
                                      np.concatenate(rx_y_preds, axis=0), 
                                      np.concatenate(rx_y_trues, axis=0),
                                      self.args.therhold)
        acc_container_group = metric_report_group(self.logger, 
                                                  np.concatenate(rx_y_preds, axis=0), 
                                                  np.concatenate(rx_y_trues, axis=0),
                                                  np.concatenate(seq_len, axis=0),
                                                  self.args.therhold)
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




