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
from llm.llama import LlamaForMedRec
from llm.lora_cls import PeftModelForCLS


class DistillTrainer(Trainer):

    def __init__(self, args, logger, device, generator):
        
        self.profile_tokenizer = generator.get_profile_tokenizer()
        self.llm_tokenizer = generator.get_llm_tokenizer()
        data_dir = args.data_dir
        self.ehr_adj = pickle.load(open(os.path.join(data_dir, 'ehr_adj_final.pkl'), 'rb'))
        self.ddi_adj = pickle.load(open(os.path.join(data_dir, 'ddi_A_final.pkl'), 'rb'))
        super().__init__(args, logger, device, generator)

        if self.args.finetune:
            self._load_model()
            self._freeze()


    def _create_model(self):
        '''Load pretrain model or not'''
        config = BertConfig(vocab_size_or_config_json_file=len(self.tokenizer.vocab.word2idx))
        config.hidden_size = self.args.hidden_size
        if self.args.model_name == "pnet":
            self.model = LEADER(config, self.args, self.tokenizer, self.device, self.profile_tokenizer)

        if not self.args.offline:
            self.teacher = LlamaForMedRec.from_pretrained(
                self.args.llm_path,
                med_voc=len(self.tokenizer.med_voc.word2idx),
            ).half().to(self.device)
            self.teacher = PeftModelForCLS.from_pretrained(self.teacher, self.args.peft_path, is_trainable=False)
        else:
            self.teacher = None

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

            if not self.args.offline:
                llm_inputs = {"input_ids": inputs["input_ids"], "labels": None}
                inputs["llm_output"] = self.teacher(**llm_inputs)
            else:
                inputs["llm_output"] = {"hidden_states": inputs["hidden_states"],
                                        "logits": inputs["logits"]}

            loss = self.model.get_loss(**inputs)
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
    
            if ('profile' in name) or ("medrec" in name):   # prompt and head are activated
                param.requires_grad = True
            else:
                param.requires_grad = False
    

    def _load_model(self):

        model_state_dict = torch.load(os.path.join(self.args.medrec_path, 'pytorch_model.bin'))
        
        remove_list = []    # record the removed modules
        for k, v in model_state_dict.items():
            if ('profile' in k) or ("medrec" in k):
                remove_list.append(k)

        for k in remove_list:
            model_state_dict.pop(k)
        
        self.model.load_state_dict(model_state_dict, strict=False)
        print("Load pretrained MedRec model")


