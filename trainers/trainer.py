# here put the import lib
import os
import numpy as np
import torch
from tqdm import tqdm, trange
from utils.earlystop import EarlyStopping
from utils.utils import get_n_params, t2n, metric_report
import time


class Trainer(object):

    def __init__(self, args, logger, device, generator):

        self.args = args
        self.logger = logger
        self.device = device
        self.tokenizer = generator.get_tokenizer()

        self.logger.info('Loading Model: ' + args.model_name)
        self._create_model()
        logger.info('# of model parameters: ' + str(get_n_params(self.model)))

        self._set_optimizer()
        self._set_scheduler()
        self._set_stopper()
        
        self.train_loader, self.eval_loader, self.test_loader = generator.make_dataloaders()
        self.generator = generator

        self.watch_metric = 'prauc'  # use which metric to select model

    
    def _create_model(self):
        '''create your model'''
        return NotImplementedError

    
    def _set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.l2)

    
    def _set_scheduler(self):

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.args.lr_dc_step,
                                                         gamma=self.args.lr_dc)


    def _set_stopper(self):

        self.stopper = EarlyStopping(patience=self.args.patience, 
                                     verbose=False,
                                     path=self.args.output_dir,
                                     trace_func=self.logger)


    def _train_one_epoch(self, epoch):

        return NotImplementedError
    

    def _prepare_train_inputs(self, data):

        assert len(self.generator.train_dataset.var_name) == len(data)
        inputs = {}
        for i, var_name in enumerate(self.generator.train_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs
    

    def _prepare_eval_inputs(self, data):

        inputs = {}
        assert len(self.generator.eval_dataset.var_name) == len(data)
        for i, var_name in enumerate(self.generator.eval_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs


    def eval(self, epoch=0, test=False):

        return NotImplementedError


    def train(self):

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Num examples = %d", self.generator.get_statistics()[0])
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        res_list = []
        train_time = []

        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch"):

            t = self._train_one_epoch(epoch)
            
            train_time.append(t)

            metric_dict = self.eval(epoch=epoch)
            res_list.append(metric_dict)
            self.scheduler.step()
            self.stopper(metric_dict[self.watch_metric], epoch, model_to_save)

            if self.stopper.early_stop:

                break
        
        best_epoch = self.stopper.best_epoch
        best_res = res_list[best_epoch]
        # with open(os.path.join(self.args.output_dir, 'bert_config.json'), 'w', encoding='utf-8') as fout:
        #     fout.write(self.model.config.to_json_string())
        self.logger.info('')
        self.logger.info('The best epoch is %d' % best_epoch)
        self.logger.info('The best results are Jaccard: %.5f, F1-score: %.5f, PRAUC: %.5f' %
                    (best_res['jaccard'], best_res['f1'], best_res['prauc']))
        
        res = self.eval(test=True)
        train_time = np.array(train_time).mean()

        return res, best_epoch, train_time



    def get_model(self):

        return self.model

    
    def get_model_param_num(self):

        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        freeze_num = total_num - trainable_num

        return freeze_num, trainable_num


