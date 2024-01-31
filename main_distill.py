# here put the import lib
import os
import argparse

import numpy as np
import pandas as pd
import torch

from generators.distill_generator import DistillGenerator
from trainers.distill_trainer import DistillTrainer
from utils.utils import set_seed, log_res
from utils.logger import Logger


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name", 
                    default='leader', 
                    type=str, 
                    choices=["leader"],
                    help="model name")
parser.add_argument("--dataset", 
                    default="eicu", 
                    choices=['eicu', 'mimic3', 'mimic4'], 
                    help="Choose the dataset")
parser.add_argument("--demo", 
                    default=False, 
                    action='store_true', 
                    help='whether run demo')
parser.add_argument("--train_file", 
                    default='record_final.pkl', 
                    type=str, 
                    required=False,
                    help="training data file.")
parser.add_argument("--filter",
                    default=False,
                    action="store_true",
                    help="Whether filter out the single-visit records for those multi-visit only models")
parser.add_argument("--output_dir",
                    default='./saved/',
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--out_exp",
                    default='./log/result.json',
                    type=str,
                    help="The output json for multiple experiments of multiple centers")
parser.add_argument("--check_path",
                    default='',
                    type=str,
                    help="the save path of checkpoints for different running")

# Other parameters
parser.add_argument("--freeze", 
                    default=False,
                    action="store_true",
                    help="Whether freeze some layers of the model for finetuning")
parser.add_argument("--graph",
                    default=False,
                    action='store_true',
                    help="if use ontology embedding")
parser.add_argument("--therhold",
                    default=0.3,
                    type=float,
                    help="therhold.")
parser.add_argument("--hidden_size",
                    default=64,
                    type=int,
                    help="hidden size")
parser.add_argument("--max_seq_length",
                    default=100,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
parser.add_argument("--max_record_num",
                    default=10,
                    type=int,
                    help="The maximum record number.")
parser.add_argument("--do_train",
                    default=False,
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    default=True,
                    action='store_true',
                    help="Whether to run on the dev set.")
parser.add_argument("--do_test",
                    default=True,
                    action='store_true',
                    help="Whether to run on the test set.")
parser.add_argument("--train_batch_size",
                    default=128,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--learning_rate",
                    default=5e-4,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2",
                    default=0,
                    type=float,
                    help='The L2 regularization')
parser.add_argument("--num_train_epochs",
                    default=30,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_dc_step",
                    default=1000,
                    type=int,
                    help='every n step, decrease the lr')
parser.add_argument("--lr_dc",
                    default=0,
                    type=float,
                    help='how many learning rate to decrease')
parser.add_argument("--patience",
                    type=int,
                    default=10,
                    help='How many steps to tolerate the performance decrease while training')
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")
parser.add_argument('--gpu_id',
                    default=0,
                    type=int,
                    help='The device id.')
parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='The number of workers in dataloader')
parser.add_argument("--log", 
                    default=False,
                    action="store_true",
                    help="whether create a new log file")
parser.add_argument("--out_file",
                    default="none",
                    type=str,
                    help="the output file to save results, if 'none', save to dataset.json")
parser.add_argument("--mark_name",
                    default="default",
                    type=str,
                    help="the marked name will be shown in result json")
parser.add_argument("--full",
                    default=False,
                    action="store_true",
                    help="use the data in original data distribution")
parser.add_argument("--num_trm_layers",
                    default=1,
                    type=int,
                    help="the number of trm layers")

# LLM Parameters
parser.add_argument("--llm_path",
                    default="./resources/llama-7b/",
                    type=str,
                    help="The path of large language model.")
parser.add_argument("--peft_path",
                    default="./saved/lora-1117_cls/checkpoint-2000/",
                    type=str,
                    help="The lora path for finetuned LLM.")
parser.add_argument("--max_source_length",
                    default=1024,
                    type=int,
                    help="The max source input length to LLM")
parser.add_argument("--distill",
                    default=False,
                    action="store_true",
                    help="whether apply the distillation")
parser.add_argument("--alpha",
                    default=0.1,
                    type=float,
                    help="The weight to adjust distillation loss.")
parser.add_argument("--medrec_path", 
                    default="./saved/mimic3/pnet",
                    type=str,
                    help="The save path for medication recommendation model")
parser.add_argument("--finetune",
                    default=False,
                    action="store_true",
                    help="If finetuning, load well-train medrec and freeze params")
parser.add_argument("--prompt_num",
                    default=1,
                    type=int,
                    help="The number of prompt embeddings.")
parser.add_argument("--d_loss",
                    type=str,
                    choices=["mse"],
                    default="mse",
                    help="The type of distillation loss")
parser.add_argument("--profile",
                    default=False,
                    action="store_true",
                    help="Whether use the profile encoder, otherwise the padding encoder")
parser.add_argument("--temperature",
                    default=10,
                    type=float,
                    help="The temperature for distillation")
parser.add_argument("--ddi",
                    default=False,
                    action="store_true",
                    help="whether adopt the ddi loss")
parser.add_argument("--target_ddi",
                    default=0.06,
                    type=float,
                    help="target ddi rate")
parser.add_argument("--ddi_temp",
                    default=2.0,
                    type=float,
                    help="the temperature for ddi update")
parser.add_argument("--ml_weight",
                    default=0.05,
                    type=float,
                    help="the weight of multi-label loss")
parser.add_argument("--align",
                    default=False,
                    action="store_true",
                    help="align the output of profile encoder with medication recommendation")
parser.add_argument("--align_weight",
                    default=0.1,
                    type=float,
                    help="the weight for alignment loss")
parser.add_argument("--offline",
                    default=False,
                    action="store_true",
                    help="whether use offline hidden states for distillation")


args = parser.parse_args()
args.data_dir = './data/' + str(args.dataset) + '/handled/'
args.output_dir = args.output_dir + str(args.dataset) + '/'
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.output_dir = os.path.join(args.output_dir, args.check_path)

set_seed(args.seed) # fix the random seed


def main():

    log_manager = Logger(args)  # initialize the log manager
    logger = log_manager.get_logger()    # get the logger
    args.mark_name = args.mark_name + "-" + log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    #generator = FinetuneGenerator(args, logger, device)
    generator = DistillGenerator(args, logger, device)
    #trainer = FinetuneTrainer(args, logger, device, generator)
    trainer = DistillTrainer(args, logger, device, generator)

    res, best_epoch, train_time = trainer.train()
    # fp_num, ap_num = trainer.get_model_param_num()  # freeze parameter num, active parameter num

    if args.log:
        log_res(args, res)

    log_manager.end_log()   # delete the logger threads



if __name__ == "__main__":

    main()



