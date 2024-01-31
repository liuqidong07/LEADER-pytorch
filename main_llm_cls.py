# here put the import lib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import json
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from llm.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaForSequenceClassification
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, HfArgumentParser, Seq2SeqTrainingArguments
from transformers import AutoModel, AutoTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from llm.llama import LlamaForMedRec
from llm.trainer_seq2seq import MedRecTrainer
from llm.lora_cls import PeftModelForCLS
from llm.arguments import DataTrainingArguments, ModelArguments
from llm.data_processor.llama import llama_train_cls, llama_eval_cls
from llm.data_processor.collator import LongestSequenceCollator
from generators.data import Voc, EHRTokenizer
from evaluate import evaluate_jsonlines
import time


# save model for PeftModel
class SavePeftModelCallback(TrainerCallback):
    def on_save(    
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            print('+++++++++++++++++save call back++++++++++++++++')
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control
        

def train():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device_map = "auto"

    # load diag, proc, med word2id tokenizer
    voc_dir = "data/mimic3/handled/voc_final.pkl"
    ehr_tokenizer = EHRTokenizer(voc_dir)

    ## Load Model ##
    model = LlamaForMedRec.from_pretrained(
        model_args.model_name_or_path,
        med_voc=len(ehr_tokenizer.med_voc.word2idx),
    ).half().cuda()

    if model_args.peft_path is not None:    # for test model
        # Resume_training
        if training_args.resume_from_checkpoint is not None:
            model = PeftModelForCLS.from_pretrained(model, model_args.peft_path, is_trainable=True)
        else:
            model = PeftModelForCLS.from_pretrained(model, model_args.peft_path, is_trainable=False)
    else:   # for train model
        # Load Lora Config
        peft_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.trainable.split(","),
            lora_dropout=model_args.lora_dropout,
            task_type="SEQ_CLS",
        )

        model = PeftModelForCLS(model, peft_config)  # LoRA wrapped llama

    if training_args.do_train:
        for name, param in model.named_parameters():    # activate the CLS head parameters
            if "cls_head" in name:
                param.requires_grad = True
    model.print_trainable_parameters()

    ## Load Tokenizer ##
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"  # define the padding direction

    ## Load Dataset ##
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print("raw_datasets: ", raw_datasets)

    if training_args.do_train:
        target_dataset = raw_datasets["train"]
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        target_dataset = raw_datasets["eval"]
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        target_dataset = raw_datasets["test"]
        # preprocess_func = llama_eval_cls(data_args, model_args, tokenizer, ehr_tokenizer)
        column_names = raw_datasets["test"].column_names
        # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id, 
        #                                        pad_to_multiple_of=None, padding=False)
    
    preprocess_func = llama_train_cls(data_args, model_args, tokenizer, ehr_tokenizer)
    data_collator = LongestSequenceCollator(tokenizer)

    with training_args.main_process_first(desc="Dataset map pre-processing"):
        target_dataset = target_dataset.map(
            preprocess_func,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on prediction dataset",
        )
    target_dataset.set_format("torch")

    ## Set Trainer ##
    trainer = MedRecTrainer(
        model=model,
        args=training_args,
        train_dataset=target_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None), # substitute the original model saver
    )

    ## Train Model
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()

    ## Evaluation ##
    results = {}

    if training_args.do_predict:
        list_test_samples = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                list_test_samples.append(line)

        start_time = time.time()
        with torch.no_grad():
            predict_results = trainer.predict(
                target_dataset,
                metric_key_prefix="predict",
                # max_tokens=512,
                # max_new_tokens=data_args.max_target_length,
                # do_sample=True,
                # top_p=0.7,     
                # temperature=0.95,
                # repetition_penalty=1.1
            )
        end_time = time.time()

        if trainer.is_world_process_zero():
            predictions = predict_results.predictions
            assert len(predictions) == len(list_test_samples)
            hidden_states = predict_results.label_ids

            output_prediction_file = os.path.join(training_args.output_dir, "test_predictions.json")

            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                for idx, p in enumerate(predictions):
                    samp = list_test_samples[idx]
                    #samp["target"] = ehr_tokenizer.med_voc.idx2word[p]
                    samp["hidden_states"] = hidden_states[idx].astype(float).tolist()
                    samp["target"] = p.astype(float).tolist()
                    res = json.dumps(samp, ensure_ascii=False)
                    writer.write(f"{res}\n")

            results = evaluate_jsonlines(output_prediction_file, ehr_tokenizer)   # output the MedRec metrics

    return results


if __name__ == "__main__":

    train()


