# here put the import lib
import numpy as np


class llama_train(object):
    
    def __init__(self, data_args, model_args, tokenizer) -> None:
    
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = "target"
        self.history_column = None
        self.tokenizer = tokenizer


    def __call__(self, examples):
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }


        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                query, answer = examples[self.prompt_column][i], examples[self.response_column][i]

                if self.history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)

                if len(a_ids) > self.data_args.max_source_length - 1:
                    a_ids = a_ids[: self.data_args.max_source_length - 1]

                if len(b_ids) > self.data_args.max_target_length - 2:
                    b_ids = b_ids[: self.data_args.max_target_length - 2]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
                labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]
                
                pad_len = max_seq_length - len(input_ids)

                if self.data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs



class llama_eval(object):
    
    def __init__(self, data_args, model_args, tokenizer) -> None:
        
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = "target"
        self.history_column = None
        self.tokenizer = tokenizer


    def __call__(self, examples):
    
        max_target_length = self.data_args.max_target_length
        inputs, targets = [], []

        for i in range(len(examples[self.prompt_column])):
            if not examples[self.response_column][i]:
                targets.append("filled in !")
            else:
                targets.append(examples[self.response_column][i])

            if examples[self.prompt_column][i]:
                query = examples[self.prompt_column][i]
                if self.history_column is None or len(examples[self.history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                inputs.append(prompt)

        inputs = [inp for inp in inputs]
        model_inputs = self.tokenizer(inputs,
                                    max_length=self.data_args.max_source_length,
                                    truncation=True,
                                    padding=True)
        labels = self.tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        if self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
        


class llama_train_cls(object):
    
    def __init__(self, data_args, model_args, tokenizer, ehr_tokenizer) -> None:
    
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = "drug_code"
        self.history_column = None
        self.tokenizer = tokenizer
        self.ehr_tokenizer = ehr_tokenizer


    def __call__(self, examples):
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }

        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                query, answer = examples[self.prompt_column][i], examples[self.response_column][i]

                if self.history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)

                if len(a_ids) > self.data_args.max_source_length - 1:
                    a_ids = a_ids[: self.data_args.max_source_length - 1]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids)

                context_length = len(a_ids)
                input_ids = a_ids + [self.tokenizer.eos_token_id]

                label_index = self.ehr_tokenizer.convert_med_tokens_to_ids(answer)
                med_voc_size = len(self.ehr_tokenizer.med_voc.word2idx)
                labels = np.zeros((med_voc_size))
                labels[label_index] = 1

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs


class llama_eval_cls(object):
    
    def __init__(self, data_args, model_args, tokenizer, ehr_tokenizer) -> None:
        
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = "drug_code"
        self.history_column = None
        self.tokenizer = tokenizer
        self.ehr_tokenizer = ehr_tokenizer


    def __call__(self, examples):
    
        max_target_length = self.data_args.max_target_length
        inputs, targets = [], []

        for i in range(len(examples[self.prompt_column])):

            label_index = self.ehr_tokenizer.convert_med_tokens_to_ids(examples[self.response_column][i])
            med_voc_size = len(self.ehr_tokenizer.med_voc.word2idx)
            labels = np.zeros((med_voc_size))
            labels[label_index] = 1
            targets.append(labels)

            if examples[self.prompt_column][i]:
                query = examples[self.prompt_column][i]
                if self.history_column is None or len(examples[self.history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                inputs.append(prompt)

        inputs = [inp for inp in inputs]
        model_inputs = self.tokenizer(inputs,
                                    max_length=self.data_args.max_source_length,
                                    truncation=True,
                                    padding=True)
        model_inputs["labels"] = targets
        #model_inputs["labels"] = None

        return model_inputs
    

class llama_dpo_cls(object):
    
    def __init__(self, data_args, model_args, tokenizer, ehr_tokenizer) -> None:
    
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.positive_column = "positive"
        self.negative_column = "negative"
        self.history_column = None
        self.tokenizer = tokenizer
        self.ehr_tokenizer = ehr_tokenizer


    def __call__(self, examples):
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "prompt_ids": [],
            "chosen": [],
            "rejected_ids": [],
        }

        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.positive_column][i]:
                query = examples[self.prompt_column][i]
                positive, negative = examples[self.positive_column][i], examples[self.negative_column][i]

                if self.history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)

                if len(a_ids) > self.data_args.max_source_length - 1:
                    a_ids = a_ids[: self.data_args.max_source_length - 1]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids)

                context_length = len(a_ids)
                input_ids = a_ids + [self.tokenizer.eos_token_id]

                med_voc_size = len(self.ehr_tokenizer.med_voc.word2idx)
                positive_labels, negative_labels = np.zeros((med_voc_size)), np.zeros((med_voc_size))
                positive_labels[positive] = 1
                negative_labels[negative] = 1

                model_inputs["prompt_ids"].append(input_ids)
                model_inputs["chosen_ids"].append(positive_labels)
                model_inputs["rejected_ids"].append(negative_labels)

        return model_inputs