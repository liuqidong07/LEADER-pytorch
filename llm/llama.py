# here put the import lib
from typing import Optional, List, Union, Tuple
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel, LlamaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


class LlamaForMedRec(LlamaPreTrainedModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        
        super().__init__(config, *inputs, **kwargs)
        self.model = LlamaModel(config)
        self.med_voc = kwargs.pop("med_voc")
        self.cls_head = nn.Linear(config.hidden_size, self.med_voc, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.model.embed_tokens


    def set_input_embeddings(self, value):
        self.model.embed_tokens = value


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # get the output from LLM (bs, seq_len, hidden_size)
        logits = self.cls_head(hidden_states)   # use a head for classification

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        # get out the last embedding of the embedding sequence
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            #loss_fct = CrossEntropyLoss()
            #loss = loss_fct(pooled_logits, labels)
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(pooled_logits, labels.float())
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        transformer_outputs.hidden_states = hidden_states[torch.arange(batch_size, device=logits.device), sequence_lengths]   # output the hidden_states for feature-based KD

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )










