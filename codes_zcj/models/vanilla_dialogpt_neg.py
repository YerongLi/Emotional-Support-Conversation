# coding=utf-8
# copied from gpt2

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import BaseModel
from transformers.generation_utils import top_k_top_p_filtering
from transformers.models.gpt2 import (GPT2Config, GPT2LMHeadModel,)
from transformers.modeling_outputs import TokenClassifierOutput

from transformers.modeling_outputs import (CausalLMOutputWithCrossAttentions,)
from .PARAMS import SAMPLE, TEMPERATURE


class Model(BaseModel, GPT2LMHeadModel):
    def __init__(self, config: GPT2Config, num_labels=2):
        super().__init__(config)
        self.dropouts = nn.Dropout(0.1)
        self.linear1 = nn.Linear(512, 240)
        self.linear1 = nn.Linear(240, 12)

        self.dropouts1 = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, num_labels)

        self.num_labels = num_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        **kwargs
    ):
        assert self.toker is not None
        
        encoded_info = kwargs
        # assert (self.training or validation) == (labels is not None) == (decoder_input_ids is not None)
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation: # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        sequence_outputs = self.dropouts(hidden_states)

        logits = self.classifier(sequence_outputs[:, 0, :].view(-1, 768))

        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))

            return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=sequence_outputs.hidden_states,
                                         attentions=sequence_outputs.attentions)

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        **kwargs
    ):
        assert not self.training
        assert self.toker is not None
        
        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        
        input_ids = torch.cat([input_ids, decoder_input_ids], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(decoder_input_ids.size())], dim=-1)
        
        assert 'min_length' in kwargs and 'max_length' in kwargs
        kwargs['min_length'] = kwargs['min_length'] + input_ids.size(1)
        kwargs['max_length'] = kwargs['max_length'] + input_ids.size(1)
        kwargs['use_cache'] = True
        
        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            kwargs['bad_words_ids'] = bad_words_ids
        
        generations = super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return encoded_info, generations[:, input_ids.size(1):]