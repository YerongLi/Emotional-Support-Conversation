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

        self.dropout1 = nn.Dropout(0.5)
        self.hidden1 = nn.Linear(768, 512)

        self.dropout2 = nn.Dropout(0.3)
        self.hidden2 = nn.Linear(512, 256)

        self.dropout3 = nn.Dropout(0.4)
        self.hidden3 = nn.Linear(256, 128)

        self.dropout4 = nn.Dropout(0.2)
        self.hidden4 = nn.Linear(128, 32)

        self.dropout_cls = nn.Dropout(0.2)
        self.classifier = nn.Linear(32, num_labels)

        self.num_labels = num_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        dev = None,
        use_cache=None,
        return_dict=None,
        **kwargs
    ):
        assert self.toker is not None
        
        encoded_info = kwargs
        # assert (self.training or validation) == (labels is not None) == (decoder_input_ids is not None)
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states0 = transformer_outputs[0]
        hidden_states1 = self.dropout1(hidden_states0)
        hidden_states1 = self.hidden1(hidden_states1)

        hidden_states2 = self.dropout2(hidden_states1)
        hidden_states2 = self.hidden1(hidden_states2)

        hidden_states3 = self.dropout3(hidden_states2)
        hidden_states3 = self.hidden1(hidden_states3)

        hidden_states4 = self.dropout4(hidden_states3)
        hidden_states4 = self.hidden1(hidden_states4)

        sequence_outputs = self.dropout_cls(hidden_states4)

        logits = self.classifier(sequence_outputs[:, 0, :].view(-1, 768))

        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits.view(-1, self.num_labels), dev.view(-1))

            return TokenClassifierOutput(loss=loss, logits=logits)

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