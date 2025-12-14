"""Causal language model for autoregressive text generation."""

import torch
import torch.nn as nn
from typing import Optional, Union, Unpack

from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.generation import GenerationMixin

from .base import PreTrainedModelBase, BaseModel


class CausalLM(PreTrainedModelBase, GenerationMixin):
    """
    Causal language model with a language modeling head.
    
    Wraps BaseModel and adds an lm_head for next-token prediction.
    Supports text generation via GenerationMixin.
    """
    
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = BaseModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
            position_ids: Position indices for the sequence
            past_key_values: Cached key/value states for generation
            inputs_embeds: Optional pre-computed input embeddings
            labels: Labels for computing the language modeling loss
            use_cache: Whether to return updated cache
            cache_position: Position in the cache
            logits_to_keep: Number of logits to compute (from the end). 
                           Use 0 for all logits, or a positive int/tensor.
            
        Returns:
            CausalLMOutputWithPast containing loss (if labels provided), logits,
            past_key_values, hidden_states, and attentions.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        
        # Only compute necessary logits, avoid upcasting if not computing loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

