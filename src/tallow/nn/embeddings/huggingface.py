import sys
from typing import Dict, Literal

import torch

try:
    import transformers
except ImportError:

    class HuggingfaceEmbedding:
        def __init__(self, *args, **kwargs) -> None:
            raise


from torch import nn

from . import classes


class HuggingfaceEmbedding(classes.SequenceEmbedding):
    def __init__(
        self,
        pretrained_path: str,
        padding_idx: int = None,
        *,
        freeze: bool = False,
        init_kwargs: Dict = None,
    ) -> None:
        super().__init__()
        init_kwargs = init_kwargs or {}
        self._hf_transformer: transformers.PreTrainedModel = (
            transformers.AutoModel.from_pretrained(
                pretrained_path, **init_kwargs
            )
        )
        if padding_idx is None:
            hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained_path
            )
            padding_idx = hf_tokenizer.pad_token_id
        self._padding_idx: int = padding_idx
        self._has_cls_sep = False
        if freeze:
            if isinstance(self._hf_transformer, nn.Module):
                self._hf_transformer.requires_grad_(False)
            else:
                raise ValueError()

        print(
            "Load %s (%d, %d) from %s"
            % (
                self._hf_transformer.__class__.__name__,
                self.num_embeds,
                self.embed_size,
                pretrained_path,
            ),
            file=sys.stderr,
        )

    def resize_token_embeddings(self, vocab_size: int):
        self._hf_transformer.resize_token_embeddings(vocab_size)
        return self

    def forward(self, input_ids: torch.Tensor):
        """
        Arguments:
            input_ids: (batch_size, seq_len)
        Return:
            cls_output: (batch_size, hidden_size)
            hidden_states: (batch_size, new_len, hidden_size)
        """
        *outer_sizes, seq_len = input_ids.size()
        flatten_input_ids = input_ids.reshape(-1, seq_len)

        input_mask = input_ids.ne(self._padding_idx)
        attention_mask = input_mask.view_as(flatten_input_ids).float()

        bert_outputs = self._hf_transformer(
            input_ids=flatten_input_ids, attention_mask=attention_mask
        )
        flatten_last_state: torch.Tensor = bert_outputs["last_hidden_state"]

        last_state = flatten_last_state.view(
            *outer_sizes, seq_len, self.embed_size
        )
        cls_output = flatten_last_state[:, 0, :].view(
            *outer_sizes, self.embed_size
        )

        return self.Output(
            hidden_states=last_state,
            pooled_output=cls_output,
            mask=input_mask,
        )

    @property
    def num_embeds(self):
        return self._hf_transformer.config.vocab_size

    @property
    def embed_size(self):
        return self._hf_transformer.config.hidden_size

    @property
    def pad_id(self) -> int:
        return self._padding_idx


class SubwordPooling(nn.Module):
    def __init__(self, pooling: Literal["first", "mean", "last"]):
        super().__init__()
        self._pooling = pooling

    def forward(self, token_embeds: torch.Tensor, token_to_words: torch.Tensor):
        """
        Arguments:
            token_embeds: (batch_size, seq_len, embed_size)
            token_to_words: (batch_size, seq_len)
        Return:
            word_embeds: (batch_size, new_len, embed_size)
                where new_len == max(token_to_words) + 1
        """
        new_length = torch.max(token_to_words) + 1
        m = torch.zeros(new_length, token_to_words.size(1))
        raise NotImplementedError()

        return m @ token_embeds
