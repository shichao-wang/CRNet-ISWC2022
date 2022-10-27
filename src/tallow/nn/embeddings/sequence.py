from typing import Literal

import torch
from torch import nn

import tallow as tl

from . import classes, lookup

_rnn_impls = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}


class RNNEmbedding(classes.SequenceEmbedding):
    def __init__(
        self,
        embedding: lookup.LookupEmbedding,
        cell_type: Literal["GRU", "LSTM", "RNN"],
        embed_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self._embedding = embedding
        rnn_class = _rnn_impls[cell_type]
        self._hidden_size = embed_size
        self._rnn = rnn_class(
            input_size=embedding.embed_size,
            hidden_size=embed_size // 2 if bidirectional else embed_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self._linear1 = nn.Linear(num_layers * embed_size, embed_size)
        self._linear2 = nn.Linear(num_layers * embed_size, embed_size)

    @property
    def pad_id(self) -> int:
        return self._embedding.pad_id

    @property
    def num_embeds(self) -> int:
        return self._embedding.num_embeds

    @property
    def embed_size(self) -> int:
        return self._hidden_size

    def forward(self, input_ids: torch.Tensor):
        emebdding_outputs = self._embedding(input_ids)
        mask = emebdding_outputs["mask"]
        hidden_states, last_state = tl.nn.rnn_forward(
            self._rnn,
            emebdding_outputs["embedding"],
            mask,
            preserve_length=True,
        )
        return self.Output(
            hidden_states=self._linear1(hidden_states),
            pooled_output=self._linear2(last_state),
            mask=mask,
        )


class BiRNNEmbedding(RNNEmbedding):
    def __init__(
        self,
        embedding: lookup.LookupEmbedding,
        cell_type: Literal["GRU", "LSTM", "RNN"],
        embed_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(
            embedding,
            cell_type,
            embed_size,
            num_layers,
            bias,
            dropout,
            bidirectional=True,
        )
