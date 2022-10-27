import abc
from typing import Callable, TypedDict

import torch
from torch import nn


class Embedding(nn.Module):
    class Output(TypedDict):
        embedding: torch.Tensor
        mask: torch.Tensor

    @property
    @abc.abstractmethod
    def num_embeds(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def embed_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def pad_id(self) -> int:
        pass


class SequenceEmbedding(Embedding):
    class Output(TypedDict):
        hidden_states: torch.Tensor
        pooled_output: torch.Tensor
        mask: torch.Tensor

    __call__: Callable[[torch.Tensor, torch.Tensor], Output]

    @abc.abstractmethod
    def forward(self, token_ids: torch.Tensor) -> Output:
        pass
