import abc

import torch

from .modules.convtranse import ConvTransE


class NodeScoreFunction(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        heads: torch.Tensor,
        rels: torch.Tensor,
        obj_emb: torch.Tensor,
    ):
        raise NotImplementedError()


class RelScoreFunction(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        heads: torch.Tensor,
        tails: torch.Tensor,
        rel_emb: torch.Tensor,
    ):
        raise NotImplementedError()


class ConvTransENS(NodeScoreFunction):
    def __init__(
        self,
        hidden_size: int,
        num_channels: int,
        kernel_size: int,
        dropout: int,
    ):
        super().__init__()
        self._convetrans = ConvTransE(
            hidden_size,
            in_channels=2,
            out_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        heads: torch.Tensor,
        rels: torch.Tensor,
        obj_emb: torch.Tensor,
    ):
        x = torch.stack([heads, rels], dim=1)
        return self._convetrans(x, obj_emb)


class ConvTransERS(RelScoreFunction):
    def __init__(
        self,
        hidden_size: int,
        num_channels: int,
        kernel_size: int,
        dropout: int,
    ):
        super().__init__()
        self._convetrans = ConvTransE(
            hidden_size,
            in_channels=2,
            out_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        heads: torch.Tensor,
        tails: torch.Tensor,
        rel_emb: torch.Tensor,
    ):
        x = torch.stack([heads, tails], dim=1)
        return self._convetrans(x, rel_emb)
