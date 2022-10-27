from typing import Callable

import dgl
import torch
from dgl.udf import EdgeBatch


class CompGCN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_rels: int,
        num_layers: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        dropout: float = 0.0,
    ):
        super().__init__()
        layer = CompGCNLayer
        self._layers = torch.nn.ModuleList(
            [layer(input_size, hidden_size, num_rels, activation, dropout)]
        )
        for _ in range(1, num_layers):
            self._layers.append(
                layer(input_size, hidden_size, num_rels, activation, dropout)
            )

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_types: torch.Tensor,
    ):
        nfeats = node_feats
        for layer in self._layers:
            nfeats = layer(graph, nfeats, edge_types)
        return nfeats


class CompGCNLayer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_rels: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rel_weights = torch.nn.Parameter(
            torch.empty(num_rels, input_size, hidden_size)
        )
        self._linear = torch.nn.Linear(input_size, hidden_size)
        self._activation = activation
        self._dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_types: torch.Tensor,
    ):
        def message_fn(edges: EdgeBatch):
            rweights = self.rel_weights[edges.data["rtype"]]
            msg = edges.src["h"] @ rweights
            return {"msg": msg}

        node_feats = self._dropout(node_feats)
        with graph.local_scope():
            graph.ndata["h"] = node_feats
            graph.edata["rtype"] = edge_types
            # node msg
            graph.update_all(message_fn, dgl.function.mean("msg", "msg"))

            node_msg = graph.ndata["msg"]

        return self._activation(node_msg + self._linear(node_feats))
