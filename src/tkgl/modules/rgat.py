from typing import Callable

import dgl
import torch
from dgl.udf import EdgeBatch
from torch.nn import functional as f

from .utils import get_param


class RelGATLayer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        self_loop: bool,
        message_op: str,
        attention_op: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.self_loop = self_loop
        self.message_op = message_op
        self.attention_op = attention_op

        if self.message_op in ("u+e", "u"):
            self.msg_weight = get_param(input_size, num_heads, hidden_size)
        elif self.message_op == "uev":
            self.msg_weight = get_param(3 * input_size, num_heads, hidden_size)

        if self.attention_op == "uev":
            self.att_weight = get_param(num_heads, 3 * input_size)
        elif self.attention_op == "m":
            self.att_weight = get_param(num_heads, hidden_size)
        elif self.attention_op == "mv":
            self.att_weight = get_param(num_heads, input_size + hidden_size)
        self.leacky_relu = torch.nn.LeakyReLU()

        if self.self_loop:
            self.loop_weight = get_param(input_size, hidden_size)

        self.rel_weight = get_param(input_size, hidden_size)

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.RReLU()

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_types: torch.Tensor,
        rel_feats: torch.Tensor,
    ):
        # (E, H)
        def message_fn(edges: EdgeBatch):
            if self.message_op == "u+e":
                ori = edges.src["h"] + edges.data["h"]
            elif self.message_op == "u":
                ori = edges.src["h"]
            elif self.message_op == "uev":
                trip = [edges.src["h"], edges.data["h"], edges.dst["h"]]
                ori = torch.cat(trip, dim=-1)
            msg = torch.einsum("ei,ikh->ekh", ori, self.msg_weight)
            return {"msg": msg}

        def multi_score_fn(edges: EdgeBatch):
            if self.attention_op == "uev":
                trip = [edges.src["h"], edges.data["h"], edges.dst["h"]]
                trip_emb = torch.cat(trip, dim=-1)
                s = torch.einsum("ei,ki->ek", trip_emb, self.att_weight)
            elif self.attention_op == "m":
                msg = edges.data["msg"]
                s = torch.einsum("ekh,kh->ek", msg, self.att_weight)
            elif self.attention_op == "mv":
                base = torch.cat([msg, edges.dst["h"]], dim=-1)
                s = torch.einsum("ekh,kh->ek", base, self.att_weight)

            score = self.leacky_relu(s)
            return {"score": score}

        with graph.local_scope():
            graph.ndata["h"] = node_feats
            graph.edata["h"] = rel_feats[edge_types]

            # incoming information
            graph.apply_edges(message_fn)  # add `msg` in edges
            graph.apply_edges(multi_score_fn)  # add `score` in edges
            # (E, X, 1)
            multi_score = graph.edata.pop("score")
            multi_weight = dgl.ops.edge_softmax(
                graph, torch.unsqueeze(multi_score, dim=-1)
            )
            multi_weight = self.dropout(multi_weight)

            graph.edata["msg"] = multi_weight * graph.edata.pop("msg")
            graph.update_all(
                dgl.function.copy_edge("msg", "msg"),
                dgl.function.sum("msg", "msg"),
            )
            in_msg = torch.mean(graph.ndata.pop("msg"), dim=1)

            # outgoing information
            # rg = dgl.reverse(graph, copy_ndata=True, copy_edata=True)
            # rg.apply_edges(message_fn)
            # rg.apply_edges(multi_score_fn)
            # multi_score = rg.edata.pop("score")
            # multi_weight = dgl.ops.edge_softmax(
            #     rg, torch.unsqueeze(multi_score, dim=-1)
            # )
            # multi_weight = self.dropout(multi_weight)
            # rg.edata["msg"] = multi_weight * rg.edata.pop("msg")
            # rg.update_all(
            #     dgl.function.copy_edge("msg", "msg"),
            #     dgl.function.sum("msg", "msg"),
            # )
            # out_msg = torch.mean(rg.ndata.pop("msg"), dim=1)

        agg_msg = in_msg  # + out_msg
        if self.self_loop:
            self_msg = node_feats @ self.loop_weight
            agg_msg = agg_msg + self_msg
        agg_msg = self.activation(agg_msg)
        agg_msg = self.dropout(agg_msg)
        return agg_msg, rel_feats @ self.rel_weight


class RelGAT(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        self_loop: bool,
        message_op: str,
        attention_op: str,
    ):
        super().__init__()
        layer_class = RelGATLayer
        layer_list = [
            layer_class(
                input_size,
                hidden_size,
                num_heads,
                dropout,
                self_loop,
                message_op,
                attention_op,
            )
        ]
        for _ in range(1, num_layers):
            layer_list.append(
                layer_class(
                    hidden_size,
                    hidden_size,
                    num_heads,
                    dropout,
                    self_loop,
                    message_op,
                    attention_op,
                )
            )
        self.layers = torch.nn.ModuleList(layer_list)

    __call__: Callable[
        [
            "RelGAT",
            dgl.DGLGraph,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        torch.Tensor,
    ]

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_types: torch.Tensor,
        rel_feats: torch.Tensor,
    ):
        for layer in self.layers:
            node_feats, rel_feats = layer(
                graph, node_feats, edge_types, rel_feats
            )
        return node_feats
