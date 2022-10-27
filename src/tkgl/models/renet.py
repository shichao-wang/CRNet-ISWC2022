from typing import List

import dgl
import torch
from dgl.udf import EdgeBatch
from torch import nn

from tkgl.models.tkgr_model import TkgrModel


def edge_neighbor_subgraph(graph: dgl.DGLGraph, ent_ids: torch.Tensor):
    def is_adj(edges: EdgeBatch):
        dst_mask = torch.sum(
            edges.dst["eid"] == ent_ids[..., None], dim=0, dtype=torch.bool
        )
        src_mask = torch.sum(
            edges.src["eid"] == ent_ids[..., None], dim=0, dtype=torch.bool
        )
        mask = torch.bitwise_or(dst_mask, src_mask)
        return mask

    edges = graph.filter_edges(is_adj)
    cgraph = graph.cpu()
    sg = dgl.edge_subgraph(
        cgraph, torch.Tensor.cpu(edges), relabel_nodes=False
    ).to(graph.device)
    return sg


def node_neighbor_subgraph(graph: dgl.DGLGraph, ent_ids: torch.Tensor):
    node_ids = torch.nonzero(graph.ndata["eid"] == ent_ids[..., None])[:, -1]
    adj_mat: torch.Tensor = torch.Tensor.to_dense(graph.adj())
    degrees = torch.sum(adj_mat[node_ids, :] + adj_mat[:, node_ids].t(), dim=0)
    degrees[node_ids] += 1
    subgraph_nodes = torch.nonzero(degrees)[:, 0].to(ent_ids)
    sg: dgl.DGLGraph = dgl.node_subgraph(graph, subgraph_nodes)
    return sg


class RENet(TkgrModel):
    def __init__(
        self, num_ents: int, num_rels: int, hidden_size: int, num_layers: int
    ):
        super().__init__(num_ents, num_rels, hidden_size)

        self._rgcn = RelGraphConv(hidden_size, hidden_size, num_layers)

        self._obj_decoder = RecurrentE(num_ents, hidden_size, hidden_size)
        self._rel_decoder = RecurrentR(num_rels, hidden_size, hidden_size)

        self.ent_emb = nn.Parameter(torch.zeros(num_ents, hidden_size))
        self.rel_emb = nn.Parameter(torch.zeros(num_rels, hidden_size))
        nn.init.xavier_normal_(self.ent_emb)
        nn.init.xavier_uniform_(self.rel_emb)

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
        obj: torch.Tensor,
    ):
        subj_hists = [
            edge_neighbor_subgraph(graph, subj) for graph in hist_graphs
        ]
        bg = dgl.batch(subj_hists)
        node_feats = self._rgcn(bg, self.ent_emb, self.rel_emb)
        batch_num_nodes_list = bg.batch_num_nodes().tolist()
        total_feats_split = torch.split(node_feats, batch_num_nodes_list)
        total_ent_split = torch.split(bg.ndata["ent_id"], batch_num_nodes_list)
        subj_feats = []
        for ent_ids, feats in zip(total_ent_split, total_feats_split):
            indexes = torch.nonzero(ent_ids == subj.unsqueeze(dim=-1))[:, 1]
            subj_feats.append(feats[indexes])
        subj_embeds = torch.stack(subj_feats, dim=1)

        # (hist_len, 3 * h)
        obj_logit = self._obj_decoder(
            self.ent_emb, self.rel_emb, subj, rel, subj_embeds
        )
        rel_logit = self._rel_decoder(self.ent_emb, subj, obj, subj_embeds)
        return {"obj_logit": obj_logit, "rel_logit": rel_logit}


class RecurrentE(torch.nn.Module):
    def __init__(self, num_classes: int, input_size: int, hidden_size: int):
        super().__init__()
        self.rnn = nn.GRU(3 * input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(2 * input_size + hidden_size, num_classes)

    def forward(
        self,
        ent_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        subj: torch.Tensor,
        rel: torch.Tensor,
        subj_embeds: torch.Tensor,
    ):
        hist_len = subj_embeds.size(1)
        ex_ent_embeds = ent_emb[subj].unsqueeze(1).repeat(1, hist_len, 1)
        ex_rel_embeds = rel_emb[rel].unsqueeze(1).repeat(1, hist_len, 1)
        # (num_preds, hist_len, h)
        inputs = torch.cat([subj_embeds, ex_ent_embeds, ex_rel_embeds], dim=-1)
        _, output = self.rnn(inputs)
        x = torch.cat(
            [ent_emb[subj], rel_emb[rel], torch.squeeze(output, dim=0)],
            dim=-1,
        )
        logit = self.linear(x)
        return logit


class RecurrentR(torch.nn.Module):
    def __init__(self, num_classes: int, input_size: int, hidden_size: int):
        super().__init__()
        self._rnn = nn.GRU(3 * input_size, hidden_size, batch_first=True)
        self._linear = nn.Linear(2 * input_size + hidden_size, num_classes)

    def forward(
        self,
        ent_embeds: torch.Tensor,
        subj: torch.Tensor,
        obj: torch.Tensor,
        subj_embeds: torch.Tensor,
    ):
        hist_len = subj_embeds.size(1)
        ex_ent_embeds = ent_embeds[subj].unsqueeze(1).repeat(1, hist_len, 1)
        ex_rel_embeds = ent_embeds[obj].unsqueeze(1).repeat(1, hist_len, 1)
        inputs = torch.cat([subj_embeds, ex_ent_embeds, ex_rel_embeds], dim=-1)
        _, output = self._rnn(inputs)
        x = torch.cat(
            [ent_embeds[subj], ent_embeds[obj], torch.squeeze(output, dim=0)],
            dim=-1,
        )
        logit = self._linear(x)
        return logit
