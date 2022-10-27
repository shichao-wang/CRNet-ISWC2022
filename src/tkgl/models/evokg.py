from typing import List

import dgl
import torch

from tkgl.models.tkgr_model import TkgrModel
from tkgl.modules.rgcn import RGCN


class EvoKg(TkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__(num_ents, num_rels, hidden_size)
        self._rgcn = RGCN(
            hidden_size, hidden_size, num_rels, num_layers, dropout
        )
        self._rnn1 = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self._rnn2 = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self._linear1 = torch.nn.Linear(3 * hidden_size, num_rels)
        self._linear2 = torch.nn.Linear(5 * hidden_size, num_ents)

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
        obj: torch.Tensor,
    ):
        bg = dgl.batch(hist_graphs)
        total_nfeats = self._rgcn(bg, self.ent_emb[bg.ndata["eid"]])
        hist_nfeat_list = torch.split_with_sizes(
            total_nfeats, bg.batch_num_nodes().tolist()
        )
        hist_nfeats = torch.stack(hist_nfeat_list, dim=1)
        _, ent_hid = self._rnn1(hist_nfeats)
        ent_hid = torch.squeeze(ent_hid, dim=0)

        hist_rfeat_list = []
        for nfeats, graph in zip(hist_nfeat_list, hist_graphs):
            hist_rfeat_list.append(self._agg_rel_nodes(graph, nfeats))
        hist_rfeats = torch.stack(hist_rfeat_list, dim=1)
        _, rel_hid = self._rnn2(hist_rfeats)
        rel_hid = torch.squeeze(rel_hid, dim=0)

        graph_pred_embed = torch.max_pool1d(
            hist_nfeats.transpose(-1, -2), len(hist_graphs)
        ).squeeze(dim=-1)
        rel_pred_embed = torch.cat(
            [graph_pred_embed[subj], self.ent_emb[subj], ent_hid[subj]],
            dim=-1,
        )
        rel_logit = self._linear1(rel_pred_embed)
        obj_pred_embed = torch.cat(
            [rel_pred_embed, self.rel_emb[rel], rel_hid[rel]], dim=-1
        )
        ent_logit = self._linear2(obj_pred_embed)
        return {"obj_logit": ent_logit * rel_logit, "rel_logit": rel_logit}

    def _agg_rel_nodes(self, graph: dgl.DGLGraph, node_feats: torch.Tensor):
        """
        Arguments:
            nfeats: (num_nodes, hidden_size)
        Return:
            (num_rels, hidden_size)
        """
        # (num_rels, num_nodes)
        rel_node_mask = node_feats.new_zeros(
            self.rel_emb.size(0), node_feats.size(0), dtype=torch.bool
        )
        src, dst, eids = graph.edges("all")
        rel_ids = graph.edata["rid"][eids]
        rel_node_mask[rel_ids, src] = True
        rel_node_mask[rel_ids, dst] = True

        node_ids = torch.nonzero(rel_node_mask)[:, 1]
        rel_embeds = dgl.ops.segment_reduce(
            rel_node_mask.sum(dim=1), node_feats[node_ids], "mean"
        )
        return torch.nan_to_num(rel_embeds, 0)
