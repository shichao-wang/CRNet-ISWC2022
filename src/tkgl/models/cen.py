from typing import List, Tuple

import dgl
import torch
from torch.nn import functional as f

from .regcn import OmegaRelGraphConv
from .tkgr_model import TkgrModel


class LengthAwareConvTransE(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hist_len: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        conv_layers = [
            torch.nn.Sequential(
                torch.nn.BatchNorm1d(in_channels),
                torch.nn.Dropout(dropout),
                torch.nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.BatchNorm1d(out_channels),
                torch.nn.Dropout(dropout),
            )
            for _ in range(hist_len)
        ]
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.fc_linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * out_channels, hidden_size),
        )
        out_layers = [
            torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.BatchNorm1d(hidden_size),
                torch.nn.ReLU(),
            )
            for _ in range(hist_len)
        ]
        self.out_layers = torch.nn.ModuleList(out_layers)

    def forward(self, tuple_feats: torch.Tensor, i: int = None):
        if i is None:
            return self.batch_forward(tuple_feats)

        feature_maps = self.conv_layers[i](tuple_feats)
        feat_hid = feature_maps.view(feature_maps.size(0), -1)
        feat_score = self.fc_linear(feat_hid)
        feat_score = self.out_layers[i](feat_score)
        return feat_score

    def batch_forward(self, hist_tuple_feats: torch.Tensor):
        score_list = []
        for i in range(hist_tuple_feats.size(0)):
            feature_maps = self.conv_layers[i](hist_tuple_feats[i])
            feat_hid = feature_maps.view(feature_maps.size(0), -1)
            feat_score = self.fc_linear(feat_hid)
            feat_score = self.out_layers[i](feat_score)
            score_list.append(feat_score)
        hist_score = torch.stack(score_list)
        return hist_score


class ComplexEvoNet(TkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        hist_len: int,
        dropout: float,
        rgcn_num_layers: int,
        rgcn_self_loop: bool,
        convtranse_kernel_size: int,
        convtranse_channels: int,
        norm_embeds: bool = True,
    ):
        super().__init__(num_ents, num_rels, hidden_size)
        self.rgcn = OmegaRelGraphConv(
            hidden_size, hidden_size, rgcn_num_layers, rgcn_self_loop, dropout
        )
        self.glinear = torch.nn.Linear(hidden_size, hidden_size)
        self.convtranse = LengthAwareConvTransE(
            hidden_size,
            hist_len,
            2,
            convtranse_channels,
            convtranse_kernel_size,
            dropout,
        )
        self._norm_embeds = norm_embeds

    def evolve(self, hist_graphs: List[dgl.DGLGraph]) -> torch.Tensor:
        ent_emb = self._origin_or_norm(self.ent_emb)

        for graph in hist_graphs:
            node_feats = ent_emb[graph.ndata["eid"]]
            edge_feats = self.rel_emb[graph.edata["rid"]]
            neigh_feats = self.rgcn(graph, node_feats, edge_feats)
            neigh_feats = self._origin_or_norm(neigh_feats)
            ent_neigh_emb = neigh_feats[torch.argsort(graph.ndata["eid"])]
            u = torch.sigmoid(self.glinear(ent_emb))
            ent_emb = f.normalize(u * ent_neigh_emb + (1 - u) * ent_emb)

        return ent_emb

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
    ):
        """

        Arguments:
            snapshot: [his_len]
            triplets: (num_triplets, 3)

        Returns:
            logits: (num_triplets, num_entities)
        """
        hist_len = len(hist_graphs)
        obj_logit_list = []
        for i in range(hist_len):
            ent_emb = self.evolve(hist_graphs[i:])
            obj_inp = torch.stack([ent_emb[subj], self.rel_emb[rel]], dim=1)
            obj_logit = self.convtranse(obj_inp, i) @ ent_emb.t()
            obj_logit_list.append(obj_logit)

        hist_obj_logit = torch.stack(obj_logit_list)
        obj_logit = hist_obj_logit.softmax(dim=-1).sum(dim=0)

        return {
            "hist_obj_logit": hist_obj_logit,
            "obj_logit": obj_logit,  # used for prediction
        }

    def _origin_or_norm(self, tensor: torch.Tensor):
        if self._norm_embeds:
            return f.normalize(tensor)
        return tensor


class HistEntLoss(torch.nn.Module):
    def forward(self, hist_obj_logit: torch.Tensor, obj: torch.Tensor):
        loss = obj.new_tensor(0.0)
        hist_len = hist_obj_logit.size(0)
        for i in range(hist_len):
            obj_logit = hist_obj_logit[i]
            loss = loss + f.cross_entropy(obj_logit, obj)  # / (i + 1)
        return loss
