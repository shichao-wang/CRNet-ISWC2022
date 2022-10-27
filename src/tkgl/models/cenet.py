import os
import random
from typing import List

import dgl
import molurus
import torch
import torch.nn.functional as f
from molurus import hierdict

from tkgl.data import build_r2n_degree, unpack_graph
from tkgl.modules.convtranse import ConvTransE
from tkgl.modules.rgat import RelGAT
from tkgl.modules.utils import get_param


def build_candidate_subgraph(
    num_nodes: int,
    total_sub: torch.Tensor,
    total_rel: torch.Tensor,
    total_obj_logit: torch.Tensor,
    k: int,
    num_partitions: int,
) -> dgl.DGLGraph:
    _, total_topk_obj = torch.topk(total_obj_logit, k=k)
    num_queries = total_sub.size(0)
    rng = torch.Generator().manual_seed(1234)
    total_indices = torch.randperm(num_queries, generator=rng)

    graph_list = []
    for indices in torch.tensor_split(total_indices, num_partitions):
        topk_obj = total_topk_obj[indices]
        sub = torch.repeat_interleave(total_sub[indices], k)
        rel = torch.repeat_interleave(total_rel[indices], k)
        obj = topk_obj.view(-1)
        graph = dgl.graph(
            (sub, obj),
            num_nodes=num_nodes,
            device=total_sub.device,
        )
        graph.ndata["eid"] = torch.arange(num_nodes, device=graph.device)
        graph.edata["rid"] = rel
        graph_list.append(graph)
    return dgl.batch(graph_list)


class CENet(torch.nn.Module):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        dropout: float,
        rgcn_num_layers: int,
        rgcn_num_heads: int,
        rgcn_self_loop: bool,
        rgcn_message_op: str,
        rgcn_attention_op: str,
        convtranse_kernel_size: int,
        convtranse_channels: int,
        evolve_rel: bool,
    ):
        super().__init__()
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.hidden_size = hidden_size
        self.evolve_rel = evolve_rel
        self.mrgcn = RelGAT(
            hidden_size,
            hidden_size,
            rgcn_num_heads,
            rgcn_num_layers,
            dropout,
            rgcn_self_loop,
            rgcn_message_op,
            rgcn_attention_op,
        )
        self.ent_gate = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_size, hidden_size), torch.nn.Sigmoid()
        )
        # self.entrnn = torch.nn.GRUCell(hidden_size, hidden_size)
        # self.relrnn = torch.nn.GRUCell(2 * hidden_size, hidden_size)
        self.rel_weight = get_param(hidden_size, hidden_size)
        self.ent_weight = get_param(hidden_size, hidden_size)
        self.convtranse = ConvTransE(
            hidden_size,
            hidden_size,
            2,
            convtranse_channels,
            convtranse_kernel_size,
            dropout,
        )
        self.ent_emb = get_param(num_ents, hidden_size)
        self.rel_emb = get_param(num_rels, hidden_size)

    def evolve(
        self,
        graphs: List[dgl.DGLGraph],
        ent_emb: torch.Tensor,
        rel_emb: torch.Tensor,
    ):
        logit_list = []
        for graph in graphs:
            # predict first
            sub, rel, obj = unpack_graph(graph)
            logit = self.convtranse(ent_emb[sub], rel_emb[rel]) @ ent_emb.t()
            logit_list.append(logit)

            # entity evolution
            if self.evolve_rel:
                new_rel_emb = f.normalize(rel_emb @ self.rel_weight)
            else:
                new_rel_emb = f.normalize(rel_emb)

            node_feats = ent_emb[graph.ndata["eid"]]
            neigh_feats = f.normalize(
                self.mrgcn(graph, node_feats, graph.edata["rid"], rel_emb)
            )
            cur_ent_emb = neigh_feats[torch.argsort(graph.ndata["eid"])]
            gate_inp = torch.cat([cur_ent_emb, ent_emb], dim=-1)
            u = self.ent_gate(gate_inp)
            new_ent_emb = f.normalize(u * cur_ent_emb + (1 - u) * ent_emb)

            ent_emb = new_ent_emb
            rel_emb = new_rel_emb

        return ent_emb, rel_emb, logit_list

    def forward(self, hist_graphs: List[dgl.DGLGraph], graph: dgl.DGLGraph):
        sub, rel, obj = unpack_graph(graph)
        ent_emb, rel_emb, hist_logits = self.evolve(
            hist_graphs,
            f.normalize(self.ent_emb),
            f.normalize(self.rel_emb),
        )
        obj_logit = self.convtranse(ent_emb[sub], rel_emb[rel]) @ ent_emb.t()
        return {"obj_logit": obj_logit, "hist_logits": hist_logits}


class CENetCP(torch.nn.Module):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        dropout: float,
        rgcn_num_layers: int,
        rgcn_num_heads: int,
        rgcn_self_loop: bool,
        rgcn_message_op: str,
        rgcn_attention_op: str,
        convtranse_kernel_size: int,
        convtranse_channels: int,
        k: int,
        logit_weight: float,
        evolve_rel: bool,
        cgraph_partitions: int,
        backbone_state_path: str,
        backbone_config_path: str,
        fine_tune: bool,
    ):
        super().__init__()
        if backbone_state_path is None:
            self.backbone = CENet(
                num_ents,
                num_rels,
                hidden_size,
                dropout,
                rgcn_num_layers,
                rgcn_num_heads,
                rgcn_self_loop,
                rgcn_message_op,
                rgcn_attention_op,
                convtranse_kernel_size,
                convtranse_channels,
                evolve_rel,
            )
        else:
            if backbone_config_path is None:
                backbone_config_path = os.path.join(
                    os.path.dirname(backbone_state_path), "config.yml"
                )
            cfg = hierdict.load(open(backbone_config_path))
            kwargs = molurus.build_fn_kwargs(
                CENet.__init__,
                cfg["model"],
            )
            assert kwargs["hidden_size"] == hidden_size
            self.backbone = CENet(num_ents, num_rels, **kwargs)
            self.backbone.load_state_dict(
                torch.load(backbone_state_path)["model"]
            )
            self.backbone.requires_grad_(fine_tune)
        self.k = k
        self.logit_weight = logit_weight
        self.cgraph_partitions = cgraph_partitions
        self.mrgcn = RelGAT(
            hidden_size,
            hidden_size,
            rgcn_num_heads,
            rgcn_num_layers,
            dropout,
            rgcn_self_loop,
            rgcn_message_op,
            rgcn_attention_op,
        )
        self.convtranse = ConvTransE(
            hidden_size,
            hidden_size,
            2,
            convtranse_channels,
            convtranse_kernel_size,
            dropout,
        )

    @property
    def num_ents(self):
        return self.backbone.num_ents

    @property
    def num_rels(self):
        return self.backbone.num_rels

    @property
    def ent_emb(self):
        return self.backbone.ent_emb

    @property
    def rel_emb(self):
        return self.backbone.rel_emb

    @property
    def cand_convtranse(self):
        return self.backbone.convtranse

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        graph: dgl.DGLGraph,
    ):
        """

        Arguments:
            snapshot: [his_len]
            triplets: (num_triplets, 3)

        Returns:
            logits: (num_triplets, num_entities)
        """
        # hist_len = len(hist_graphs)
        # hist_obj_list = []
        # for i in reversed(range(hist_len)):
        sub, rel, obj = unpack_graph(graph)
        ent_emb, rel_emb, _ = self.backbone.evolve(
            hist_graphs,
            f.normalize(self.ent_emb),
            f.normalize(self.rel_emb),
        )
        orig_obj_logit = (
            self.cand_convtranse(ent_emb[sub], rel_emb[rel]) @ ent_emb.t()
        )
        cand_graph = build_candidate_subgraph(
            self.num_ents,
            sub,
            rel,
            orig_obj_logit,
            self.k,
            self.cgraph_partitions,
        )
        total_neigh_feats = f.normalize(
            self.mrgcn(
                cand_graph,
                ent_emb[cand_graph.ndata["eid"]],
                cand_graph.edata["rid"],
                rel_emb,
            )
        )
        avg_neigh_feats = torch.split_with_sizes(
            total_neigh_feats, cand_graph.batch_num_nodes().tolist()
        )
        neigh_feats = torch.stack(avg_neigh_feats, dim=0).mean(dim=0)
        enhanced_ent_emb = neigh_feats

        enhanced_obj_logit = (
            self.convtranse(enhanced_ent_emb[sub], rel_emb[rel])
            @ enhanced_ent_emb.t()
        )

        obj_logit = (
            orig_obj_logit * (1 - self.logit_weight)
            + enhanced_obj_logit * self.logit_weight
        )
        return {"obj_logit": obj_logit}
