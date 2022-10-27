import dataclasses
from typing import List

import dgl
import torch
from torch.nn import functional as f

from tkgl.data import unpack_graph


@dataclasses.dataclass(frozen=True)
class CyGNetConfig:
    hidden_size: int
    num_ents: int
    num_rels: int


class CyGNetModel(torch.nn.Module):
    def __init__(self, cfg: CyGNetConfig):
        super().__init__()
        self.cfg = cfg
        self.ent_emb = torch.nn.Parameter(
            torch.empty(cfg.num_ents, cfg.hidden_size)
        )
        self.rel_emb = torch.nn.Parameter(
            torch.empty(cfg.num_rels, cfg.hidden_size)
        )
        self.tim_emb = torch.nn.Parameter(torch.empty(1, cfg.hidden_size))

        self.glinear = torch.nn.Sequential(
            torch.nn.Linear(3 * self.cfg.hidden_size, self.cfg.hidden_size),
        )
        self.clinear = torch.nn.Sequential(
            torch.nn.Linear(3 * self.cfg.hidden_size, self.cfg.hidden_size),
            torch.nn.Tanh(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.ent_emb, gain=torch.nn.init.calculate_gain("relu")
        )
        torch.nn.init.xavier_uniform_(
            self.rel_emb, gain=torch.nn.init.calculate_gain("relu")
        )
        torch.nn.init.xavier_uniform_(
            self.tim_emb, gain=torch.nn.init.calculate_gain("relu")
        )

    def regularize(self):
        x = torch.mean(self.rel_emb.pow(2))
        x += torch.mean(self.ent_emb.pow(2))
        x += torch.mean(self.tim_emb.pow(2))
        return x

    def forward(self, hist_graphs: List[dgl.DGLGraph], graph: dgl.DGLGraph):
        vocab = torch.zeros(
            self.cfg.num_ents, self.cfg.num_rels, self.cfg.num_ents
        )
        print("forward")
        for graph in hist_graphs:
            sub, rel, obj = unpack_graph(graph)
            vocab[sub, rel, obj] += 1

        sub, rel, obj = unpack_graph(graph)
        sub_emb = self.ent_emb[sub]
        rel_emb = self.rel_emb[rel]
        tim_emb = self.tim_emb.repeat(sub.size(0), 1)

        x = torch.cat([sub_emb, rel_emb, tim_emb], dim=1)
        # generate
        gscore = self.glinear(x)
        # copy
        mask = torch.sum(vocab[sub, rel])
        mask = torch.masked_fill(mask != 0, 1)
        cscore = torch.softmax(self.clinear(x) + mask, dim=1)

        fscore = gscore + self.cfg.alpha * (cscore - gscore)
        return {"score": torch.log(fscore), "reg": x}


class CyGNetLoss(torch.nn.Module):
    def __init__(self, reg_term: float = 0.01):
        super().__init__()
        self.reg_term = reg_term

    def forward(
        self, score: torch.Tensor, reg: torch.Tensor, graph: dgl.DGLGraph
    ):
        sub, rel, obj = unpack_graph(graph)
        loss = f.nll_loss(score, obj)
        return loss + self.reg_term * reg
