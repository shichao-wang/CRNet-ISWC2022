from typing import Dict, List

import dgl
import torch
import torchmetrics

from tkgl.data import unpack_graph


class RankMetric(torchmetrics.Metric):
    ranks: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("ranks", torch.tensor([]))

    def update(self, logit: torch.Tensor, target: torch.Tensor):
        ranks = logit_ranks(logit, target)
        self.ranks = torch.cat([self.ranks, ranks], dim=-1)


class FilteredRankMetric(RankMetric):
    def update(
        self,
        logit: torch.Tensor,
        target: torch.Tensor,
        all_target_mask: torch.Tensor,
    ):
        """
        Arguments:
            logit: (B, N)
            target: (B, )
            all_target_mask: (B, N)
        """
        bs = logit.size(0)
        ground_truth = logit[range(bs), target]
        logit[all_target_mask] = 0
        logit[range(bs), target] = ground_truth
        super().update(logit, target)


class FilteredMRR(FilteredRankMetric):
    def update(
        self,
        obj_logit: torch.Tensor,
        obj: torch.Tensor,
        all_obj_mask: torch.Tensor,
    ):
        return super().update(obj_logit, obj, all_obj_mask)

    def compute(self):
        return {"fe_mrr": mrr_value(self.ranks)}


def logit_ranks(logit: torch.Tensor, target: torch.Tensor):
    total_ranks = logit.argsort(dim=-1, descending=True)
    ranks = torch.nonzero(total_ranks == target.view(-1, 1))[:, 1]
    return ranks


def mrr_value(ranks: torch.Tensor):
    return torch.mean(1.0 / (ranks.float() + 1))


def hit_value(ranks: torch.Tensor, k: int):
    return (ranks < k).float().mean()


class EntMRR(RankMetric):
    def update(self, obj_logit: torch.Tensor, obj: torch.Tensor) -> None:
        return super().update(logit=obj_logit, target=obj)

    def compute(self):
        return {"e_mrr": mrr_value(self.ranks)}


class EntMetric(torchmetrics.Metric):
    ranks: torch.Tensor
    franks: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("ranks", torch.tensor([]))
        self.add_state("franks", torch.tensor([]))

    def update(
        self,
        obj_logit: torch.Tensor,
        graph: dgl.DGLGraph,
        sr_dict: Dict[int, Dict[int, List[int]]],
    ):
        sub, rel, obj = unpack_graph(graph)
        # raw metric
        ranks = logit_ranks(obj_logit, obj)
        self.ranks = torch.cat([self.ranks, ranks], dim=-1)
        # filetered metric

        obj_flogit = obj_logit.clone().softmax(dim=-1)
        for i, (s, r, o) in enumerate(
            zip(sub.tolist(), rel.tolist(), obj.tolist())
        ):
            valid_o = list(sr_dict[s][r])  # deep copy
            valid_o.remove(o)
            obj_flogit[i][valid_o] = 0
        franks = logit_ranks(obj_flogit, obj)
        self.franks = torch.cat([self.franks, franks], dim=-1)

    def compute(self):
        values = {}
        values["e_mrr"] = mrr_value(self.ranks)
        for k in (1, 3, 10):
            values[f"e_hit@{k}"] = hit_value(self.ranks, k)

        values["e_fmrr"] = mrr_value(self.franks)
        for k in (1, 3, 10):
            values[f"e_fhit@{k}"] = hit_value(self.franks, k)
        return values


class JointMetric(torchmetrics.Metric):
    ent_ranks: torch.Tensor
    rel_ranks: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("ent_ranks", torch.tensor([]))
        self.add_state("rel_ranks", torch.tensor([]))

    def update(
        self,
        obj_logit: torch.Tensor,
        rel_logit: torch.Tensor,
        obj: torch.Tensor,
        rel: torch.Tensor,
    ):
        ent_ranks = logit_ranks(obj_logit, obj)
        self.ent_ranks = torch.cat([self.ent_ranks, ent_ranks], dim=-1)

        rel_ranks = logit_ranks(rel_logit, rel)
        self.rel_ranks = torch.cat([self.rel_ranks, rel_ranks], dim=-1)

    def compute(self):
        values = {}
        values["e_mrr"] = mrr_value(self.ent_ranks)
        for k in (1, 3, 10):
            values[f"e_hit@{k}"] = hit_value(self.ent_ranks, k)
        values["r_mrr"] = mrr_value(self.rel_ranks)
        for k in (1, 3, 10):
            values[f"r_hit@{k}"] = hit_value(self.rel_ranks, k)
        return values
