import logging
from typing import List

import dgl
import torch
from torch.nn import functional as tf

from tkgl.data import unpack_graph

logger = logging.getLogger(__name__)


class JointLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self._alpha = alpha

    def forward(
        self,
        obj_logit: torch.Tensor,
        obj: torch.Tensor,
        rel_logit: torch.Tensor,
        rel: torch.Tensor,
    ):
        rel_loss = tf.cross_entropy(rel_logit, rel)
        ent_loss = tf.cross_entropy(obj_logit, obj)
        return ent_loss * self._alpha + rel_loss * (1 - self._alpha)


class EntLoss(torch.nn.Module):
    def forward(
        self,
        obj_logit: torch.Tensor,
        graph: dgl.DGLGraph,
    ):
        sub, rel, obj = unpack_graph(graph)
        loss = tf.cross_entropy(obj_logit, obj)
        return loss
        # f = 1.0
        # hist_loss = loss.new_tensor(0.0)
        # for logit, g in zip(hist_logits, hist_graphs):
        #     f = f / 2
        #     sub, rel, obj = unpack_graph(g)
        #     hist_loss = hist_loss + f * tf.cross_entropy(logit, obj)
        # return loss + (hist_loss / len(hist_graphs))
