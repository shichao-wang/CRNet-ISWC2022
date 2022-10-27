import dgl
import torch
from dgl.udf import EdgeBatch, NodeBatch

graph = dgl.graph(([0, 0, 1], [1, 2, 2]), num_nodes=4)


def msg(edges: EdgeBatch):
    return {"h": edges.src["h"]}


graph.ndata["h"] = torch.arange(1, graph.num_nodes() + 1).float()

graph.update_all(msg, dgl.function.mean("h", "h"))
neigh_msg = graph.ndata["h"]
print(neigh_msg)
