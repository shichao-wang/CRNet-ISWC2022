from html import entities

import dgl
import torch
from dgl.udf import EdgeBatch


def node_neighbor_subgraph(graph: dgl.DGLGraph, ent_ids: torch.Tensor):
    node_ids = torch.nonzero(graph.ndata["ent_id"] == ent_ids[..., None])[:, 1]

    adj_mat: torch.Tensor = torch.Tensor.to_dense(graph.adj())
    degrees = torch.sum(adj_mat[node_ids, :] + adj_mat[:, node_ids].t(), dim=0)
    degrees[node_ids] += 1
    subgraph_nodes = torch.nonzero(degrees)[:, 0].to(ent_ids)
    sg: dgl.DGLGraph = dgl.node_subgraph(graph, subgraph_nodes)
    return sg


def edge_neighbor_subgraph(graph: dgl.DGLGraph, ent_ids: torch.Tensor):
    def is_adj(edges: EdgeBatch):
        dst_mask = torch.count_nonzero(
            edges.dst["ent_id"] == ent_ids[..., None], dim=0
        )
        src_mask = torch.count_nonzero(
            edges.src["ent_id"] == ent_ids[..., None], dim=0
        )
        mask = torch.bitwise_or(dst_mask, src_mask)
        return mask

    return dgl.edge_subgraph(graph, graph.filter_edges(is_adj))


u = torch.tensor([1, 1, 2, 3])
v = torch.tensor([2, 3, 4, 5])

graph = dgl.graph((u, v), device="cuda")
graph.ndata["ent_id"] = torch.tensor([1, 2, 3, 4, 5, 6], device="cuda")
subj = torch.tensor([2, 4], device="cuda")
sg = node_neighbor_subgraph(graph, subj)

print(sg)
print(sg.ndata["ent_id"])
print(subj[..., None])
print(sg.ndata["ent_id"] == subj[..., None])
print(torch.nonzero(sg.ndata["ent_id"] == subj[..., None]))

print(subj[..., None])
