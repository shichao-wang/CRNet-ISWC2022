from typing import Dict, Hashable, List, Tuple, TypedDict

import dgl
import torch


class Quadruple(TypedDict):
    subj: Hashable
    rel: Hashable
    obj: Hashable
    mmt: Hashable


class TkgrFeature(TypedDict):
    hist_graphs: List[dgl.DGLGraph]
    graph: dgl.DGLGraph
    sr_dict: Dict[int, Dict[int, List[int]]]
    so_dict: Dict[int, Dict[int, List[int]]]
    # all_obj_mask: torch.Tensor
    # all_rel_mask: torch.Tensor


def unpack_graph(
    graph: dgl.DGLGraph,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src, dst, eid = graph.edges("all")
    sub = graph.ndata["eid"][src]
    rel = graph.edata["rid"][eid]
    obj = graph.ndata["eid"][dst]
    return sub, rel, obj


def group_reduce_nodes(
    graph: dgl.DGLGraph,
    node_feats: torch.Tensor,
    edge_types: torch.Tensor,
    num_nodes: int = None,
    num_rels: int = None,
    *,
    reducer: str = "mean",
):
    in_degree, out_degree = build_r2n_degree(
        graph, edge_types, num_rels, num_nodes
    )
    rn_mask = torch.bitwise_or(in_degree.bool(), out_degree.bool())

    nids = torch.nonzero(rn_mask)[:, 1]
    rel_emb = dgl.ops.segment_reduce(
        rn_mask.sum(dim=1), node_feats[nids], reducer
    )
    return torch.nan_to_num(rel_emb, 0)


def graph_to_triplets(
    graph: dgl.DGLGraph, ent_embeds: torch.Tensor, rel_embeds: torch.Tensor
) -> torch.Tensor:
    """
    Returns:
        (T, 3, H)
    """

    src, dst, eid = graph.edges("all")
    subj = graph.ndata["eid"][src]
    rel = graph.edata["rid"][eid]
    obj = graph.ndata["eid"][dst]
    embed_list = [ent_embeds[subj], rel_embeds[rel], ent_embeds[obj]]
    return torch.stack(embed_list, dim=1)


def build_r2n_degree(
    graph: dgl.DGLGraph,
    edge_types: torch.Tensor,
    num_rels: int,
    num_nodes: int = None,
):
    if num_nodes is None:
        num_nodes = graph.num_nodes()

    in_degrees = torch.zeros(
        num_rels, num_nodes, dtype=torch.long, device=graph.device
    )
    out_degrees = torch.zeros(
        num_rels, num_nodes, dtype=torch.long, device=graph.device
    )
    src, dst, eids = graph.edges("all")
    rel_ids = edge_types[eids]
    in_degrees[rel_ids, src] += 1
    out_degrees[rel_ids, dst] += 1
    return in_degrees, out_degrees
