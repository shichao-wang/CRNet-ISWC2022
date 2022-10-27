import copy
import itertools
import logging
import os
import re
from collections import defaultdict
from typing import Counter, Dict, Iterator, List, Mapping, Optional, Tuple

import dgl
import numpy
import pandas
import torch
from tallow.data import datasets, vocabs
from tallow.data.datasets import Dataset
from tallow.data.vocabs import Vocab
from torch.nn import functional as f

from tkgl.data import Quadruple, TkgrFeature

logger = logging.getLogger(__name__)


def _sr_so_dict(quads: numpy.ndarray):
    subj = quads[:, 0]
    rel = quads[:, 1]
    obj = quads[:, 2]
    sr_dict = defaultdict(lambda: defaultdict(list))
    so_dict = defaultdict(lambda: defaultdict(list))
    for s, r, o in zip(subj, rel, obj):
        sr_dict[s][r].append(o)
        so_dict[s][o].append(r)
    return sr_dict, so_dict


def build_sr_so_dict_list(temporal_quads: List[numpy.ndarray]):
    sr_list, so_list = [], []
    for quads in temporal_quads:
        sr_dict, so_dict = _sr_so_dict(quads)
        sr_list.append(sr_dict)
        so_list.append(so_dict)
    return sr_list, so_list


# class TkgrDynamicDataset(Dataset[TkgrFeature]):
#     def __init__(
#         self,
#         temporal_quads: List[numpy.ndarray],
#         vocabs: Dict[str, Vocab],
#         hist_len: int,
#     ) -> None:
#         super().__init__()
#         self.vocabs = vocabs
#         self.hist_len = hist_len

#         self.temporal_graphs = [
#             build_knowledge_graph(quads, len(self.vocabs["ent"]))
#             for quads in temporal_quads
#         ]
#         self._sr_dict_list, self._so_dict_list = build_sr_so_dict_list(
#             temporal_quads
#         )

#     def __iter__(self) -> Iterator[TkgrFeature]:
#         for pid in range(1, len(self.temporal_graphs)):
#             xlice = slice(max(pid - self.hist_len, 0), pid)
#             hist_graphs = self.temporal_graphs[xlice]
#             quads = self._temporal_quads[pid]
#             subj = torch.from_numpy(quads[:, 0])
#             rel = torch.from_numpy(quads[:, 1])
#             obj = torch.from_numpy(quads[:, 2])
#             for i in range(len(hist_graphs) - 1, 0, -1):
#                 yield TkgrFeature(
#                     hist_graphs=hist_graphs[i:],
#                     subj=subj,
#                     rel=rel,
#                     obj=obj,
#                     sr_dict=self._sr_dict_list[pid],
#                     so_dict=self._so_dict_list[pid],
#                 )

#     def __len__(self) -> int:
#         n = len(self.temporal_graphs) - 1


class TkgrDataset(Dataset[TkgrFeature]):
    def __init__(
        self,
        temporal_quads: List[numpy.ndarray],
        vocabs: Dict[str, Vocab],
        hist_len: int,
    ) -> None:
        super().__init__()
        self.vocabs = vocabs
        self.hist_len = hist_len

        self.temporal_graphs = [
            build_knowledge_graph(quads, len(self.vocabs["ent"]))
            for quads in temporal_quads
        ]
        self._sr_dict_list, self._so_dict_list = build_sr_so_dict_list(
            temporal_quads
        )

    def __iter__(self) -> Iterator[TkgrFeature]:
        for pid in range(1, len(self.temporal_graphs)):
            xlice = slice(max(pid - self.hist_len, 0), pid)
            hist_graphs = self.temporal_graphs[xlice]
            graph = self.temporal_graphs[pid]
            yield TkgrFeature(
                hist_graphs=hist_graphs,
                graph=graph,
                sr_dict=self._sr_dict_list[pid],
                so_dict=self._so_dict_list[pid],
            )

    def __len__(self) -> int:
        return len(self.temporal_graphs) - 1


class TkgrEvalDataset(Dataset[TkgrFeature]):
    def __init__(
        self,
        temporal_quads: List[List[Quadruple]],
        vocabs: int,
        hist_len: int,
        init_quads: List[List[Quadruple]],
    ) -> None:
        super().__init__()
        # self._temporal_quads = temporal_quads
        self.vocabs = vocabs
        self.hist_len = hist_len

        self.init_graphs = [
            build_knowledge_graph(quads, len(self.vocabs["ent"]))
            for quads in init_quads[-self.hist_len :]
        ]
        self.temporal_graphs = [
            build_knowledge_graph(quads, len(self.vocabs["ent"]))
            for quads in temporal_quads
        ]
        self._sr_dict_list, self._so_dict_list = build_sr_so_dict_list(
            temporal_quads
        )

    def __iter__(self) -> Iterator[TkgrFeature]:
        hist_graphs = self.init_graphs
        for i, graph in enumerate(self.temporal_graphs):
            yield TkgrFeature(
                hist_graphs=hist_graphs,
                graph=graph,
                sr_dict=self._sr_dict_list[i],
                so_dict=self._so_dict_list[i],
            )
            hist_graphs.pop(0)
            hist_graphs.append(graph)

    def __len__(self) -> int:
        return len(self.temporal_graphs)


def groupby_temporal(quadruplets: numpy.ndarray) -> List[numpy.ndarray]:
    temporal_sorted = quadruplets[numpy.argsort(quadruplets[:, 3])]
    _, uniq_indices = numpy.unique(temporal_sorted[:, 3], return_index=True)
    return numpy.split(temporal_sorted, uniq_indices[1:])


def build_knowledge_graph(
    quadruplets: numpy.ndarray,
    num_nodes: int,
) -> dgl.DGLGraph:
    src_ids = quadruplets[:, 0]
    rel_ids = quadruplets[:, 1]
    dst_ids = quadruplets[:, 2]

    graph = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes)
    graph.ndata["eid"] = graph.nodes()
    graph.edata["rid"] = torch.tensor(rel_ids)

    return graph


def load_quadruples(
    quadruple_file: str, bidirectional: bool
) -> List[Quadruple]:
    quadruples: List[Quadruple] = []
    with open(quadruple_file) as fp:
        for line in fp:
            line_splits = line.strip().split("\t")
            subj, rel, obj, mmt, *_ = line_splits
            quadruples.append(Quadruple(subj=subj, rel=rel, obj=obj, mmt=mmt))

    if bidirectional:
        reversed_quads = []
        for quad in quadruples:
            rev_quad = Quadruple(
                subj=quad["obj"],
                rel="REVERSE # " + quad["rel"],
                obj=quad["subj"],
                mmt=quad["mmt"],
            )
            reversed_quads.append(rev_quad)
        quadruples += reversed_quads

    return quadruples


def build_vocab_from_quadruples(
    quadruples: List[Quadruple],
) -> Dict[str, vocabs.Vocab]:
    ents = Counter()
    rels = Counter()
    for quad in quadruples:
        ents.update([quad["subj"], quad["obj"]])
        rels.update([quad["rel"]])

    presv_toks = {"unk": "[UNK]"}
    tknzs = {
        "ent": vocabs.counter_vocab(ents, presv_toks),
        "rel": vocabs.counter_vocab(rels, presv_toks),
    }
    return tknzs


def load_txt_quadruplets(txt_file: str) -> numpy.ndarray:
    data = pandas.read_csv(txt_file, sep="\t", header=None)
    return data.values[:, :4]


def load_txt_vocab(txt_file: str, sep: str = "\t") -> Vocab:
    item2id = {}
    with open(txt_file) as fp:
        for line in fp:
            item, ind, *_ = line.strip().split(sep)
            ind = int(ind)
            item2id[item] = ind
    return Vocab(item2id)


def _quad_meta(quads: numpy.ndarray):
    nodes = numpy.concatenate((quads[:, 0], quads[:, 2]))
    num_nodes = len(numpy.unique(nodes))
    num_edges = len(numpy.unique(quads[:, 1]))
    return num_nodes, num_edges


def _collect_meta(tquads: List[numpy.ndarray]):
    num_ent_list = []
    num_rel_list = []
    for quads in tquads:
        tents = numpy.concatenate((quads[:, 0], quads[:, 2]))
        trels = quads[:, 1]
        num_ent_list.append(len(numpy.unique(tents)))
        num_rel_list.append(len(numpy.unique(trels)))
    length = len(tquads)
    avg_num_nodes = numpy.mean(num_ent_list)
    avg_num_edges = numpy.mean(num_rel_list)
    return avg_num_nodes, avg_num_edges, length


def load_tkg_dataset(
    data_folder: str,
    hist_len: int,
    bidirectional: bool,
    prepend_val_and_test: bool,
    shuffle_seed: Optional[int],
) -> Tuple[Dict[str, TkgrDataset], Dict[str, Vocab]]:
    # load quadruplets
    data_files = {
        "train": "train.txt",
        "valid": "valid.txt",
        "test": "test.txt",
    }
    orig_quadruplets: Dict[str, List[Quadruple]] = {}
    for subset, data_file in data_files.items():
        data_file = os.path.join(data_folder, data_file)
        orig_quadruplets[subset] = load_txt_quadruplets(data_file)

    logger.info(
        "# Edges: \tTrain: %d  Val: %d  Test: %d",
        len(orig_quadruplets["train"]),
        len(orig_quadruplets["valid"]),
        len(orig_quadruplets["test"]),
    )
    # load dicts
    ent2id_file = os.path.join(data_folder, "entity2id.txt")
    rel2id_file = os.path.join(data_folder, "relation2id.txt")
    vocabs = {
        "ent": load_txt_vocab(ent2id_file),
        "rel": load_txt_vocab(rel2id_file),
    }
    logger.info(
        "num_entities: %d\tnum_relations: %d",
        len(vocabs["ent"]),
        len(vocabs["rel"]),
    )

    if bidirectional:
        num_rels = len(vocabs["rel"])
        forward_rels = dict(vocabs["rel"].items())
        backward_rels = {
            "REV " + rel: ind + num_rels for rel, ind in forward_rels.items()
        }
        bid_rels = {**forward_rels, **backward_rels}
        assert len(bid_rels) == 2 * len(vocabs["rel"])
        vocabs["rel"] = Vocab(bid_rels)

        for key, forward_quads in orig_quadruplets.items():
            backward_quads = forward_quads[:, [2, 1, 0, 3]]
            backward_quads[:, 1] = backward_quads[:, 1] + num_rels
            orig_quadruplets[key] = numpy.concatenate(
                (forward_quads, backward_quads), axis=0
            )

    temporal_quads = {}
    for subset in data_files:
        temporal_quads[subset] = groupby_temporal(orig_quadruplets[subset])
        avg_num_nodes, avg_num_rels, length = _collect_meta(
            temporal_quads[subset]
        )
        logger.info(
            "%s:  \tlength: %d\t# avg nodes: %.6f\t# avg rels\t%.6f",
            subset.capitalize(),
            length,
            avg_num_nodes,
            avg_num_rels,
        )

    datasets = {}
    datasets["train"] = TkgrDataset(temporal_quads["train"], vocabs, hist_len)
    if prepend_val_and_test:
        datasets["valid"] = TkgrEvalDataset(
            temporal_quads["valid"], vocabs, hist_len, temporal_quads["train"]
        )
        datasets["test"] = TkgrEvalDataset(
            temporal_quads["test"],
            vocabs,
            hist_len,
            temporal_quads["train"] + temporal_quads["valid"],
        )
    else:
        datasets["valid"] = TkgrDataset(
            temporal_quads["valid"], vocabs, hist_len
        )
        datasets["test"] = TkgrDataset(temporal_quads["test"], vocabs, hist_len)

    if shuffle_seed:
        datasets["train"] = datasets["train"].shuffle(seed=shuffle_seed)

    return datasets, vocabs
