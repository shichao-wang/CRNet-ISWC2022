import argparse
import logging
import os
from typing import List, Set, Tuple

import molurus
import torch
from molurus import hierdict
from tallow.nn import forwards

from tkgl.data import unpack_graph
from tkgl.datasets import load_tkg_dataset
from tkgl.models import tkgr_model

logger = logging.getLogger(__name__)


def load_cfg(checkpoint_path: str, config_path: str = None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(checkpoint_path), "config.yml"
        )

    cfg = hierdict.load(open(config_path))
    return cfg


def load_model(model_cfg, checkpoint_path: str, **kwargs):
    model: torch.nn.Module = molurus.smart_instantiate(model_cfg, **kwargs)
    model.load_state_dict(torch.load(checkpoint_path)["model"])
    return model


def chains_to_group(tuples: List[Tuple[int, int]]) -> List[Set[int]]:
    increasing_sets: List[Set[int]] = []
    while True:
        for tup in tuples:
            for inc_set in increasing_sets:
                if tup[0] in inc_set:
                    inc_set.add(tup[0])
                if tup[1] in inc_set:
                    inc_set.add(tup[1])

        total_sets: Set[List] = set()
        for inc_set in increasing_sets:
            total_sets.add(sorted(inc_set))
        new_increasing_sets = [set(s) for s in total_sets]
        if sum(map(len, new_increasing_sets)) == sum(map, len(increasing_sets)):
            return new_increasing_sets
        increasing_sets = new_increasing_sets


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def red_str(obj: str):
    return bcolors.FAIL + f"{obj}" + bcolors.ENDC


def green_str(obj: str):
    return bcolors.OKGREEN + f"{obj}" + bcolors.ENDC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("regcn")
    parser.add_argument("crnet")
    args = parser.parse_args()

    regcn_cfg = load_cfg(args.regcn)
    crnet_cfg = load_cfg(args.crnet)
    assert regcn_cfg["data"] == crnet_cfg["data"]
    data_cfg = crnet_cfg["data"]

    datasets, vocabs = molurus.smart_call(
        load_tkg_dataset, data_cfg, shuffle_seed=None
    )

    regcn = load_model(
        regcn_cfg["model"],
        args.regcn,
        num_ents=len(vocabs["ent"]),
        num_rels=len(vocabs["rel"]),
    ).eval()
    crnet = load_model(
        crnet_cfg["model"],
        args.crnet,
        num_ents=len(vocabs["ent"]),
        num_rels=len(vocabs["rel"]),
    ).eval()

    def triplet_string(s: torch.Tensor, r: torch.Tensor, o: torch.Tensor):
        s = vocabs["ent"].to_token(s.item())
        r = vocabs["rel"].to_token(r.item())
        o = vocabs["ent"].to_token(o.item())
        return "({},\t{},\t{})".format(s, r, o)

    def quadruple_string(
        s: torch.Tensor, r: torch.Tensor, o1: torch.Tensor, o2: torch.Tensor
    ):
        s = vocabs["ent"].to_token(s.item())
        r = vocabs["rel"].to_token(r.item())
        o1 = vocabs["ent"].to_token(o1.item())
        o2 = vocabs["ent"].to_token(o2.item())
        sr = f"({s},\t{r})"
        o = f"({red_str(o1)}\t{green_str(o2)})"
        return sr + "\t" + o

    def print_ent_trips(ent):
        ent_mask = gtrip[:, 0] == ent
        ent_trips = gtrip[ent_mask]
        for s, r, o in ent_trips:
            print(triplet_string(s, r, o))

    for model_inputs in datasets["test"]:
        with torch.set_grad_enabled(False):
            regcn_outputs = forwards.module_forward(regcn, model_inputs)
            # regcn_outputs1 = forwards.module_forward(regcn, model_inputs)
            # regcn_outputs2 = forwards.module_forward(regcn, model_inputs)
            # assert torch.allclose(
            #     regcn_outputs1["obj_logit"], regcn_outputs2["obj_logit"]
            # )
            crnet_outputs = forwards.module_forward(crnet, model_inputs)
        regcn_opred = torch.argmax(regcn_outputs["obj_logit"], dim=-1)
        _, regcn_indices = torch.sort(regcn_outputs["obj_logit"])
        crnet_opred = torch.argmax(crnet_outputs["obj_logit"], dim=-1)

        # diff_mask = regcn_opred != crnet_opred
        # num_diffs = torch.count_nonzero(diff_mask).item()
        # logger.info("Find %d differences", num_diffs)
        gsub, grel, gobj = unpack_graph(model_inputs["graph"])

        regcn_false_mask = regcn_opred != gobj
        regcn__true_mask = regcn_opred == gobj
        crnet__true_mask = crnet_opred == gobj
        true_mask = torch.bitwise_and(regcn__true_mask, crnet__true_mask)
        target_mask = torch.bitwise_and(regcn_false_mask, crnet__true_mask)
        num_targets = torch.count_nonzero(target_mask).item()
        logger.info("Find %d target_samples", num_targets)
        true_indexes = torch.nonzero(true_mask)[:, -1]
        for ind in true_indexes:
            if grel[ind] >= len(vocabs["rel"]) // 2:
                continue
            print(
                "ctx: " + triplet_string(gsub[ind], grel[ind], regcn_opred[ind])
            )

        target_indexes = torch.nonzero(target_mask)[:, -1]
        for ind in target_indexes:
            if grel[ind] >= len(vocabs["rel"]) // 2:
                continue
            print(
                "tgt: "
                + quadruple_string(
                    gsub[ind], grel[ind], regcn_opred[ind], crnet_opred[ind]
                )
                + "\t"
                + str(regcn_indices[ind][crnet_opred[ind]].item())
            )

        # rel_q_mask = torch.zeros(
        #     len(vocabs["rel"]), grel.size(0), dtype=torch.bool
        # )
        # rel_q_mask[grel, torch.arange(grel.size(0))] = True
        # for srel in rel_q_mask:
        #     rel_target_mask = torch.bitwise_and(target_mask, srel)
        #     indexes = torch.nonzero(rel_target_mask)[:, -1]

        input("Enter to continue")

        # for target in target_indexes:
        #     if grel[target] >= len(vocabs["rel"]) // 2:
        #         continue
        #     gtrip = torch.stack([gsub, grel, gobj], dim=1)

        #     sub = gsub[target]
        #     rel = grel[target]
        #     print("REGCN & RERANK ")
        #     print(triplet_string(sub, rel, regcn_opred[target]))
        #     print(triplet_string(sub, rel, crnet_opred[target]))

        #     print("Subj Context: ")
        #     print_ent_trips(sub)

        #     print("False Context: ")
        #     print_ent_trips(regcn_opred[target])

        #     print("Truth Context: ")
        #     print_ent_trips(crnet_opred[target])

        #     print()
        #     print("=" * 20)
        #     input("Enter to continue")


if __name__ == "__main__":
    main()
