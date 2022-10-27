import argparse
import logging
import os
import re
import string
from typing import Dict, List

from tkgl.data import Quadruple

logger = logging.getLogger(__name__)


def get_out_folder_path(data_path: str):
    return data_path + "-comp"


def load_index_to_item(path: str):

    with open(path) as fp:
        lines = fp.read().splitlines()

    index2item = {}
    for line in lines:
        line_splits = line.split("\t")
        item, index, *_ = line_splits
        index2item[index] = item

    return index2item


QUAD_TEMPLATE = string.Template("${subj}\t${rel}\t${obj}\t${mmt}")


def dump_quadruples(file: str, quadruples: List):
    with open(file, "w") as fp:
        for quad in quadruples:
            quad_string = QUAD_TEMPLATE.safe_substitute(quad)
            fp.write(quad_string + "\n")


QUAD_RE = re.compile(
    r"^(?P<subj>\d+)\s+(?P<rel>\d+)\s+(?P<obj>\d+)\s+(?P<mmt>\d+)"
)


def parse_quadruple_line(
    line: str, id2ent: Dict[str, str] = None, id2rel: Dict[str, str] = None
) -> Quadruple:
    match = QUAD_RE.match(line)
    assert match is not None
    subj, rel, obj, mmt = match.group("subj", "rel", "obj", "mmt")
    if id2ent:
        subj = id2ent[subj]
        obj = id2ent[obj]
    if id2rel:
        rel = id2rel[rel]
    return Quadruple(subj=subj, rel=rel, obj=obj, mmt=mmt)


def load_quadruples(file: str, id2ent: Dict[str, str], id2rel: Dict[str, str]):
    quads = []
    with open(file) as fp:
        for line in fp:
            quad = parse_quadruple_line(line, id2ent, id2rel)
            quads.append(quad)
    return quads


FILE_MAP = (
    ("train.txt", "train.tsv"),
    ("valid.txt", "val.tsv"),
    ("test.txt", "test.tsv"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath")
    args = parser.parse_args()

    data_path = args.datapath
    assert os.path.exists(data_path)
    out_path = get_out_folder_path(data_path)
    os.mkdir(out_path)

    id2rel = load_index_to_item(os.path.join(data_path, "relation2id.txt"))
    id2ent = load_index_to_item(os.path.join(data_path, "entity2id.txt"))

    logger.info("Load # Entity: %d\t# Relation: %d", len(id2ent), len(id2rel))

    for ori_file, new_file in FILE_MAP:
        quads = load_quadruples(
            os.path.join(data_path, ori_file), id2ent, id2rel
        )
        logger.info("Load %d quadruples from %s ", len(quads), ori_file)
        dump_quadruples(os.path.join(out_path, new_file), quads)


if __name__ == "__main__":
    main()
