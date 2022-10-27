import argparse
import logging
import operator
import os
from functools import reduce

import numpy

import tkgl

logging.getLogger(__name__)


def count_file(dfile: str):
    assert os.path.exists(dfile)

    total_quads = tkgl.datasets.load_txt_quadruplets(dfile)
    tquads = tkgl.datasets.groupby_temporal(total_quads)

    num_ent_list = []
    num_rel_list = []
    for quads in tquads:
        tents = numpy.concatenate((quads[:, 0], quads[:, 2]))
        trels = quads[:, 1]
        num_ent_list.append(len(numpy.unique(tents)))
        num_rel_list.append(len(numpy.unique(trels)))

    avg_ents = numpy.mean(num_ent_list)
    avg_rels = numpy.mean(num_rel_list) * 2
    print("# avg ent %.6f" % avg_ents)
    print("# avg rel %.6f" % avg_rels)


datafiles = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder")
    args = parser.parse_args()

    data_folder = args.data_folder
    assert os.path.exists(data_folder)
    for subset, data_file in datafiles.items():
        data_path = os.path.join(data_folder, data_file)
        print(subset)
        count_file(data_path)


if __name__ == "__main__":
    main()
