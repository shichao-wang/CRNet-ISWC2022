import argparse
import os

import molurus
import torch
from molurus import hierdict
from tallow import evaluators

from tkgl.datasets import load_tkg_dataset
from tkgl.metrics import EntMetric, JointMetric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    args = parser.parse_args()
    ckpt_path = args.checkpoint_path
    assert os.path.exists(ckpt_path)

    config_path = os.path.join(os.path.dirname(ckpt_path), "config.yml")
    cfg = hierdict.load(open(config_path))
    datasets, vocabs = load_tkg_dataset(**cfg["data"])

    model = molurus.smart_instantiate(
        cfg["model"], num_ents=len(vocabs["ent"]), num_rels=len(vocabs["rel"])
    )
    datasets.pop("train")
    evaluator = evaluators.Evaluator(datasets, EntMetric())
    results = evaluator.execute(model, torch.load(ckpt_path)["model"])
    print(results * 100)


if __name__ == "__main__":
    main()
