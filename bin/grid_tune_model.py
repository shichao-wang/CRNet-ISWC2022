import dataclasses
import hashlib
import itertools
import os
from typing import Callable, Dict, Iterable, List, Mapping, Tuple, TypedDict

import molurus
import torch
from molurus import hierdict
from pandas import DataFrame
from tallow.common.supports import SupportsStateDict
from tallow.data.datasets import Dataset
from tallow.data.vocabs import Vocab
from tallow.evaluators import Evaluator
from tallow.trainers import TrainerStateDict
from train_model import train_model, validate_and_dump_config

from tkgl import models
from tkgl.datasets import load_tkg_dataset
from tkgl.metrics import JointMetric


def grid_product(dict_args: hierdict.HierDict, product_fields: List[str]):
    call_args = dict_args.copy()
    search_values = [dict_args[key] for key in product_fields]
    assert all(isinstance(v, Iterable) for v in search_values)
    for search_args in itertools.product(*search_values):
        search_kwargs = dict(zip(product_fields, search_args))
        call_args.replace(search_kwargs)
        yield call_args


TUNE_FIELDS = ["model.k", "training.beta"]


class TunerStateDict(TypedDict):
    pass


@dataclasses.dataclass()
class TunerContext(SupportsStateDict[TunerStateDict]):
    tune_cfg: hierdict.HierDict
    tune_fields: List[str]
    trials: Mapping[str, str]

    def load_state_dict(self, state_dict: TunerStateDict):
        return super().load_state_dict(state_dict)

    def state_dict(self) -> TunerStateDict:
        return super().state_dict()


class HyperParameterTuner:
    def __init__(
        self,
        tune_folder_path: str,
        evaluator: Evaluator,
        trial_fn: Callable[[hierdict.HierDict], TrainerStateDict],
    ) -> None:
        self.tune_folder_path = tune_folder_path
        self.evaluator = evaluator
        self.trial_fn = trial_fn

    def execute(
        self,
        tune_cfg: hierdict.HierDict,
        tune_fields: str,
    ) -> TunerStateDict:
        trials = []

        for cfg in grid_product(tune_cfg, tune_fields):
            self.execute_tune(cfg)

    def execute_tune(self, cfg: hierdict.HierDict):
        cfg_hash = f"{hash(cfg):X}"
        trial_folder_path = os.path.join(self.tune_folder_path, cfg_hash)
        os.makedirs(trial_folder_path, exist_ok=True)
        state_dict = self.trial_fn(trial_folder_path, cfg)

        trial_cfg_path = os.path.join(trial_folder_path, "config.yml")
        hierdict.dump(cfg, open(trial_cfg_path, "w"))
        torch.save(state_dict, os.path.join(trial_folder_path, "tune.pt"))
        pass


def save_trial(
    trial_folder_path: str, cfg: hierdict.HierDict, state_dict: TrainerStateDict
):
    trial_cfg_path = os.path.join(trial_folder_path, "config.yml")
    hierdict.dump(cfg, open(trial_cfg_path, "w"))
    torch.save(state_dict, os.path.join(trial_folder_path, "tune.pt"))


def main():
    parser = hierdict.ArgumentParser()
    parser.add_argument("--save-folder-path")
    args = parser.parse_args()
    tune_cfg = hierdict.parse_args(args.cfg, args.overrides)
    print(tune_cfg)

    tune_folder_path = args.save_folder_path
    os.makedirs(tune_folder_path, exist_ok=True)
    validate_and_dump_config(tune_folder_path, tune_cfg, "tune-config.yml")
    datasets, vocabs = molurus.smart_call(load_tkg_dataset, tune_cfg["data"])

    evaluator = Evaluator(datasets, JointMetric())
    tune_cfg_iter = grid_product(tune_cfg, TUNE_FIELDS)
    mrr_dict: Dict[float, DataFrame] = {}
    for i, cfg in enumerate(tune_cfg_iter):
        cfg_hash = f"{hash(cfg):X}"
        trial_folder_path = os.path.join(tune_folder_path, cfg_hash)
        model = models.build_model(
            cfg["model"],
            num_ents=len(vocabs["ent"]),
            num_rels=len(vocabs["rel"]),
        )
        best_state_dict = train_model(
            trial_folder_path,
            model,
            datasets["train"],
            datasets["val"],
            cfg["training"],
        )
        dataframe = evaluator.execute(model, best_state_dict["model"])
        mrr_dict[dataframe["val"]["e_mrr"]] = dataframe["test"]
        for f in TUNE_FIELDS:
            print(f"{f}={cfg[f]}\t", end="")
        print()
        print(dataframe.T * 100)

    max_mrr = max(mrr_dict)
    best_test_dataframe = mrr_dict[max_mrr]

    print(best_test_dataframe.T * 100)


if __name__ == "__main__":
    main()
