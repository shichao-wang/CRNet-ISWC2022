import logging
import os
import typing
from typing import Dict, Optional, Union

import molurus
import torch
from molurus import config_dict
from molurus.functions import smart_call
from pandas import DataFrame, Series
from tallow import backends
from tallow.backends.grad_clipper import grad_clipper
from tallow.data.datasets import Dataset
from tallow.evaluators import Evaluator
from tallow.trainers import Trainer, TrainerBuilder, TrainerStateDict, hooks

from .datasets import load_tkg_dataset
from .metrics import EntMRR, JointMetric
from .models import TkgrModel

logger = logging.getLogger(__name__)


class TkgrTrial:
    def __init__(self, save_folder_path: str) -> None:
        self.save_folder_path = save_folder_path

        os.makedirs(self.save_folder_path, exist_ok=True)

    def validate_and_dump_config(self, cfg: Dict, cfg_file: str = "config.yml"):
        cfg_path = os.path.join(self.save_folder_path, cfg_file)
        if not os.path.exists(cfg_path):
            with open(cfg_path, "w") as fp:
                config_dict.dump(cfg, fp)

        orig_cfg = config_dict.load(open(cfg_path))
        assert orig_cfg == cfg

    def load_datasets_and_vocab(self, dt_cfg: Dict):
        datasets, vocabs = smart_call(load_tkg_dataset, dt_cfg)
        num_ents = len(vocabs["ent"])
        num_rels = len(vocabs["rel"])
        num_edges = len(datasets["train"])
        logger.info(
            "# Entities: %d\t# Relations: %d\t# Edges: %d",
            num_ents,
            num_rels,
            num_edges,
        )
        return datasets, vocabs

    def train_model(
        self,
        model: TkgrModel,
        train_data: Dataset,
        criterion: torch.nn.Module,
        val_data: Dataset,
        tr_cfg: Dict,
    ):
        backend = molurus.smart_call(backends.auto_select, tr_cfg)
        val_metric = EntMRR()
        optimizer = molurus.smart_instaniate(
            "torch.optim.Adam", tr_cfg, params=model.parameters()
        )
        clipper = molurus.smart_call(grad_clipper, tr_cfg)
        trainer = (
            TrainerBuilder(self.save_folder_path, backend, grad_clipper=clipper)
            .earlystop(val_data, val_metric, tr_cfg["patient"])
            .build()
        )
        _ = trainer.execute(model, train_data, criterion, optimizer)
        return retrieve_best_model(trainer)

    @typing.overload
    def eval_model(
        self,
        model: TkgrModel,
        datasets: Dataset,
        state_dict: TrainerStateDict,
    ) -> Series:
        pass

    @typing.overload
    def eval_model(
        self,
        model: TkgrModel,
        datasets: Dict[str, Dataset],
        state_dict: TrainerStateDict,
    ) -> DataFrame:
        pass

    def eval_model(
        self,
        model: TkgrModel,
        datasets: Union[Dataset, Dict[str, Dataset]],
        state_dict: TrainerStateDict,
    ):
        metric = JointMetric()
        model.load_state_dict(state_dict["model"])
        if isinstance(datasets, Dataset):
            datasets = {"test": datasets}
        evaluator = Evaluator(datasets, metric)
        return evaluator.execute(model).squeeze()


def retrieve_best_model(trainer: Trainer) -> TrainerStateDict:
    for hook in trainer.hook_mgr:
        if isinstance(hook, hooks.EarlyStopHook):
            return hook._disk_mgr.load()
    raise ValueError()
