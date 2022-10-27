import argparse
import os

import torch
from tallow.data.datasets import Dataset
from tallow.evaluators import Evaluator
from tallow.metrics import TorchMetric
from tallow.trainers import Trainer

from tkgl.datasets import load_tkg_dataset
from tkgl.metrics import EntMetric
from tkgl.models.cygnet import CyGNetConfig, CyGNetLoss, CyGNetModel


def train_model(
    save_folder_path: str,
    model: torch.nn.Module,
    train_data: Dataset,
    val_data: Dataset,
    metric: TorchMetric,
    args,
):
    criterion = CyGNetLoss(args.reg_term)
    optimizer = torch.optim.Adam(
        params=(p for p in model.parameters() if p.requires_grad),
    )
    trainer = Trainer(
        save_folder_path,
        earlystop_dataset="valid",
        earlystop_monitor="e_fmrr",
        earlystop_patient=5,
    )
    state_dict = trainer.execute(
        model, train_data, criterion, optimizer, val_data, metric
    )
    return state_dict


def train(save_folder_path: str, args) -> float:
    # logger.info(cfg)
    datasets, vocabs = load_tkg_dataset(
        "./datasets/" + args.dataset, args.hist_len, True, True, True
    )
    train_data = datasets.pop("train")
    model_cfg = CyGNetConfig(
        args.hidden_size, len(vocabs["ent"]), len(vocabs["rel"])
    )
    model = CyGNetModel(model_cfg)
    metric = EntMetric()
    criterion = CyGNetLoss(args.reg_term)
    optimizer = torch.optim.Adam(
        params=(p for p in model.parameters() if p.requires_grad),
    )
    trainer = Trainer(
        save_folder_path,
        earlystop_dataset="valid",
        earlystop_monitor="e_fmrr",
        earlystop_patient=5,
    )
    best_state_dict = trainer.execute(
        model, train_data, criterion, optimizer, datasets["valid"], metric
    )
    evaluator = Evaluator(datasets, metric)
    dataframe = evaluator.execute(model, best_state_dict["model"])
    print(dataframe * 100)
    return dataframe["e_fmrr"]["valid"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-folder-path")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--hist_len", type=int)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--reg_term", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.save_folder_path, exist_ok=True)
    train(args.save_folder_path, args)


if __name__ == "__main__":
    main()
