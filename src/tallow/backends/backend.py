import abc
from typing import TYPE_CHECKING, Callable, Dict, TypeVar

import torch
from torch import nn

from tallow.data.datasets import Dataset

if TYPE_CHECKING:
    from tallow.trainers.trainer import TrainContext

T_co = TypeVar("T_co", covariant=True)


class Backend:
    @abc.abstractmethod
    def setup_dataset(self, dataset: Dataset):
        raise NotImplementedError()

    @abc.abstractmethod
    def setup_module(self, module: nn.Module) -> nn.Module:
        raise NotImplementedError()

    @abc.abstractmethod
    def backward(self, loss: torch.Tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def step_and_zero_grad(self, ctx: "TrainContext"):
        raise NotImplementedError()


class DistributedBackend(Backend):
    @abc.abstractmethod
    def launch(self, launch_fn: Callable, kwargs: Dict):
        raise NotImplementedError()
