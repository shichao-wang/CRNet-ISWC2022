import contextlib
import dataclasses
import logging
import numbers
import os
import sys
from typing import Collection, Dict, List, Optional, Tuple, TypedDict, Union

import torch
from pandas import DataFrame
from tqdm import tqdm

from tallow import backends
from tallow.common.supports import SupportsStateDict
from tallow.data.datasets import Dataset
from tallow.evaluators import Evaluator
from tallow.grad_clipper import GradClipper, smart_grad_clipper
from tallow.metrics import TorchMetric
from tallow.nn import forwards

from .hooks.checkpoint import CheckpointHook
from .hooks.early_stop import EarlyStopHook
from .hooks.hook import Hook, HookManager, Hooks
from .signals import StopTraining

logger = logging.getLogger(__name__)


class TrainerStateDict(TypedDict):
    model: Dict[str, torch.Tensor]
    criterion: Dict[str, torch.Tensor]
    optimizer: Dict[str, torch.Tensor]
    num_epochs: int
    num_batches: int


@dataclasses.dataclass()
class TrainContext(SupportsStateDict[TrainerStateDict]):
    model: torch.nn.Module
    dataset: Dataset
    criterion: torch.nn.Module
    optim: torch.optim.Optimizer
    grad_clipper: Optional[GradClipper]

    val_dataframe: DataFrame = None
    num_epochs: int = 0
    num_batches: int = 0

    def state_dict(self) -> TrainerStateDict:
        return TrainerStateDict(
            model=self.model.state_dict(),
            criterion=self.criterion.state_dict(),
            optimizer=self.optim.state_dict(),
            num_epochs=self.num_epochs,
            num_batches=self.num_batches,
        )

    def load_state_dict(self, state_dict: TrainerStateDict):
        self.model.load_state_dict(state_dict["model"])
        self.criterion.load_state_dict(state_dict["criterion"])
        self.optim.load_state_dict(state_dict["optimizer"])
        self.num_epochs = state_dict["num_epochs"]
        self.num_batches = state_dict["num_batches"]


class Trainer:
    def __init__(
        self,
        save_folder_path: str,
        enable_checkpoint: bool = True,
        max_epochs: int = None,
        val_batches_interval: Optional[int] = None,
        earlystop_monitor: str = None,
        earlystop_patient: int = 5,
        earlystop_dataset: str = "val",
        grad_clip_norm: Union[float, float] = None,
        grad_clip_val: float = None,
        grad_accumulate_batches: int = 1,
        hooks: List[Hook] = None,
    ) -> None:
        os.makedirs(save_folder_path, exist_ok=True)

        self.backend = backends.auto_select()

        self._max_epochs = max_epochs
        self._val_batches_interval = val_batches_interval

        self.checkpoint = None
        if enable_checkpoint:
            self.checkpoint = CheckpointHook(save_folder_path)

        self.earlystop = None
        if earlystop_monitor is not None:
            self.earlystop = EarlyStopHook(
                save_folder_path,
                monitor=earlystop_monitor,
                patient=earlystop_patient,
                dataset=earlystop_dataset,
            )

        if isinstance(grad_clip_norm, numbers.Number):
            norm_type = 2.0
        elif isinstance(grad_clip_norm, Collection):
            grad_clip_norm, norm_type = grad_clip_norm
        self._grad_clipper = None
        if not all(x is None for x in [grad_clip_norm, grad_clip_val]):
            self._grad_clipper = smart_grad_clipper(
                grad_norm=grad_clip_norm,
                grad_value=grad_clip_val,
                norm_type=norm_type,
            )

        self._grad_accumulate_batches = grad_accumulate_batches

        internal_hooks = [self.checkpoint, self.earlystop]
        internal_hooks = [h for h in internal_hooks if h is not None]
        if hooks is None:
            hooks = []
        self.hook_mgr = HookManager(*internal_hooks, *hooks)

    def execute(
        self,
        model: torch.nn.Module,
        train_data: Dataset,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        val_data: Union[Dataset, Dict[str, Dataset]],
        val_metric: TorchMetric,
        try_eval_first: bool = False,
    ) -> TrainerStateDict:
        if isinstance(val_data, Dataset):
            val_data = {"valid": val_data}
        if self.earlystop:
            assert self.earlystop._dataset_field in val_data
        self.evaluator = Evaluator(val_data, val_metric, self.backend)
        if try_eval_first:
            self.evaluator.execute(model)

        tr_ctx = TrainContext(
            model, train_data, criterion, optimizer, self._grad_clipper
        )
        # following should be `self.backend.launch`
        return self._execute_impl(tr_ctx)

    def load_checkpoint_state_dict(self):
        if self.checkpoint is not None:
            return self.checkpoint._disk_mgr.load()

    def _execute_impl(self, ctx: TrainContext) -> Tuple[TrainerStateDict, Dict]:
        self.hook_mgr.call_hook(Hooks.ON_TRAIN_BEGIN, ctx)
        try:
            self._try_train(ctx)
        except (StopTraining):
            logger.info("StopTraining")
            self.hook_mgr.call_hook(Hooks.ON_TRAIN_END, ctx)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt")

        if self.earlystop is not None:
            return self.earlystop._disk_mgr.load()
        return ctx.state_dict()

    def _try_train(self, ctx: TrainContext):
        ctx.dataset = self.backend.setup_dataset(ctx.dataset)
        ctx.model = self.backend.setup_module(ctx.model)

        while True:
            if (
                self._max_epochs is not None
                and ctx.num_epochs >= self._max_epochs
            ):
                raise StopTraining(ctx)

            ctx.num_epochs += 1
            self.hook_mgr.call_hook(Hooks.ON_EPOCH_BEGIN, ctx)
            with training(ctx.model):
                tqdm_prog = tqdm(
                    ctx.dataset,
                    desc=f"Epoch: {ctx.num_epochs}",
                    leave=False,
                    file=sys.stderr,
                    dynamic_ncols=True,
                )
                for model_inputs in tqdm_prog:
                    ctx.num_batches += 1
                    self.hook_mgr.call_hook(Hooks.ON_BATCH_BEGIN, ctx)
                    model_outputs = forwards.module_forward(
                        ctx.model, model_inputs
                    )
                    loss: torch.Tensor = forwards.module_forward(
                        ctx.criterion, model_inputs, model_outputs
                    )
                    self.backend.backward(loss)
                    if ctx.num_batches % self._grad_accumulate_batches == 0:
                        self.backend.step_and_zero_grad(ctx)
                    self.hook_mgr.call_hook(Hooks.ON_BATCH_END, ctx)

                    if (
                        self._val_batches_interval is not None
                        and ctx.num_batches % self._val_batches_interval == 0
                    ):
                        self._do_validation(ctx)

                if self._val_batches_interval is None:
                    self._do_validation(ctx)

            self.hook_mgr.call_hook(Hooks.ON_EPOCH_END, ctx)

    def _do_validation(self, ctx: TrainContext):
        self.hook_mgr.call_hook(Hooks.ON_VALID_BEGIN, ctx)
        ctx.val_dataframe = self.evaluator.execute(ctx.model)
        logger.info("\n" + ctx.val_dataframe.to_string())
        self.hook_mgr.call_hook(Hooks.ON_VALID_END, ctx)


@contextlib.contextmanager
def training(model: torch.nn.Module):
    prev = model.training
    model.train(True)
    with torch.set_grad_enabled(True):
        yield
    model.train(prev)


@contextlib.contextmanager
def evaluating(model: torch.nn.Module):
    prev = model.training
    model.train(False)
    with torch.set_grad_enabled(False):
        yield
    model.train(prev)
