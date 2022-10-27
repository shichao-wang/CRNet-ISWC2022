import logging
import os
import re
from typing import TYPE_CHECKING, Optional

from .hook import Hook
from .model_save import DiskManager

if TYPE_CHECKING:
    from ..trainer import TrainContext


logger = logging.getLogger(__name__)

CHECKPOINT_RE = re.compile(r"(.*[/\\])?ckpt-e(\d+)_b(\d+)\.pt")
CHECKPOINT_TEMPLATE = "ckpt-e{epoch}_b{batch}.pt"


def parse_checkpoint(filename: str) -> Optional[int]:
    matches = CHECKPOINT_RE.match(filename)
    if matches is None:
        return None
    return os.stat(filename).st_ctime_ns


class CheckpointHook(Hook):
    def __init__(self, save_folder_path: str) -> None:
        super().__init__()
        self._disk_mgr = DiskManager(
            save_folder_path, 1, sort_key=parse_checkpoint
        )

    def on_train_begin(self, ctx: "TrainContext"):
        if len(self._disk_mgr.saved_models) != 0:
            ckpt_state_dict = self._disk_mgr.load()
            ctx.load_state_dict(ckpt_state_dict)
            logger.info(
                "Restore checkpoint from epoch=%d, batch=%d",
                ctx.num_epochs,
                ctx.num_batches,
            )

    def on_epoch_end(self, ctx: "TrainContext"):
        model_file = CHECKPOINT_TEMPLATE.format(
            epoch=ctx.num_epochs, batch=ctx.num_batches
        )
        ckpt_path = self._disk_mgr.save_or_substitute(
            ctx.state_dict(), model_file
        )
        logger.info("Save checkpoint to %s", ckpt_path)

    def on_train_end(self, ctx: "TrainContext"):
        ckpt_path = self._disk_mgr.saved_models.pop()
        os.remove(ckpt_path)
        return super().on_train_end(ctx)
