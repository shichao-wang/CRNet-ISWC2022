import contextlib
import errno
import logging
import os
import warnings
from numbers import Number
from typing import TYPE_CHECKING, Callable, Optional

import sortedcollections
import torch

if TYPE_CHECKING:
    from ..trainer import TrainerStateDict

logger = logging.getLogger(__name__)


class DiskManager:
    def __init__(
        self,
        save_folder_path: str,
        max_saves: int,
        sort_key: Callable[[str], Number],  # larger better
    ) -> None:
        assert os.path.exists(save_folder_path), "Create Folder First"
        self._save_folder_path = save_folder_path
        self._num_saves = max_saves

        self.saved_models = sortedcollections.SortedList(key=sort_key)
        logger.debug(
            "Parsing previous saved objects. %s", self._save_folder_path
        )
        for file in os.listdir(self._save_folder_path):
            model_path = os.path.join(self._save_folder_path, file)
            if sort_key(model_path) is None:
                logger.debug("Skip %s", file)
                continue
            self.saved_models.add(model_path)

        if len(self.saved_models) > self._num_saves:
            warnings.warn("Found but")

    @property
    def best_model_path(self) -> Optional[str]:
        if self.saved_models:
            return self.saved_models[-1]
        return None

    def save_or_substitute(
        self, state_dict: "TrainerStateDict", save_file: str
    ):
        save_path = os.path.join(self._save_folder_path, save_file)
        try:
            torch.save(state_dict, save_path)
            logger.debug("Save state dict to %s", save_path)

        except OSError as e:
            if e.errno == errno.ENOSPC:
                raise
            raise
        else:
            self.saved_models.add(save_path)

        if len(self.saved_models) > self._num_saves:
            model_to_remove = self.saved_models.pop(0)
            with contextlib.suppress(FileNotFoundError):
                os.remove(model_to_remove)
            logger.debug("Remove previous state dict: %s", model_to_remove)

        return save_path

    def load(self, model_path: str = None) -> "TrainerStateDict":
        if model_path is None:
            model_path = self.saved_models[-1]
        if not os.path.exists(model_path):
            raise FileNotFoundError("")  # Extreme error

        return torch.load(model_path)
