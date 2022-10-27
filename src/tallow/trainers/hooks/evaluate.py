# import logging
# from typing import TYPE_CHECKING, Dict

# import torchmetrics as tm

# from tallow.data.datasets import Dataset
# from tallow.evaluators import trainer_evaluator

# from .hook import Hook

# if TYPE_CHECKING:
#     from tallow.trainers.trainer import Trainer


# logger = logging.getLogger(__name__)


# class EvaluateHook(Hook):
#     def __init__(
#         self, datasets: Dict[str, Dataset], metric: tm.Metric = None
#     ) -> None:
#         super().__init__()
#         self._datasets = datasets
#         self._metric = metric
#         self._evaluator = None

#     def on_train_begin(self, trainer: "Trainer"):
#         self._evaluator = trainer_evaluator(trainer, self._metric)

#     def on_valid_end(self, trainer: "Trainer"):
#         result_dataframe = self._evaluator.execute(self._datasets)
#         logger.info("\n" + result_dataframe.to_string())
