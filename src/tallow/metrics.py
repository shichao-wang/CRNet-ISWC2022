from abc import abstractmethod
from typing import Dict, Protocol

import torch


class TorchMetric(Protocol):
    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Override this method to update the state variables of your metric class.
        """

    @abstractmethod
    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Override this method to compute the final metric value from
         state variables synchronized across the distributed backend.
        """

    def reset(self) -> None:
        pass

    def to(self, device) -> "TorchMetric":
        pass
