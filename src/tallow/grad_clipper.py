import abc
from numbers import Number
from typing import Iterable

import torch


class GradClipper:
    @abc.abstractmethod
    def clip(self, parameters: Iterable[torch.nn.Parameter]):
        raise NotImplementedError()


class GradNormClipper(GradClipper):
    def __init__(self, max_norm: float, norm_type: float = 2.0) -> None:
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip(self, parameters: Iterable[torch.nn.Parameter]):
        return torch.nn.utils.clip_grad.clip_grad_norm_(
            parameters, self.max_norm, self.norm_type
        )


class GradValueClipper(GradClipper):
    def __init__(self, clip_value: float) -> None:
        self._clip_value = clip_value

    def clip(self, parameters: Iterable[torch.nn.Parameter]):
        return torch.nn.utils.clip_grad.clip_grad_value_(
            parameters, self._clip_value
        )


def smart_grad_clipper(
    *, grad_norm: float = None, grad_value: float = None, norm_type: float = 2.0
) -> GradClipper:
    assert not all(x is None for x in [grad_norm, grad_value])
    if all(isinstance(x, Number) for x in [grad_norm, grad_value]):
        raise ValueError("Either 'grad_norm' or 'grad_value' should be empty")
    if grad_norm is not None:
        return GradNormClipper(grad_norm, norm_type)
    if grad_value is not None:
        return GradValueClipper(grad_value)
