from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import TrainContext


class StopTraining(Exception):
    def __init__(self, ctx: "TrainContext", reason: str = None) -> None:
        self.ctx = ctx
        self.reason = reason

    def __str__(self) -> str:
        return f"Stop training by {self.reason}"
