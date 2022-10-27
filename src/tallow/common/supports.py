import typing
from typing import Protocol, TypeVar

T = TypeVar("T")


@typing.runtime_checkable
class SupportsStateDict(Protocol[T]):
    def state_dict(self) -> T:
        pass

    def load_state_dict(self, state_dict: T):
        pass
