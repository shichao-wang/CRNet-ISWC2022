import abc
import typing
from typing import Dict, List, Union

import numpy
import torch


class SpecialTokensMixin:
    @property
    @abc.abstractmethod
    def pad_tok(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def pad_id(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def unk_tok(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def unk_id(self):
        raise NotImplementedError()


class Tokenizer(metaclass=abc.ABCMeta):
    # typing overloads
    @typing.overload
    def to_id(self, tokens: List[str]) -> List[int]:
        pass

    @typing.overload
    def to_id(self, token: str) -> int:
        pass

    @typing.overload
    def to_token(self, indexes: List[int]) -> List[str]:
        pass

    @typing.overload
    def to_token(self, index: int) -> str:
        pass

    # abstract methods
    @abc.abstractmethod
    def to_id(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_token(self, indexes: Union[int, List[int]]) -> Union[str, List[str]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def tokenize(self, words: List[str]) -> List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        pass

    @abc.abstractmethod
    def generate_embedding(self, embedding_size: int = None) -> torch.nn.Module:
        pass

    # partial implemented
    @property
    def vocab_size(self) -> int:
        return len(self.get_vocab())

    def __call__(self, words: List[str]) -> numpy.ndarray:
        tokens = self.tokenize(words)
        token_ids = self.to_id(tokens)
        return numpy.asarray(token_ids)
