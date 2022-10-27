import logging
from typing import Counter, Dict, List, Mapping, Optional, Union

import torch

from tallow.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def _set_property(obj, name: str, value):
    shadow_name = f"_{name}"
    setattr(obj, shadow_name, value)
    prop = property(lambda self: getattr(self, shadow_name))
    setattr(obj.__class__, name, prop)


class LookupTokenizer(Tokenizer):
    def __init__(
        self,
        token_to_index: Mapping[str, int],
        unk_tok: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.token_to_index = token_to_index
        self.index_to_token = {ind: tok for tok, ind in token_to_index.items()}

        self.unk_tok = unk_tok
        if self.unk_tok is not None:
            assert self.unk_tok in self.token_to_index
            self.unk_id = self.token_to_index[self.unk_tok]

    def get_vocab(self) -> Dict[str, int]:
        return self.token_to_index

    def tokenize(self, words: List[str]) -> List[str]:
        return words

    def to_id(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            token = tokens

            if token in self.token_to_index:
                return self.token_to_index[token]
            else:
                return self.unk_id  # if not unk token provided

        return [self.to_id(token) for token in tokens]

    def to_token(self, indexes: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(indexes, int):
            index = indexes
            return self.index_to_token[index]

        else:
            return [self.to_token(index) for index in indexes]

    def generate_embedding(self, embedding_size: int) -> torch.nn.Embedding:
        return torch.nn.Embedding(
            self.vocab_size, embedding_size, padding_idx=None
        )


class CounterLookupTokenizer(LookupTokenizer):
    def __init__(
        self,
        counter: Counter,
        max_size: int = None,
        min_freq: int = 0,
        unk_tok: Optional[str] = None,
        addtional_tokens: Optional[List[str]] = None,
    ) -> None:

        counter_tokens = []
        token: str
        for token, cnt in counter.most_common(max_size):
            if cnt < min_freq:
                break
            counter_tokens.append(token)

        if addtional_tokens is None:
            addtional_tokens = []

        total_tokens = addtional_tokens + counter_tokens
        token_to_index = {tok: ind for ind, tok in enumerate(total_tokens)}
        super().__init__(token_to_index=token_to_index, unk_tok=unk_tok)
