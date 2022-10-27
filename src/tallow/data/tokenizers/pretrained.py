# import os
import logging
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional

import numpy
import torch

from tallow.data.tokenizers.lookup import LookupTokenizer
from tallow.data.tokenizers.tokenizer import Tokenizer

if TYPE_CHECKING:

    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_word2vec_weight(
    glove_file: str, vocab: Mapping[str, int]
) -> torch.Tensor:
    embedding_dict = {}
    with open(glove_file, "rt") as fp:
        for line in fp:
            word, *vector_string = str.split(line, sep=" ")
            vector = numpy.asfarray(vector_string)
            embedding_dict[word] = vector
    embed_size = numpy.size(embedding_dict[word])

    vocab_size = len(vocab)
    weights = torch.empty(vocab_size, embed_size)
    weights.normal_()

    found_cnt = 0
    for word, index in vocab.items():
        if word in embedding_dict:
            found_cnt += 1
            weights[index] = torch.from_numpy(embedding_dict[word])

    msg = "Load %d from %d pre-trained Glove embedding (%f)"
    logger.info(msg, found_cnt, vocab_size, found_cnt / vocab_size)
    return weights


class Word2VecTokenizer(LookupTokenizer):
    def __init__(
        self,
        word2vec_file: str,
        unk_tok: Optional[str] = None,
    ) -> None:
        self.word2vec_file = word2vec_file
        token_to_index = {}
        # this should extremely fast
        with open(word2vec_file) as fp:
            for lineno, line in enumerate(fp):
                word, *_ = str.split(line, sep=" ")
                token_to_index[word] = lineno

        msg = "Load %d tokens from %s."
        logger.info(msg, len(token_to_index), word2vec_file)

        super().__init__(token_to_index, unk_tok)

    def generate_embedding(self, embedding_size: int) -> torch.nn.Embedding:
        weights = load_word2vec_weight(self.word2vec_file, self.get_vocab())
        embedding = torch.nn.Embedding.from_pretrained(weights)
        if embedding.embedding_dim == embedding_size:
            return embedding
        else:
            return torch.nn.Sequential(
                embedding,
                torch.nn.Linear(
                    embedding.embedding_dim, embedding_size, bias=False
                ),
            )


class HuggingfaceTokenizer(Tokenizer):
    def __init__(
        self, pretrained_path: str, *, init_kwargs: Dict = None
    ) -> None:
        super().__init__()
        import transformers

        self.pretrained_path = pretrained_path
        init_kwargs = init_kwargs or {}
        self.hf_tokenizer: "PreTrainedTokenizer" = (
            transformers.AutoTokenizer.from_pretrained(
                pretrained_path, **init_kwargs
            )
        )

    @property
    def vocab_size(self) -> int:
        return self.hf_tokenizer.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        return self.hf_tokenizer.get_vocab()

    def tokenize(
        self,
        words: List[str],
        **hf_kwargs,
    ) -> List[str]:
        hf_kwargs.setdefault("is_split_into_words", True)
        tokens = self.hf_tokenizer.tokenize(words, **hf_kwargs)
        return tokens

    def __call__(self, words: List[str], **hf_kwargs) -> numpy.ndarray:
        hf_kwargs.setdefault("is_split_into_words", True)
        hf_kwargs.setdefault("return_tensors", "np")
        tokenize_outputs = self.hf_tokenizer(words, **hf_kwargs)
        return tokenize_outputs["input_ids"][0]

    def to_token(self, index: int) -> str:
        return self.hf_tokenizer.convert_ids_to_tokens(index)

    def to_id(self, token: str) -> int:
        return self.hf_tokenizer.convert_tokens_to_ids(token)

    def to_tokens(self, indexes: List[int]) -> List[str]:
        return self.hf_tokenizer.convert_ids_to_tokens(indexes)

    def to_ids(self, tokens: List[str]) -> List[int]:
        return self.hf_tokenizer.convert_tokens_to_ids(tokens)

    # special tokens
    @property
    def pad_tok(self):
        return self.hf_tokenizer.pad_token

    @property
    def unk_tok(self):
        return self.hf_tokenizer.unk_token

    @property
    def pad_id(self):
        return self.hf_tokenizer.pad_token_id

    @property
    def unk_id(self):
        return self.hf_tokenizer.unk_token_id

    def generate_embedding(
        self, embedding_size: int = None
    ) -> "PreTrainedModel":

        import transformers

        hf_model = transformers.AutoModel.from_pretrained(self.pretrained_path)
        return hf_model


def token_to_words(tokens: List[str]):
    current_word_id = -1
    ret = []
    for token in tokens:
        if not token.startswith("##"):
            current_word_id += 1
        ret.append(current_word_id)

    return ret
