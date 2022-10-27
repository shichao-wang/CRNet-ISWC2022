import sys

import numpy
import torch
import tqdm
from torch import nn

from tallow.data import tokenizers

from . import classes


class LookupEmbedding(nn.Embedding, classes.Embedding):
    @property
    def embed_size(self):
        return self.embedding_dim

    @property
    def num_embeds(self):
        return self.num_embeddings

    @property
    def pad_id(self):
        return self.padding_idx

    def forward(self, input_ids: torch.Tensor):
        if self.pad_id is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            mask = input_ids.ne(self.pad_id)
        return self.Output(
            embedding=super().forward(input_ids),
            mask=mask,
        )


def random_init_embedding_weight(num_embedding: int, embed_size: int):
    weights = torch.zeros((num_embedding, embed_size))
    torch.nn.init.uniform_(
        weights, -numpy.sqrt(3 / embed_size), numpy.sqrt(3 / embed_size)
    )
    return weights


class Word2VecEmbedding(LookupEmbedding):
    def __init__(
        self,
        word2vec_file: str,
        tokenizer: tokenizers.Tokenizer,
        *,
        freeze: bool = False,
    ) -> None:
        weight = load_word2vec_weight(word2vec_file, tokenizer)
        weight.requires_grad = not freeze
        num_embeds, embed_size = weight.size()
        pad_id = getattr(tokenizer, "pad_id", None)
        super().__init__(
            num_embeds, embed_size, padding_idx=pad_id, _weight=weight
        )


def load_word2vec_weight(
    glove_file: str, tokenizer: tokenizers.Tokenizer
) -> torch.Tensor:
    embedding_dict = {}
    with open(glove_file, "rt") as fp:
        for line in tqdm.tqdm(fp, "Loading glove"):
            word, *vector_string = str.split(line, sep=" ")
            vector = numpy.asfarray(vector_string)
            embedding_dict[word] = vector
    embed_size = numpy.size(embedding_dict[word])

    weights = random_init_embedding_weight(
        num_embedding=tokenizer.vocab_size, embed_size=embed_size
    )
    unknown_weight = torch.mean(weights, dim=0)
    found_cnt = 0
    vocab = tokenizer.get_vocab()
    for word, index in vocab.items():
        if word in embedding_dict:
            found_cnt += 1
            weights[index] = torch.from_numpy(embedding_dict[word])
        else:
            weights[index] = unknown_weight

    msg = (
        f"Load {found_cnt} from {tokenizer.vocab_size} "
        f"pre-trained Glove embedding ({found_cnt / tokenizer.vocab_size})"
    )
    print(msg, file=sys.stderr)
    return weights
