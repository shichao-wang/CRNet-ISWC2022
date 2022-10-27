from .classes import Embedding, SequenceEmbedding
from .lookup import (
    LookupEmbedding,
    Word2VecEmbedding,
    random_init_embedding_weight,
)
from .sequence import BiRNNEmbedding, RNNEmbedding

try:
    from .huggingface import HuggingfaceEmbedding
except ImportError:
    pass
    # HuggingfaceEmbedding = None

# isort: list
__all__ = [
    "BiRNNEmbedding",
    "Embedding",
    "HuggingfaceEmbedding",
    "LookupEmbedding",
    "RNNEmbedding",
    "SequenceEmbedding",
    "Word2VecEmbedding",
    "random_init_embedding_weight",
]
