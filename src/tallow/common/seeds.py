import os


def seed_torch(seed: int = None, cudnn: bool = False):
    import torch

    torch.manual_seed(seed)
    if cudnn:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from torch.backends import cudnn

        cudnn.benchmark = False
        cudnn.deterministic = True

        torch.use_deterministic_algorithms(True)


def seed_numpy(seed: int = None):
    import numpy

    numpy.random.seed(seed)


def seed_python(seed: int = None):
    import random

    random.seed(seed)


def seed_all(seed: int, cudnn: bool = False):
    seed_python(seed)
    seed_numpy(seed)
    seed_torch(seed, cudnn)
