import logging

import torch

from .backend import Backend
from .pytorch_native import (
    DistributedCudaBackend,
    SingletonCpuBackend,
    SingletonCudaBackend,
)

__all__ = [
    "Backend",
    "SingletonCudaBackend",
    "SingletonCpuBackend",
    "auto_select",
]
logger = logging.getLogger(__name__)


def auto_select(plugin: str = None, amp: bool = False):
    if plugin is None:
        # fallback to native
        num_devices = torch.cuda.device_count()
        logger.info("Find %d CUDA devices", num_devices)
        if num_devices == 0:
            return SingletonCpuBackend()
        if num_devices == 1:
            device = torch.device("cuda:0")
            return SingletonCudaBackend(device, amp)
        else:
            raise NotImplementedError()
            devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]
            return DistributedCudaBackend(devices, amp)
