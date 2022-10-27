from typing import Dict, Iterable, TypeVar

import molurus
import torch

TorchDevice = TypeVar("TorchDevice", int, str, torch.device)


def smart_optimizer(opt_cfg: Dict, params: Iterable[torch.nn.Parameter], **kw):
    return molurus.smart_instaniate(opt_cfg, params=params, **kw)
