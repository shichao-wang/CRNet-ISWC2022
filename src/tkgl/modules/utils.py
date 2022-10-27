import torch


def get_param(*shape):
    p = torch.nn.Parameter(torch.empty(shape))
    torch.nn.init.xavier_normal_(p)
    return p
