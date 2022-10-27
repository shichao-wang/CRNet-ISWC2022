import torch

logit = torch.tensor([[0.8, 0.2, 0.0], [0.3, 0.5, 0.4], [0.2, 0.5, 0.3]])
target = torch.tensor([1, 2, 0])


def my_mrr():
    total_ranks = logit.argsort(dim=-1, descending=True)
    ranks = torch.nonzero(total_ranks == target.view(-1, 1))[:, 1]
    return (1 / (ranks + 1)).mean(dim=-1)


def his_mrr():
    _, indices = torch.sort(logit, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    ranks = indices[:, 1].view(-1)
    return (1 / (ranks + 1)).mean(dim=-1)


print(my_mrr(), his_mrr())
