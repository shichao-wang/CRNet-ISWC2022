import math
import multiprocessing
import numbers
import queue
import random
import threading
from typing import Callable, Iterator, List, Mapping, Optional, TypeVar

import torch

from . import batch, dataset

T_co = TypeVar("T_co", contravariant=True)


class BatchDataset(dataset.Dataset[batch.Batch]):
    def __init__(
        self,
        dataset: dataset.Dataset,
        batch_size: int,
        collate_fn: Callable[[List], batch.Batch] = batch.default_collate,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._batch_size = batch_size
        self._collate_fn = collate_fn

    def __iter__(self):
        items = []
        for item in self._dataset:
            if len(items) == self._batch_size:
                batch = self._collate_fn(items)
                yield batch
                del items, batch
                items = []
            items.append(item)

        if items:
            batch = self._collate_fn(items)
            yield batch
            del items, batch

    def __len__(self) -> int:
        src_len = len(self._dataset)
        return math.ceil(src_len / self._batch_size)


def prefetch_worker(dataset: dataset.Dataset, queue: multiprocessing.Queue):
    for item in dataset:
        queue.put(item)
    queue.put(None)


class PrefetchDataset(dataset.Dataset[T_co]):
    """
    Faster batch collate, pin_memory
    """

    def __init__(self, dataset: dataset.Dataset, fetch_size: int) -> None:
        super().__init__()
        self._dataset = dataset
        self._fetch_size = fetch_size
        self._queue: queue.Queue = queue.Queue(self._fetch_size)

    def __iter__(self) -> Iterator[T_co]:
        worker = threading.Thread(
            target=prefetch_worker,
            args=(self._dataset, self._queue),
            daemon=True,
        )
        worker.start()
        while True:
            item = self._queue.get()
            if item is None:
                break

            yield item

        worker.join()

    def __len__(self) -> int:
        return len(self._dataset)


R_co = TypeVar("R_co", contravariant=True)


class TransformDataset(dataset.Dataset[R_co]):
    def __init__(
        self,
        dataset: dataset.Dataset[T_co],
        transform_fn: Callable[[T_co], R_co],
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._transform = transform_fn
        self._args = args
        self._kwargs = kwargs

    def __iter__(self) -> Iterator[R_co]:
        for item in self._dataset:
            output = self._transform(item, *self._args, **self._kwargs)
            yield output

    def __len__(self) -> int:
        return len(self._dataset)


class BufferShuffledDataset(dataset.Dataset[T_co]):
    def __init__(
        self, dataset: dataset.Dataset, buffer_size: int, seed: int = None
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._buffer_size = buffer_size
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[T_co]:
        buffer: List[T_co] = []
        for item in self._dataset:
            if len(buffer) == self._buffer_size:
                idx = self._rng.randrange(0, self._buffer_size)
                yield buffer[idx]
                buffer[idx] = item
            else:
                buffer.append(item)

        self._rng.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def __len__(self) -> int:
        return len(self._dataset)


def to_tensor(item: Mapping[str, Optional[torch.Tensor]], pin_memory: bool):
    import numpy

    for key, value in item.items():
        if isinstance(value, numbers.Number):
            item[key] = torch.as_tensor(value)
        elif isinstance(value, numpy.ndarray):
            item[key] = torch.from_numpy(value)
        else:
            continue

        if pin_memory:
            item[key] = item[key].pin_memory()

    return item


class ReSizedDataset(dataset.SizedDataset[T_co]):
    def __init__(self, dataset: dataset.Dataset[T_co], length: int) -> None:
        super().__init__(length)
        self._dataset = dataset

    def __iter__(self) -> Iterator[T_co]:
        return iter(self._dataset)


class RepeatDataset(dataset.Dataset[T_co]):
    def __init__(self, dataset: dataset.Dataset[T_co], n: int = None) -> None:
        super().__init__()
        self._dataset = dataset
        self._n = n or float("inf")

    def __iter__(self) -> Iterator[T_co]:
        i = 0
        while True:
            yield from self._dataset
            i += 1
            if i == self._n:
                break

    def __len__(self):
        return len(self._dataset) * self._n
