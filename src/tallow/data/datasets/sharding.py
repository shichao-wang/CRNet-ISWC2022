import atexit
import itertools
import random
import time
from typing import Iterable, Iterator, TypeVar

import torch
from torch import multiprocessing

from .dataset import Dataset

T_co = TypeVar("T_co", contravariant=True)


def worker_loop(
    worker_id: int,
    num_workers: int,
    chunk_size: int,
    job_queue: multiprocessing.Queue,
    data_queue: multiprocessing.Queue,
):
    torch.set_num_threads(1)
    buf = []
    try:
        while True:
            dataset = job_queue.get()
            for item in dataset:
                if len(buf) == chunk_size:
                    data_queue.put(buf)
                    buf = []
                buf.append(item)
            if buf:
                data_queue.put(buf)
            data_queue.put(None)
    except KeyboardInterrupt:
        pass


def monitor_worker(
    job_queue: multiprocessing.Queue, data_queue: multiprocessing.Queue
):
    while True:
        print(f"Job queue: {job_queue.qsize():d}")
        print(f"Data queue: {data_queue.qsize():d}")
        time.sleep(1)


class ShardingDataset(Dataset[T_co]):
    def __init__(
        self,
        datasets: Iterable[Dataset[T_co]],
        num_workers: int,
        chunk_size: int = 1,
        random_state: int = None,
        mp_context=None,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self._chunk_size = chunk_size
        self.rng = random.Random(random_state)
        if mp_context is None:
            mp_context = multiprocessing.get_context()
        self.mp_context = mp_context

        self.job_queue = self.mp_context.Queue()
        self.data_queue = self.mp_context.Queue(2 * num_workers)

        self.workers = [
            self.mp_context.Process(
                target=worker_loop,
                args=(
                    i,
                    num_workers,
                    self._chunk_size,
                    self.job_queue,
                    self.data_queue,
                ),
                daemon=True,
            )
            for i in range(num_workers)
        ]
        for w in self.workers:
            w.start()
        atexit.register(self.cleanup_workers)

    def __iter__(self) -> Iterator[T_co]:
        self.rng.shuffle(self.datasets)

        for ds in self.datasets:
            self.job_queue.put(ds)

        unfinished_jobs = len(self.datasets)
        while unfinished_jobs:
            data = self.data_queue.get()
            if data is None:
                unfinished_jobs -= 1
            else:
                yield from data
                del data

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    def cleanup_workers(self):
        for worker in self.workers:
            worker.terminate()


class ShardableChainDataset(Dataset[T_co]):
    def __init__(
        self,
        datasets: Iterable[Dataset[T_co]],
        random_state: int = None,
    ) -> None:
        super().__init__()
        self._datasets = datasets
        self.random_state = random_state
        self.rng = random.Random(random_state)

    def __iter__(self) -> Iterator[T_co]:
        from torch.utils import data

        if self.random_state:
            self.rng.shuffle(self._datasets)

        info = data.get_worker_info()
        if info is None:
            datasets = self._datasets
        else:
            worker_id, num_workers = info.id, info.num_workers
            datasets = self._datasets[worker_id::num_workers]

        chain = itertools.chain.from_iterable(datasets)
        yield from chain

    def __len__(self) -> int:
        return sum(len(ds) for ds in self._datasets)


class PytorchShardingDataset(Dataset[T_co]):
    def __init__(
        self,
        datasets: Iterable[Dataset[T_co]],
        num_workers: int,
        chunk_size: int,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        from torch.utils import data

        dataset = ShardableChainDataset(datasets, shuffle)
        self._loader = data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=chunk_size,
            collate_fn=lambda x: x,
            pin_memory=True,
        )

    def __iter__(self) -> Iterator[T_co]:
        for data in self._loader:
            yield from data
