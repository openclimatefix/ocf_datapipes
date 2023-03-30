from collections import deque
from concurrent import futures
from typing import Callable, Iterator


def run_with_threadpool(
    data_iterator: Iterator,
    fn: Callable,
    max_workers: int,
    scheduled_tasks: int,
    **threadpool_kwargs
):
    with futures.ThreadPoolExecutor(max_workers=max_workers, **threadpool_kwargs) as executor:
        futures_deque: deque = deque()
        has_next = True
        itr = iter(data_iterator)
        for _ in range(scheduled_tasks):
            try:
                futures_deque.append(executor.submit(fn, next(itr)))
            except StopIteration:
                has_next = False
                break

        while len(futures_deque) > 0:
            if has_next:
                try:
                    futures_deque.append(executor.submit(fn, next(itr)))
                except StopIteration:
                    has_next = False
            yield futures_deque.popleft().result()
