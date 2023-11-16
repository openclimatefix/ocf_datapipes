# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from concurrent import futures
from typing import Callable, Iterator, Optional, Sized, TypeVar

from torch.utils.data.datapipes.utils.common import _check_unpickable_fn, validate_input_col
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes._decorator import functional_datapipe

T_co = TypeVar("T_co", covariant=True)


def _no_op_fn(*args):
    """
    No-operation function, returns passed arguments.
    """
    if len(args) == 1:
        return args[0]
    return args


@functional_datapipe("ocf_threadpool_map")
class ThreadPoolMapperIterDataPipe(IterDataPipe[T_co]):
    r"""
    Applies a function over each item from the source DataPipe concurrently
    using ``ThreadPoolExecutor`` (functional name: ``threadpool_map``).
    The function can be any regular Python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.

    Args:
        source_datapipe: Source IterDataPipe
        fn: Function being applied over each item
        input_col: Index or indices of data which ``fn`` is applied, such as:
            - ``None`` as default to apply ``fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.
        output_col: Index of data where result of ``fn`` is placed. ``output_col`` can be specified
            only when ``input_col`` is not ``None``
            - ``None`` as default to replace the index that ``input_col`` specified; For ``input_col`` with
              multiple indices, the left-most one is used, and other indices will be removed.
            - Integer is used for list/tuple. ``-1`` represents to append result at the end.
            - Key is used for dict. New key is acceptable.
        scheduled_tasks: How many tasks will be scheduled at any given time (Default value: 128)
        max_workers: Maximum number of threads to execute function calls
        **threadpool_kwargs: additional arguments to be given to the ``ThreadPoolExecutor``
    Note:
         For more information about ``max_workers`` and additional arguments for the ``ThreadPoolExecutor``
         please refer to: https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
    Note:
        For optimal use of all threads, ``scheduled_tasks`` > ``max_workers`` is strongly recommended. The higher the
        variance of the time needed to finish execution of the given ``fn`` is, the higher the value
        of ``scheduled_tasks`` needs to be to avoid threads sitting idle while waiting
        for the next result (as results are returned in correct order).
        However, too high value of ``scheduled_tasks`` might lead to long waiting period until the first element is yielded
        as ``next`` is called ``scheduled_tasks`` many times on ``source_datapipe`` before yielding.
        We encourage you to try out different values of ``max_workers`` and ``scheduled_tasks``
        in search for optimal values for your use-case.


    Example:
    .. testsetup::
        from torch.utils.data.datapipes.iter import IterableWrapper
        import requests
        import time
        from unittest.mock import MagicMock
        requests.get = MagicMock()
        urls = []
    .. testcode::
        # fetching html from remote
        def fetch_html(url: str, **kwargs):
            r = requests.get(url, **kwargs)
            r.raise_for_status()
            return r.content
        dp = IterableWrapper(urls)
        dp = dp.threadpool_map(fetch_html,max_workers=16)
    .. testcode::
        def mul_ten(x):
            time.sleep(0.1)
            return x * 10
        dp = IterableWrapper([(i, i) for i in range(50)])
        dp = dp.threadpool_map(mul_ten, input_col=1)
        print(list(dp))
    .. testoutput::
        [(0, 0), (1, 10), (2, 20), (3, 30), ...]
    .. testcode::
        dp = IterableWrapper([(i, i) for i in range(50)])
        dp = dp.threadpool_map(mul_ten, input_col=1, output_col=-1)
        print(list(dp))
    .. testoutput::
        [(0, 0, 0), (1, 1, 10), (2, 2, 20), (3, 3, 30), ...]
    """

    datapipe: IterDataPipe
    fn: Callable

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        fn: Callable,
        input_col=None,
        output_col=None,
        scheduled_tasks: int = 128,
        max_workers: Optional[int] = None,
        **threadpool_kwargs,
    ) -> None:
        super().__init__()
        self.datapipe = source_datapipe

        _check_unpickable_fn(fn)
        self.fn = fn  # type: ignore[assignment]

        if scheduled_tasks <= 0:
            raise ValueError("'scheduled_tasks' is required to be a positive integer.")
        self.scheduled_tasks = scheduled_tasks
        if max_workers is not None and max_workers <= 0:
            raise ValueError("'max_workers' is required to be a positive integer.")
        self.max_workers = max_workers
        self.threadpool_kwargs = threadpool_kwargs

        self.input_col = input_col
        if input_col is None and output_col is not None:
            raise ValueError("`output_col` must be None when `input_col` is None.")
        if isinstance(output_col, (list, tuple)):
            if len(output_col) > 1:
                raise ValueError("`output_col` must be a single-element list or tuple")
            output_col = output_col[0]
        self.output_col = output_col
        validate_input_col(fn, input_col)

    def _apply_fn(self, data):
        if self.input_col is None and self.output_col is None:
            return self.fn(data)

        if self.input_col is None:
            res = self.fn(data)
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple(data[col] for col in self.input_col)
            res = self.fn(*args)
        else:
            res = self.fn(data[self.input_col])

        # Copy tuple to list and run in-place modification because tuple is immutable.
        if isinstance(data, tuple):
            t_flag = True
            data = list(data)
        else:
            t_flag = False

        if self.output_col is None:
            if isinstance(self.input_col, (list, tuple)):
                data[self.input_col[0]] = res
                for idx in sorted(self.input_col[1:], reverse=True):
                    del data[idx]
            else:
                data[self.input_col] = res
        else:
            if self.output_col == -1:
                data.append(res)
            else:
                data[self.output_col] = res

        # Convert list back to tuple
        return tuple(data) if t_flag else data

    def __iter__(self) -> Iterator[T_co]:
        with futures.ThreadPoolExecutor(
            max_workers=self.max_workers, **self.threadpool_kwargs
        ) as executor:
            futures_deque: deque = deque()
            has_next = True
            itr = iter(self.datapipe)
            for _ in range(self.scheduled_tasks):
                try:
                    futures_deque.append(executor.submit(self._apply_fn, next(itr)))
                except StopIteration:
                    has_next = False
                    break

            while len(futures_deque) > 0:
                if has_next:
                    try:
                        futures_deque.append(executor.submit(self._apply_fn, next(itr)))
                    except StopIteration:
                        has_next = False
                yield futures_deque.popleft().result()

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
