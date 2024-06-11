"""Datapipes from TorchData that have been copied in for use with core PyTorch Datapipes"""

from typing import Iterator, List, Optional, Sequence, Sized, Tuple, TypeVar

from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.iter.combining import T_co, _ChildDataPipe, _ForkerIterDataPipe

T = TypeVar("T")


# https://github.com/pytorch/data/issues/865
@functional_datapipe("zip_ocf")
class ZipperIterDataPipe(IterDataPipe[Tuple[T_co]]):
    """
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    The output is stopped as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Iterable DataPipes being aggregated
    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(5)), IterableWrapper(range(10, 15)),
        >>> IterableWrapper(range(20, 25))
        >>> list(dp1.zip(dp2, dp3))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (3, 13, 23), (4, 14, 24)]
    """

    datapipes: Tuple[IterDataPipe]
    length: Optional[int]

    def __init__(self, *datapipes: IterDataPipe):
        """Init"""
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError(
                "All inputs are required to be `IterDataPipe` " "for `ZipIterDataPipe`."
            )
        super().__init__()
        self.datapipes = datapipes  # type: ignore[assignment]
        self.length = None

    def __iter__(self) -> Iterator[Tuple[T_co]]:
        """Iter"""
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        for data in zip(*iterators):
            yield data

    def __len__(self) -> int:
        """Len"""
        if self.length is not None:
            if self.length == -1:
                raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
            return self.length
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            self.length = min(len(dp) for dp in self.datapipes)
        else:
            self.length = -1
        return len(self)


@functional_datapipe("repeat")
class RepeaterIterDataPipe(IterDataPipe[T_co]):
    """
    Repeater

    Repeatedly yield each element of source DataPipe for the specified number of times before
    moving onto the next element (functional name: ``repeat``). Note that no copy is made
    in this DataPipe,
    the same element is yielded repeatedly.

    If you would like to yield the whole DataPipe in order multiple times, use :class:`.Cycler`.

    Args:
        source_datapipe: source DataPipe that will be iterated through
        times: the number of times an element of ``source_datapipe`` will be yielded
        before moving onto the next element

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(3))
        >>> dp = dp.repeat(2)
        >>> list(dp)
        [0, 0, 1, 1, 2, 2]
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co], times: int) -> None:
        """Init"""
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        self.times: int = times
        if times <= 1:
            raise ValueError(f"The number of repetition must be > 1, got {times}")

    def __iter__(self) -> Iterator[T_co]:
        """Iter"""
        for element in self.source_datapipe:
            for _ in range(self.times):
                yield element

    def __len__(self) -> int:
        """Len"""
        return self.times * len(self.source_datapipe)


@functional_datapipe("unzip")
class UnZipperIterDataPipe(IterDataPipe[T]):
    r"""
    UnZipper Iterator DataPipe

    Takes in a DataPipe of Sequences, unpacks each Sequence, and return the
    elements in separate DataPipes
    based on their position in the Sequence (functional name: ``unzip``).
    The number of instances produced equals to
    the sequence length minus the number of columns to skip.

    Note:
        Each sequence within the DataPipe should have the same length, specified by
        the input argument `sequence_length`.

    Args:
        source_datapipe: Iterable DataPipe with sequences of data
        sequence_length: Length of the sequence within the source_datapipe.
            All elements should have the same length.
        buffer_size: this restricts how far ahead the leading child DataPipe can read relative
            to the slowest child DataPipe. Use -1 for the unlimited buffer.
        columns_to_skip: optional indices of columns that the DataPipe should skip
            (each index should be an integer from 0 to sequence_length - 1)

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper([(i, i + 10, i + 20) for i in range(3)])
        >>> dp1, dp2, dp3 = source_dp.unzip(sequence_length=3)
        >>> list(dp1)
        [0, 1, 2]
        >>> list(dp2)
        [10, 11, 12]
        >>> list(dp3)
        [20, 21, 22]
    """

    def __new__(
        cls,
        source_datapipe: IterDataPipe[Sequence[T]],
        sequence_length: int,
        buffer_size: int = 1000,
        columns_to_skip: Optional[Sequence[int]] = None,
    ):
        """Create a new instance"""
        if columns_to_skip is None:
            instance_ids = list(range(sequence_length))
        else:
            skips = set(columns_to_skip)
            instance_ids = [i for i in range(sequence_length) if i not in skips]

        if len(instance_ids) == 0:
            raise RuntimeError(
                "All instances are being filtered out in UnZipperIterDataPipe. Please check"
                "the input `sequence_length` and `columns_to_skip`."
            )

        # The implementation basically uses Forker but only yields a
        # specific element within the sequence
        container = _UnZipperIterDataPipe(source_datapipe, instance_ids, buffer_size)  # type: ignore[arg-type]
        return [_ChildDataPipe(container, i) for i in range(len(instance_ids))]


class _UnZipperIterDataPipe(_ForkerIterDataPipe):
    """Internal UnZipper"""

    def __init__(self, datapipe: IterDataPipe, instance_ids: List[int], buffer_size: int = 1000):
        """Init"""
        super().__init__(datapipe, len(instance_ids), buffer_size)  # type: ignore[arg-type]
        self.instance_ids = instance_ids

    def get_next_element_by_instance(self, instance_id: int):
        r"""
        Get next element by instance

        Note:
            Each element returned from the source datapipe is required to be a sequnce that can
            be subscribed with a column index
        """
        for return_val in super().get_next_element_by_instance(instance_id):
            yield return_val[self.instance_ids[instance_id]]

    def __getstate__(self):
        """Get state"""
        state = super().__getstate__()
        return (*state, self.instance_ids)

    def __setstate__(self, state):
        """Set state"""
        super().__setstate__(state[:-1])
        self.instance_ids = state[-1]


@functional_datapipe("set_length")
class LengthSetterIterDataPipe(IterDataPipe[T_co]):
    r"""
    Length setter

    Set the length attribute of the DataPipe, which is returned by
    ``__len__`` (functional name: ``set_length``).
    This can be used after DataPipes whose final length cannot be known in advance
    (e.g. ``filter``). If you
    know the final length with certainty, you can manually set it, which can then be used by
    DataLoader or other DataPipes.

    Note:
        This DataPipe differs from :class:`.Header` in that this doesn't restrict
        the number of elements that
        can be yielded from the DataPipe; this is strictly used for setting an attribute
        so that it can be used later.

    Args:
        source_datapipe: a DataPipe
        length: the integer value that will be set as the length

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10)).filter(lambda x: x < 5).set_length(3)
        >>> list(dp)  # Notice that the number of elements yielded is unchanged
        [0, 1, 2, 3, 4]
        >>> len(dp)
        3
        >>> header_dp = IterableWrapper(range(10)).filter(lambda x: x < 5).header(3)
        >>> list(header_dp)  # Use `.header()` if you want to limit the number of elements yielded
        [0, 1, 2]
        >>> len(header_dp)
        3
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co], length: int) -> None:
        """Init"""
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        assert length >= 0
        self.length: int = length

    def __iter__(self) -> Iterator[T_co]:
        """Iter"""
        yield from self.source_datapipe

    def __len__(self) -> int:
        """Len"""
        return self.length


@functional_datapipe("header")
class HeaderIterDataPipe(IterDataPipe[T_co]):
    r"""
    Header Iterator DataPipe

    Yields elements from the source DataPipe from the start, up to the specfied
    limit (functional name: ``header``).

    If you would like to manually set the length of a DataPipe to a certain value;
     we recommend you to
    use :class:`.LengthSetter`.

    Args:
        source_datapipe: the DataPipe from which elements will be yielded
        limit: the number of elements to yield before stopping

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> header_dp = dp.header(3)
        >>> list(header_dp)
        [0, 1, 2]
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co], limit: Optional[int] = 10) -> None:
        """Init"""
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        self.limit: Optional[int] = limit

    def __iter__(self) -> Iterator[T_co]:
        """Iter"""
        i: int = 0
        for value in self.source_datapipe:
            i += 1
            if self.limit is None or i <= self.limit:
                yield value
            else:
                break

    def __len__(self) -> int:
        """Len"""
        try:
            source_len = len(self.source_datapipe)
            return source_len if self.limit is None else min(source_len, self.limit)
        except TypeError as error:
            if self.limit is None:
                raise TypeError(
                    "The length of this HeaderIterDataPipe cannot be determined."
                ) from error
            return self.limit
