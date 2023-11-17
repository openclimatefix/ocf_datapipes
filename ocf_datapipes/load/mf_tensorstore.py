import os
from functools import partial
from glob import glob
from io import BytesIO
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    Type,
    Union,
    cast,
    overload,
)

from xarray.core.combine import (
    _infer_concat_order_from_positions,
    _nested_combine,
    combine_by_coords,
)
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import Index
from xarray.core.utils import is_remote_uri
from xarray.backends.common import _normalize_path
import xarray_tensorstore

if TYPE_CHECKING:
    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None  # type: ignore
    from io import BufferedIOBase

    from xarray.core.types import (
        CombineAttrsOptions,
        CompatOptions,
        JoinOptions,
        NestedSequence,
    )


def open_mfdataset_tensorstore(
    paths: str | NestedSequence[str | os.PathLike],
    chunks=None,
    concat_dim: str
    | DataArray
    | Index
    | Sequence[str]
    | Sequence[DataArray]
    | Sequence[Index]
    | None = None,
    compat: CompatOptions = "no_conflicts",
    preprocess: Callable[[Dataset], Dataset] | None = None,
    engine=None,
    data_vars: Literal["all", "minimal", "different"] | list[str] = "all",
    coords="different",
    combine: Literal["by_coords", "nested"] = "by_coords",
    parallel: bool = False,
    join: JoinOptions = "outer",
    attrs_file: str | os.PathLike | None = None,
    combine_attrs: CombineAttrsOptions = "override",
    **kwargs,
) -> Dataset:
    """Open multiple files as a single dataset.

    If combine='by_coords' then the function ``combine_by_coords`` is used to combine
    the datasets into one before returning the result, and if combine='nested' then
    ``combine_nested`` is used. The filepaths must be structured according to which
    combining function is used, the details of which are given in the documentation for
    ``combine_by_coords`` and ``combine_nested``. By default ``combine='by_coords'``
    will be used. Requires dask to be installed. See documentation for
    details on dask [1]_. Global attributes from the ``attrs_file`` are used
    for the combined dataset.

    Parameters
    ----------
    paths : str or nested sequence of paths
        Either a string glob in the form ``"path/to/my/files/*.nc"`` or an explicit list of
        files to open. Paths can be given as strings or as pathlib Paths. If
        concatenation along more than one dimension is desired, then ``paths`` must be a
        nested list-of-lists (see ``combine_nested`` for details). (A string glob will
        be expanded to a 1-dimensional list.)
    chunks : int, dict, 'auto' or None, optional
        Dictionary with keys given by dimension names and values given by chunk sizes.
        In general, these should divide the dimensions of each dataset. If int, chunk
        each dimension by ``chunks``. By default, chunks will be chosen to load entire
        input files into memory at once. This has a major impact on performance: please
        see the full documentation for more details [2]_.
    concat_dim : str, DataArray, Index or a Sequence of these or None, optional
        Dimensions to concatenate files along.  You only need to provide this argument
        if ``combine='nested'``, and if any of the dimensions along which you want to
        concatenate is not a dimension in the original datasets, e.g., if you want to
        stack a collection of 2D arrays along a third dimension. Set
        ``concat_dim=[..., None, ...]`` explicitly to disable concatenation along a
        particular dimension. Default is None, which for a 1D list of filepaths is
        equivalent to opening the files separately and then merging them with
        ``xarray.merge``.
    combine : {"by_coords", "nested"}, optional
        Whether ``xarray.combine_by_coords`` or ``xarray.combine_nested`` is used to
        combine all the data. Default is to use ``xarray.combine_by_coords``.
    compat : {"identical", "equals", "broadcast_equals", \
              "no_conflicts", "override"}, default: "no_conflicts"
        String indicating how to compare variables of the same name for
        potential conflicts when merging:

         * "broadcast_equals": all values must be equal when variables are
           broadcast against each other to ensure common dimensions.
         * "equals": all values and dimensions must be the same.
         * "identical": all values, dimensions and attributes must be the
           same.
         * "no_conflicts": only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
         * "override": skip comparing and pick variable from first dataset

    preprocess : callable, optional
        If provided, call this function on each dataset prior to concatenation.
        You can find the file-name from which each dataset was loaded in
        ``ds.encoding["source"]``.
    engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib", \
        "pseudonetcdf", "zarr", None}, installed backend \
        or subclass of xarray.backends.BackendEntrypoint, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        "netcdf4".
    data_vars : {"minimal", "different", "all"} or list of str, default: "all"
        These data variables will be concatenated together:
          * "minimal": Only data variables in which the dimension already
            appears are included.
          * "different": Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * "all": All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the "minimal" data variables.
    coords : {"minimal", "different", "all"} or list of str, optional
        These coordinate variables will be concatenated together:
         * "minimal": Only coordinates in which the dimension already appears
           are included.
         * "different": Coordinates which are not equal (ignoring attributes)
           across all datasets are also concatenated (as well as all for which
           dimension already appears). Beware: this option may load the data
           payload of coordinate variables into memory if they are not already
           loaded.
         * "all": All coordinate variables will be concatenated, except
           those corresponding to other dimensions.
         * list of str: The listed coordinate variables will be concatenated,
           in addition the "minimal" coordinates.
    parallel : bool, default: False
        If True, the open and preprocess steps of this function will be
        performed in parallel using ``dask.delayed``. Default is False.
    join : {"outer", "inner", "left", "right", "exact", "override"}, default: "outer"
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    attrs_file : str or path-like, optional
        Path of the file used to read global attributes from.
        By default global attributes are read from the first file provided,
        with wildcard matches sorted by filename.
    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts", \
                     "override"} or callable, default: "override"
        A callable or a string indicating how to combine attrs of the objects being
        merged:

        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "drop_conflicts": attrs from all objects are combined, any that have
          the same name but different values are dropped.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.

        If a callable, it must expect a sequence of ``attrs`` dicts and a context object
        as its only parameters.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset

    Notes
    -----
    ``open_mfdataset`` opens files with read-only access. When you modify values
    of a Dataset, even one linked to files on disk, only the in-memory copy you
    are manipulating in xarray is modified: the original file on disk is never
    touched.

    See Also
    --------
    combine_by_coords
    combine_nested
    open_dataset

    Examples
    --------
    A user might want to pass additional arguments into ``preprocess`` when
    applying some operation to many individual files that are being opened. One route
    to do this is through the use of ``functools.partial``.

    >>> from functools import partial
    >>> def _preprocess(x, lon_bnds, lat_bnds):
    ...     return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))
    ...
    >>> lon_bnds, lat_bnds = (-110, -105), (40, 45)
    >>> partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    >>> ds = xr.open_mfdataset(
    ...     "file_*.nc", concat_dim="time", preprocess=_preprocess
    ... )  # doctest: +SKIP

    References
    ----------

    .. [1] https://docs.xarray.dev/en/stable/dask.html
    .. [2] https://docs.xarray.dev/en/stable/dask.html#chunking-and-performance
    """
    if isinstance(paths, str):
        if is_remote_uri(paths) and engine == "zarr":
            try:
                from fsspec.core import get_fs_token_paths
            except ImportError as e:
                raise ImportError(
                    "The use of remote URLs for opening zarr requires the package fsspec"
                ) from e

            fs, _, _ = get_fs_token_paths(
                paths,
                mode="rb",
                storage_options=kwargs.get("backend_kwargs", {}).get("storage_options", {}),
                expand=False,
            )
            tmp_paths = fs.glob(fs._strip_protocol(paths))  # finds directories
            paths = [fs.get_mapper(path) for path in tmp_paths]
        elif is_remote_uri(paths):
            raise ValueError(
                "cannot do wild-card matching for paths that are remote URLs "
                f"unless engine='zarr' is specified. Got paths: {paths}. "
                "Instead, supply paths as an explicit list of strings."
            )
        else:
            paths = sorted(glob(_normalize_path(paths)))
    elif isinstance(paths, os.PathLike):
        paths = [os.fspath(paths)]
    else:
        paths = [os.fspath(p) if isinstance(p, os.PathLike) else p for p in paths]

    if not paths:
        raise OSError("no files to open")

    if combine == "nested":
        if isinstance(concat_dim, (str, DataArray)) or concat_dim is None:
            concat_dim = [concat_dim]  # type: ignore[assignment]

        # This creates a flat list which is easier to iterate over, whilst
        # encoding the originally-supplied structure as "ids".
        # The "ids" are not used at all if combine='by_coords`.
        combined_ids_paths = _infer_concat_order_from_positions(paths)
        ids, paths = (
            list(combined_ids_paths.keys()),
            list(combined_ids_paths.values()),
        )
    elif combine == "by_coords" and concat_dim is not None:
        raise ValueError(
            "When combine='by_coords', passing a value for `concat_dim` has no "
            "effect. To manually combine along a specific dimension you should "
            "instead specify combine='nested' along with a value for `concat_dim`.",
        )

    open_ = xarray_tensorstore.open_zarr
    getattr_ = getattr

    datasets = [open_(p) for p in paths]
    closers = [getattr_(ds, "_close") for ds in datasets]
    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]

    # Combine all datasets, closing them in case of a ValueError
    try:
        if combine == "nested":
            # Combined nested list by successive concat and merge operations
            # along each dimension, using structure given by "ids"
            combined = _nested_combine(
                datasets,
                concat_dims=concat_dim,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                ids=ids,
                join=join,
                combine_attrs=combine_attrs,
            )
        elif combine == "by_coords":
            # Redo ordering from coordinates, ignoring how they were ordered
            # previously
            combined = combine_by_coords(
                datasets,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                join=join,
                combine_attrs=combine_attrs,
            )
        else:
            raise ValueError(
                "{} is an invalid option for the keyword argument" " ``combine``".format(combine)
            )
    except ValueError:
        for ds in datasets:
            ds.close()
        raise

    combined.set_close(partial(_multi_file_closer, closers))

    # read global attributes from the attrs_file or from the first dataset
    if attrs_file is not None:
        if isinstance(attrs_file, os.PathLike):
            attrs_file = cast(str, os.fspath(attrs_file))
        combined.attrs = datasets[paths.index(attrs_file)].attrs

    return combined


def _multi_file_closer(closers):
    for closer in closers:
        closer()
