# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import functools
import hashlib
import os
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
)

import numpy as np
import xarray as xr

from . import backends
from ._qubed import expand_as_qube
from .graph import Graph, Output
from .graph import Node as BaseNode
from .nodetree import nodetree_array, nodetree_arrays, nodetree_from_dict

PayloadFunc = Callable | str


class Payload:
    """Class for detailing function, args and kwargs to be computing in a graph node"""

    def __init__(
        self,
        func: PayloadFunc,
        args: Iterable | None = None,
        kwargs: dict | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.args: list
        if isinstance(func, functools.partial):
            if args is not None or kwargs is not None:
                raise ValueError("Partial function should not have args or kwargs")
            self.func = func.func
            self.args = list(func.args)
            self.kwargs = func.keywords
        else:
            self.func = func
            self.args = [] if args is None else list(args)
            self.kwargs = kwargs or {}

        self.metadata = getattr(self.func, "_cascade", {})
        self.metadata.update(metadata or {})

    def to_tuple(self) -> tuple:
        """Return
        ------
        tuple, containing function, arguments and kwargs
        """
        return (self.func, self.args, self.kwargs, self.metadata)

    def name(self) -> str:
        """Return
        ------
        str, name of function, or if a partial function, the function name and partial
        arguments
        """
        if isinstance(self.func, str):
            return self.func
        if hasattr(self.func, "__name__"):
            return self.func.__name__
        return ""

    def __str__(self) -> str:
        return f"{self.name()}{self.args}{self.kwargs}:{self.metadata}:{repr(self.func)}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Payload):
            return False
        return str(self) == str(other)

    def copy(self) -> "Payload":
        return Payload(self.func, self.args, self.kwargs, metadata=self.metadata)


def custom_hash(string: str) -> str:
    ret = hashlib.sha256()
    ret.update(string.encode())
    return ret.hexdigest()


Coord = tuple[str, list[Any]]
Input = BaseNode | Output

P = ParamSpec("P")
R = TypeVar("R")


def capture_payload_metadata(func: Callable[P, R]) -> Callable[P, R]:
    """Wrap a function which returns a new action and insert
    given `payload_metadata`
    """

    # @functools.wraps(func)
    def decorator(*args, **kwargs):
        metadata = kwargs.pop("payload_metadata", {})
        result = func(*args, **kwargs)

        if isinstance(result, Action):
            for _, narray in nodetree_arrays(result.nodes):
                for node in np.atleast_1d(narray.values).flatten():
                    node.payload.metadata.update(metadata)
        elif isinstance(result, Node):
            result.payload.metadata.update(metadata)
        else:
            raise TypeError(f"Expected Action or Node, got {type(result)}")
        return result

    return decorator


class Node(BaseNode):
    def __init__(
        self,
        payload: PayloadFunc | Payload,
        inputs: Input | Sequence[Input] = [],
        num_outputs: int = 1,
        name: str | None = None,
    ):
        self._for_copy = (payload, inputs, num_outputs, name)
        if not isinstance(payload, Payload):
            payload = Payload(payload)
        else:
            payload = payload.copy()
        if isinstance(inputs, Input):
            inputs = [inputs]
        # Insert inputs not already present in args
        for x in range(len(inputs)):
            if self.input_name(x) not in payload.args:
                payload.args.append(self.input_name(x))

        if name is None:
            name = payload.name()
        name += ":" + custom_hash(
            f'{payload}{[x.name if isinstance(x, BaseNode) else f"{x.parent.name}.{x.name}" for x in inputs]}'
        )

        super().__init__(
            name,
            outputs=(
                None
                if num_outputs == 1
                else [f"{x:0{len(str(num_outputs - 1))}d}" for x in range(num_outputs)]
            ),
            payload=payload,
            **{self.input_name(x): node for x, node in enumerate(inputs)},
        )
        self.attributes: dict[str, Any] = {}

    @staticmethod
    def input_name(index: int):
        return f"input{index}"

    def __str__(self) -> str:
        return f"Node {self.name}, inputs: {[x.parent.name for x in self.inputs.values()]}, payload: {self.payload}"

    def copy(self) -> "Node":
        return self.__class__(*self._for_copy)


class Action:

    REGISTRY: dict[str, type[Action]] = {}

    def __init__(self, nodetree: xr.DataTree, yields: Optional[Coord] = None):
        if yields:
            ydim, ycoords = yields
            new_nodes = {}
            for npath, narray in nodetree_arrays(nodetree):
                new_array = xr.apply_ufunc(
                    lambda x: np.asarray([x.get_output(out) for out in x.outputs]),
                    narray,
                    output_core_dims=[[ydim]],
                    vectorize=True,
                )
                new_array.coords[ydim] = ycoords
                new_nodes[npath] = new_array
        else:
            new_nodes = nodetree.to_dict()
        self.nodes = nodetree_from_dict(new_nodes)

    def graph(self) -> Graph:
        """Creates graph from the nodes of the action.

        Return
        ------
        Graph instance constructed from list of nodes

        """
        sinks = set()
        for _, array in nodetree_arrays(self.nodes):
            for node in array.data.flatten():
                if isinstance(node, Output):
                    sinks.add(node.parent)
                else:
                    sinks.add(node)
        return Graph(list(sinks))

    @classmethod
    def register(cls, name: str, obj: type[Action]):
        """Register an Action class under `name`

        Will be accessible from the fluent API as `Action().<name>`

        Parameters
        ----------
        name : str
            Name to register Action under
        obj : type[Action]
            Action class to register

        Raises
        ------
        ValueError
            If `name` is an attr on `obj` or `name` is already registered
        """

        if not issubclass(obj, Action):
            raise TypeError(f"obj must be a type of Action, not {type(obj)}")

        if name in cls.REGISTRY:
            raise ValueError(f"{name} already registered, will not override")

        if hasattr(obj, name):
            raise ValueError(
                f"Action class {obj} already has an attribute {name}, will not override"
            )

        cls.REGISTRY[name] = obj

    @classmethod
    def flush_registry(cls):
        """Flush the registry of all registered actions"""
        cls.REGISTRY = {}

    def as_action(self, other) -> Action:
        """Parse action into another action class"""
        return other(self.nodes)

    def join(
        self,
        other_action: "Action",
        dim: str | Coord,
        match_coord_values: bool = False,
    ) -> "Action":
        node_arrays = {}
        for npath, narray in nodetree_arrays(self.nodes):
            oarray = nodetree_array(other_action.nodes, npath)
            if match_coord_values:
                for coord, values in narray.coords.items():
                    if coord in oarray.coords:
                        assigned = oarray.assign_coords(
                            **{str(coord): values}
                        )
                        other_action.nodes[npath] = assigned.to_dataset()
            node_arrays[npath] = xr.concat(
                [narray, oarray],
                dim=(
                    dim if isinstance(dim, str) else xr.DataArray(dim[1], name=dim[0])
                ),
                join="exact",
                combine_attrs="no_conflicts",
                coords="minimal",
            )
        return type(self)(nodetree_from_dict(node_arrays))

    def transform(
        self,
        func: Callable[["Action", Any], "Action"],
        params: list,
        dim: str | Coord,
        axis: int = 0,
        path: Optional[str] = None,
    ) -> "Action":
        """Create new nodes by applying function on action with different
        parameters. The result actions from applying function are joined
        along the specified dimension.

        Parameters
        ----------
        func: function with signature func(Action, *args) -> Action
        params: list, containing different arguments to pass into func
        for generating new nodes
        dim: str or `Coord`, name of dimension to join actions or `Coord` specifying new dimension name and
        coordinate values
        axis: int, position to insert new dimension
        path: str, path to select subset of nodes to operate on, if provided

        Return
        ------
        Action
        """
        res = None
        dim_values: list[int] | np.ndarray[Any, Any]
        if isinstance(dim, str):
            dim_name = dim
            dim_values = list(range(len(params)))
        else:
            dim_name = dim[0]
            dim_values = dim[1]

        for index, param in enumerate(params):
            new_res = func(self.select(path=path), *param)
            if dim_name not in new_res.nodes.coords:
                new_res._add_dimension(dim_name, dim_values[index], axis, path=path)
            if res is None:
                res = new_res
            else:
                res = res.join(new_res, dim_name)

        if not res:
            raise ValueError("No new actions generated from transform")
        # Remove expanded dimension if only a single element
        res._squeeze_dimension(dim_name, path=path)
        # Modify node array path to contain new nodes
        new_nodes = {npath: narray for npath, narray in nodetree_arrays(self.nodes)}
        new_nodes.update(
            {npath: narray for npath, narray in nodetree_arrays(res.nodes)}
        )
        return type(self)(nodetree_from_dict(new_nodes))

    def broadcast(
        self,
        other_action: "Action",
        exclude: list[str] | None = None,
        path: Optional[str] = None,
    ) -> "Action":
        """Broadcast nodes against nodes in other_action

        Parameters
        ----------
        other_action: Action containing nodes to broadcast against
        exclude: List of str, dimension names to exclude from broadcasting
        path: Optional[str], path to select subset of nodes to operate on, if provided

        Return
        ------
        Action
        """
        node_arrays = {}
        for npath, narray in nodetree_arrays(other_action.select(path=path).nodes):
            array = nodetree_array(self.nodes, npath)
            # Ensure coordinates in existing dimensions match, otherwise obtain NaNs
            for key, values in narray.coords.items():
                if key in array.coords and (exclude is None or key not in exclude):
                    assert np.all(
                        values.data == array.coords[key].data
                    ), f"Existing coordinates must match for broadcast. Found mismatch in {key}!"
            broadcasted_nodes = array.broadcast_like(narray, exclude=exclude)
            new_nodes = np.empty(broadcasted_nodes.shape, dtype=object)
            it = np.nditer(
                array.transpose(*broadcasted_nodes.dims, missing_dims="ignore"),
                flags=["multi_index", "refs_ok"],
            )
            for node in it:
                new_nodes[it.multi_index] = Node(Payload(backends.trivial), node[()])  # type: ignore

            node_arrays[npath] = xr.DataArray(
                new_nodes,
                coords=broadcasted_nodes.coords,
                dims=broadcasted_nodes.dims,
                attrs=array.attrs,
            )
        
        new_nodes = {npath: narray for npath, narray in nodetree_arrays(self.nodes)}
        new_nodes.update(node_arrays)
        return type(self)(nodetree_from_dict(new_nodes))

    def expand(
        self,
        dim: str | Coord,
        internal_dim: int | str | Coord,
        dim_size: int | None = None,
        axis: int = 0,
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        """Create new dimension in array of nodes of specified size by
        taking elements of internal data in each node. Indexing is taken along the specified axis
        dimension of internal data and graph execution will fail if
        dim_size exceeds the dimension size of this axis in the internal data.

        Parameters
        ----------
        dim: str or `Coord`, name of dimension or `Coord` specifying new dimension name and
        coordinate values
        internal_dim: int, str or DataArray, index or name of internal dimension to expand, or
        `Coord` specifying dimension name and list of selection criteria
        dim_size: int | None, size of new dimension. If not given `internal_dim` must be `Coord`
        axis: int, position to insert new dimension
        path: Optional[str], path to select subset of nodes to operate on, if provided
        backend_kwargs: dict, kwargs for the underlying backend take method

        Return
        ------
        Action
        """
        if isinstance(internal_dim, (int, str)):
            if dim_size is None:
                raise TypeError(
                    "If `internal_dim` is str or int, then `dim_size` must be provided"
                )
            params = [(i, internal_dim, backend_kwargs) for i in range(dim_size)]
        else:
            params = [(x, internal_dim[0], backend_kwargs) for x in internal_dim[1]]

        if not isinstance(dim, str) and len(params) != len(dim[1]):
            raise ValueError(
                "Length of values in `dim` must match `dim_size` or length of values in `internal_dim`"
            )
        return self.transform(_expand_transform, params, dim, axis=axis, path=path)
    
    expand_as_qube = expand_as_qube

    def map(
        self,
        payload: PayloadFunc | Payload | np.ndarray[Any, Any],
        yields: Coord | None = None,
        path: Optional[str] = None,
    ) -> "Action":
        """Apply specified payload on all nodes. If argument is an array of payloads,
        this must be the same size as the array of nodes and each node gets a
        unique payload from the array

        Parameters
        ----------
        payload: function or array of functions
        yields: Coord | None, name and coords of dimension yielded by payload, if generator
        path: str, path to select subset of nodes to operate on, if provided

        Return
        ------
        Action where nodes are a result of applying the same
        payload to all nodes, or in the case where payload is an array,
        applying a different payload to each node

        Raises
        ------
        AssertionError if the shape of the payload array does not match the shape of the
        array of nodes
        """
        # NOTE this method is really not mypy friendly, just ignore everything
        node_arrays = {}
        for npath, narray in nodetree_arrays(self.select(path=path).nodes):
            if not isinstance(payload, PayloadFunc | Payload):  # type: ignore
                payload = np.asarray(payload)
                assert payload.shape == narray.shape, (
                    f"For unique payloads for each node, payload shape {payload.shape}"
                    f"must match node array shape {narray.shape}"
                )

            # Applies operation to every node, keeping node array structure
            new_nodes = np.empty(narray.shape, dtype=object)
            it = np.nditer(narray, flags=["multi_index", "refs_ok"])
            node_payload = payload
            for node in it:
                if not isinstance(payload, PayloadFunc | Payload):  # type: ignore
                    node_payload = payload[it.multi_index]  # type: ignore
                new_nodes[it.multi_index] = Node(
                    node_payload,  # type: ignore
                    node[()],  # type: ignore
                    num_outputs=len(yields[1]) if yields else 1,
                )

            node_arrays[npath] = xr.DataArray(
                new_nodes,
                coords=narray.coords,
                dims=narray.dims,
                attrs=narray.attrs,
            )

        new_nodes = {npath: narray for npath, narray in nodetree_arrays(self.nodes)}
        new_nodes.update(node_arrays)
        return type(self)(nodetree_from_dict(new_nodes), yields)

    def reduce(
        self,
        payload: PayloadFunc | Payload,
        yields: Coord | None = None,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        path: Optional[str] = None,
    ) -> "Action":
        """Reduction operation across the named dimension using the provided
        function in the payload. If batch_size > 1 and less than the size
        of the named dimension, the reduction will be computed first in
        batches and then aggregated, otherwise no batching will be performed.

        Parameters
        ----------
        payload: function for performing the reduction
        yields: Coord | None, name and coords of dimension yielded by payload, if generator
        dim: str, name of dimension along which to reduce
        batch_size: int, size of batches to split reduction into. If 0,
        computation is not batched
        keep_dim: bool, whether to keep the reduced dimension in the result. Dimension
        is kept in the original axis position
        path: str, path to select subset of nodes to operate on, if provided

        Return
        ------
        Action

        Raises
        ------
        ValueError if payload function is not batchable and batch_size is not 0
        """
        node_arrays = {}
        for npath, narray in nodetree_arrays(self.select(path=path).nodes):
            if len(dim) == 0:
                dim = str(narray.dims[0])

            batched = self.select(path=npath)
            level = 0
            if not isinstance(payload, Payload):
                payload = Payload(payload)
            if yields and batch_size != 0:
                raise ValueError("Can not batch the execution of a generator")
            if batch_size > 1 and batch_size < nodetree_array(batched.nodes).sizes[dim]:
                if not getattr(payload.func, "batchable", False):
                    raise ValueError(
                        f"Function {payload.func.name()} is not batchable, but batch_size {batch_size} is specified"
                    )

                while batch_size < nodetree_array(batched.nodes).sizes[dim]:
                    lst = nodetree_array(batched.nodes).coords[dim].data
                    batched = batched.transform(
                        _batch_transform,
                        [
                            ({dim: lst[i : i + batch_size]}, payload)  # noqa: E203
                            for i in range(0, len(lst), batch_size)
                        ],
                        f"batch.{level}.{dim}",
                        path=npath,
                    )
                    dim = f"batch.{level}.{dim}"
                    level += 1

            batched_narray = nodetree_array(batched.nodes)
            new_dims = [x for x in batched_narray.dims if x != dim]
            transposed_nodes = batched_narray.transpose(dim, *new_dims)
            new_nodes = np.empty(transposed_nodes.shape[1:], dtype=object)
            it = np.nditer(new_nodes, flags=["multi_index", "refs_ok"])
            for _ in it:
                inputs = transposed_nodes[(slice(None, None, 1), *it.multi_index)].data
                new_nodes[it.multi_index] = Node(
                    payload, inputs, num_outputs=len(yields[1]) if yields else 1
                )

            new_coords = {key: batched_narray.coords[key] for key in new_dims}
            # Propagate scalar coords
            new_coords.update(
                {
                    k: v
                    for k, v in batched_narray.coords.items()
                    if k not in batched_narray.dims
                }
            )
            nodes = xr.DataArray(
                new_nodes,
                coords=new_coords,
                dims=new_dims,
                attrs=batched_narray.attrs,
            )
            if keep_dim:
                nodes = nodes.expand_dims(
                    {dim: [f"{nodes.coords[dim][0]}-{nodes.coords[dim][-1]}"]},
                    nodes.dims.index(dim),
                )
            node_arrays[npath] = nodes
        
        new_nodes = {npath: narray for npath, narray in nodetree_arrays(self.nodes)}
        new_nodes.update(node_arrays)
        return type(batched)(nodetree_from_dict(new_nodes), yields)

    @capture_payload_metadata
    def flatten(
        self,
        dim: str = "",
        axis: int = 0,
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        """Flattens the array of nodes along specified dimension by creating new
        nodes from stacking internal data of nodes along that dimension.

        Parameters
        ----------
        dim: str, name of dimension to flatten along
        axis: int, axis of new dimension in internal data
        path: str, path to select subset of nodes to operate on, if provided
        backend_kwargs: dict, kwargs for the underlying array module stack method

        Return
        ------
        Action
        """
        return self.reduce(
            Payload(backends.stack, kwargs={"axis": axis, **backend_kwargs}),
            dim=dim,
            path=path,
        )

    def set_path(self, path: str) -> "Action":
        """Create path for current node array

        Parameters
        ----------
        path: str, new path for node array

        Raises
        ------
        NotImplementedError if multiple node arrays are present
        """
        if len(self.nodes.leaves) > 1:
            raise NotImplementedError(
                "Multiple node arrays present, can not set single path"
            )
        return type(self)(nodetree_from_dict({path: nodetree_array(self.nodes)}))


    def split(self, expansion: Optional[dict[str, PayloadFunc | Payload]] = None) -> "Action":
        """Create action containing new node arrays by splitting an existing node array
        by the specified functions in expansion

        Parameters
        ----------
        expansion: dict[str, PayloadFunc | Payload], dictionary of paths and functions to create
        new node arrays. All paths must be branches extending from an existing path, and the functions
        will be applied to the node array at the existing path to create the new node arrays at the
        branched paths

        Return
        ------
        Action
        """
        node_arrays = {npath: narray for npath, narray in nodetree_arrays(self.nodes)}
        parent = os.path.commonpath(expansion.keys())
        if parent not in node_arrays:
            raise ValueError(f"Parent path {parent} not found in node tree")
        node_arrays.pop(parent)
        action = self.select(path=parent)
        for path, func in expansion.items():
            node_arrays[path] = nodetree_array(action.map(func).nodes, parent)
        return type(self)(nodetree_from_dict(node_arrays))


    def _validate_criteria(cls, array: xr.DataArray, criteria: dict) -> tuple[bool, dict]:
        keys = list(criteria.keys())
        new_criteria = criteria.copy()
        for key in keys:
            if key not in array.dims:
                if array.coords.get(key, None) == criteria[key]:
                    new_criteria.pop(key)
                else:
                    return False, {}
        return True, new_criteria

    def select(
        self,
        criteria: dict | None = None,
        drop: bool = False,
        path: Optional[str] = None,
        **kwargs,
    ) -> "Action":
        """Create action contaning nodes match selection criteria

        Parameters
        ----------
        criteria: dict, key-value pairs specifying selection criteria
        drop: bool, drop coord variables in criteria if True
        path: str, path to select subset of nodes to operate on, if provided

        Return
        ------
        Action
        """
        crit: dict = criteria or {}
        crit.update(kwargs)

        nodes = self.nodes if path is None else self.nodes[path]
        new_nodes = {}
        for npath, narray in nodetree_arrays(nodes):     
            valid, new_criteria = self._validate_criteria(narray, crit)
            if valid:
                try:
                    new_nodes[npath] = narray.sel(**new_criteria, drop=drop)
                except KeyError:
                    pass

        if len(new_nodes) == 0:
            raise KeyError(f"No nodes match select criteria {criteria}")
        return type(self)(nodetree_from_dict(new_nodes))

    sel = select

    def iselect(
        self,
        criteria: dict | None = None,
        drop: bool = False,
        path: Optional[str] = None,
        **kwargs,
    ) -> "Action":
        """Create action contaning nodes match index selection criteria

        Parameters
        ----------
        criteria: dict, key-value pairs specifying selection criteria
        drop: bool, drop coord variables in criteria if True
        path: str, path to select subset of nodes to operate on, if provided

        Return
        ------
        Action
        """
        crit: dict = criteria or {}
        crit.update(kwargs)

        nodes = self.nodes if path is None else self.nodes[path]
        new_nodes = {}
        for npath, narray in nodetree_arrays(nodes):     
            valid, new_criteria = self._validate_criteria(narray, crit)
            if valid:
                try:
                    new_nodes[npath] = narray.isel(**new_criteria, drop=drop)
                except IndexError:
                    pass

        if len(new_nodes) == 0:
            raise IndexError(f"No nodes match iselect criteria {criteria}")
        return type(self)(nodetree_from_dict(new_nodes))

    isel = iselect

    @capture_payload_metadata
    def concatenate(
        self,
        dim: str,
        batch_size: int = 0,
        keep_dim: bool = False,
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        return _combine_nodes(
            self, "concat", dim, batch_size, keep_dim, path, backend_kwargs
        )

    @capture_payload_metadata
    def stack(
        self,
        dim: str,
        batch_size: int = 0,
        keep_dim: bool = False,
        axis: int = 0,
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        return _combine_nodes(
            self,
            "stack",
            dim,
            batch_size,
            keep_dim,
            path,
            backend_kwargs={"axis": axis, **backend_kwargs},
        )

    @capture_payload_metadata
    def sum(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.reduce(
            Payload(backends.sum, kwargs=backend_kwargs),
            dim=dim,
            path=path,
            batch_size=batch_size,
            keep_dim=keep_dim,
        )

    @capture_payload_metadata
    def mean(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":

        action = self
        for npath, narray in nodetree_arrays(self.select(path=path).nodes):

            if len(dim) == 0:
                dim = str(narray.dims[0])
            size = narray.sizes[dim]

            if batch_size <= 1 or batch_size >= size:
                action = action.reduce(
                    Payload(backends.mean, kwargs=backend_kwargs),
                    dim=dim,
                    path=npath,
                    keep_dim=keep_dim,
                )
            else:
                action = action.sum(
                    dim=dim,
                    path=npath,
                    batch_size=batch_size,
                    keep_dim=keep_dim,
                    **backend_kwargs,
                ).divide(size, path=npath)
        return action

    @capture_payload_metadata
    def std(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        action = self
        for npath, narray in nodetree_arrays(self.select(path=path).nodes):

            if len(dim) == 0:
                dim = str(narray.dims[0])
            size = narray.sizes[dim]

            if batch_size <= 1 or batch_size >= size:
                action = action.reduce(
                    Payload(backends.std, kwargs=backend_kwargs), dim=dim, path=npath
                )

            else:
                mean_sq = action.mean(
                    dim=dim,
                    path=npath,
                    batch_size=batch_size,
                    keep_dim=keep_dim,
                    **backend_kwargs,
                ).power(2, path=npath)
                norm = (
                    action.power(2, path=npath)
                    .sum(
                        dim=dim,
                        path=npath,
                        batch_size=batch_size,
                        keep_dim=keep_dim,
                        **backend_kwargs,
                    )
                    .divide(size, path=npath)
                )
                action = norm.subtract(mean_sq, path=npath).power(0.5, path=npath)
        return action

    @capture_payload_metadata
    def max(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.reduce(
            Payload(backends.max, kwargs=backend_kwargs),
            dim=dim,
            path=path,
            batch_size=batch_size,
            keep_dim=keep_dim,
        )

    @capture_payload_metadata
    def min(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.reduce(
            Payload(backends.min, kwargs=backend_kwargs),
            dim=dim,
            path=path,
            batch_size=batch_size,
            keep_dim=keep_dim,
        )

    @capture_payload_metadata
    def prod(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.reduce(
            Payload(backends.prod, kwargs=backend_kwargs),
            dim=dim,
            path=path,
            batch_size=batch_size,
            keep_dim=keep_dim,
        )

    def __two_arg_method(
        self,
        method: Callable,
        other: "Action | float",
        path: Optional[str] = None,
        **kwargs,
    ) -> "Action":
        if isinstance(other, Action):
            return self.join(other, "**datatype**", match_coord_values=True).reduce(
                Payload(method, kwargs=kwargs), dim="**datatype**", path=path
            )
        return self.map(
            Payload(method, args=(Node.input_name(0), other), kwargs=kwargs), path=path
        )

    @capture_payload_metadata
    def subtract(
        self,
        other: "Action | float",
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.__two_arg_method(
            backends.subtract, other, path=path, **backend_kwargs
        )

    @capture_payload_metadata
    def divide(
        self,
        other: "Action | float",
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.__two_arg_method(
            backends.divide, other, path=path, **backend_kwargs
        )

    @capture_payload_metadata
    def add(
        self,
        other: "Action | float",
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.__two_arg_method(backends.add, other, path=path, **backend_kwargs)

    @capture_payload_metadata
    def multiply(
        self,
        other: "Action | float",
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.__two_arg_method(
            backends.multiply, other, path=path, **backend_kwargs
        )

    @capture_payload_metadata
    def power(
        self,
        other: "Action | float",
        path: Optional[str] = None,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.__two_arg_method(backends.pow, other, path=path, **backend_kwargs)

    def add_attributes(self, attrs: dict):
        self.nodes.attrs.update(attrs)

    def _add_dimension(self, name: str, value: Any, axis: int = 0, path: Optional[str] = None):
        new_tree = self.nodes.map_over_datasets(
            lambda ds: ds.expand_dims({name: [value]}, axis)
        )
        assert isinstance(new_tree, xr.DataTree)
        if path is not None:
            self.nodes[path] = new_tree[path]
        else:
            self.nodes = new_tree

    def _squeeze_dimension(self, dim_name: str, drop: bool = False, path: Optional[str] = None):
        new_tree = self.nodes.map_over_datasets(
            lambda ds: (
                ds.squeeze(dim_name, drop=drop)
                if dim_name in ds.dims and len(ds.coords[dim_name]) == 1
                else ds
            )
        )
        assert isinstance(new_tree, xr.DataTree)
        if path is not None:
            self.nodes[path] = new_tree[path]
        else:
            self.nodes = new_tree

    def __getattr__(self, attr):
        if attr in Action.REGISTRY:
            return RegisteredAction(
                attr, Action.REGISTRY[attr], self
            )  # When the attr is a registered action class
        raise AttributeError(f"{self.__class__.__name__} has no attribute {attr!r}")


class RegisteredAction:
    """Wrapper around registered actions"""

    def __init__(self, name: str, action: type[Action], root_action: Action) -> None:
        self._name = name
        self.action = action
        self.root_action = root_action

    def __getattr__(self, func):
        if not hasattr(self.action, func):
            raise AttributeError(f"{self.action.__name__} has no attribute {func!r}")

        def cast(origin_action: Action, new_action: type[Action]):
            return new_action(origin_action.nodes)

        @functools.wraps(getattr(self.action, func))
        def return_cast(*args, **kwargs):
            result = getattr(cast(self.root_action, self.action), func)(*args, **kwargs)
            return cast(result, self.root_action.__class__)

        return return_cast

    def __repr__(self):
        return f"Registered action: {self._name!r} at {self.action.__qualname__}"


def _batch_transform(
    action: Action, selection: dict, payload: PayloadFunc | Payload
) -> Action:
    selected = action.select(selection, drop=True)
    dim = list(selection.keys())[0]
    for npath, narray in nodetree_arrays(selected.nodes):
        if dim not in narray.dims:
            continue
        if narray.sizes[dim] == 1:
            selected._squeeze_dimension(dim, drop=True, path=npath)
        else:
            selected = selected.reduce(payload, dim=dim, path=npath)
    return selected


def _expand_transform(
    action: Action, index: int | Hashable, dim: int | str, backend_kwargs: dict = {}
) -> Action:
    ret = action.map(
        Payload(
            backends.take, [Node.input_name(0), index], {"dim": dim, **backend_kwargs}
        )
    )
    return ret


def _combine_nodes(
    action: Action,
    backend_method: str,
    dim: str,
    batch_size: int = 0,
    keep_dim: bool = False,
    path: Optional[str] = None,
    backend_kwargs: dict = {},
) -> Action:
    if action.nodes.sizes[dim] == 1:
        # no-op
        if not keep_dim:
            action._squeeze_dimension(dim, path=path)
        return action
    return action.reduce(
        Payload(getattr(backends, backend_method), kwargs=backend_kwargs),
        dim=dim,
        path=path,
        batch_size=batch_size,
        keep_dim=keep_dim,
    )


def from_source(
    payloads_list: (
        np.ndarray[Any, Any] | dict[str, np.ndarray[Any, Any]]
    ),  # values are Callables
    yields: Coord | None = None,
    dims: list | None = None,
    coords: dict | None = None,
    action=Action,
) -> Action:
    if not isinstance(payloads_list, dict):
        payloads_list = {"/": payloads_list}

    node_arrays = {}
    for nindex, (path, parray) in enumerate(payloads_list.items()):
        payloads = xr.DataArray(parray, dims=dims, coords=coords)
        nodes = xr.DataArray(
            np.empty(payloads.shape, dtype=object), dims=dims, coords=coords
        )
        it = np.nditer(payloads, flags=["multi_index", "refs_ok"])
        # Ensure all source nodes have a unique name
        node_names = set()
        for item in it:
            pit = item[()]  # type: ignore
            if not isinstance(pit, Payload):
                payload = Payload(pit)
            else:
                payload = pit
            name = payload.name()
            if name in node_names:
                name += str(it.multi_index)
            node_names.add(name)
            nodes[it.multi_index] = Node(
                payload, name=name, num_outputs=len(yields[1]) if yields else 1
            )
        node_arrays[path] = nodes

    return action(
        nodetree_from_dict(node_arrays),
        yields,
    )


def merge(*args, **kwargs) -> Action:
    """Merge node arrays in actions. If provided as keyword arguments, the key
    is used as the root path for the action's node arrays


    Return
    ------
    Action

    Raises
    ------
    ValueError if node paths overlap between actions
    """
    new_nodes = {}
    for action in args:
        for npath, narray in nodetree_arrays(action.nodes):
            if npath in new_nodes:
                raise ValueError(f"Cannot merge actions with overlapping node paths {npath}")
            new_nodes[npath] = narray

    for path, action in kwargs.items():
        for npath, narray in nodetree_arrays(xr.DataTree.from_dict({path: action.nodes})):
            if npath in new_nodes:
                raise ValueError(f"Cannot merge actions with overlapping node paths {npath}")
            new_nodes[npath] = narray

    action_type = args[0].__class__ if len(args) > 0 else list(kwargs.values())[0].__class__
    return action_type(nodetree_from_dict(new_nodes))

Action.register("default", Action)

__all__ = [
    "Action",
    "Payload",
    "Node",
    "from_source",
    "merge",
]
