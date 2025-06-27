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
from typing import Any, Callable, Hashable, Iterable, Optional, Sequence, TypeVar

import numpy as np
import xarray as xr

from . import backends
from .graph import Graph
from .graph import Node as BaseNode
from .graph import Output

ActionType = TypeVar("ActionType", bound="Action")


class Payload:
    """Class for detailing function, args and kwargs to be computing in a graph node"""

    def __init__(
        self,
        func: Callable,
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
        if hasattr(self.func, "__name__"):
            return self.func.__name__
        return ""

    def __str__(self) -> str:
        return f"{self.name()}{self.args}{self.kwargs}:{self.metadata}"

    def __eq__(self, other: Payload) -> bool:
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

F = TypeVar("F", bound=Callable[..., Any])


def capture_payload_metadata(func: F) -> F:
    """Wrap a function which returns a new action and insert
    given `payload_metadata`
    """

    @functools.wraps(func)
    def decorator(*args, **kwargs):
        metadata = kwargs.pop("payload_metadata", {})
        result = func(*args, **kwargs)

        if isinstance(result, Action):
            for node in np.atleast_1d(result.nodes.values).flatten():
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
        payload: Callable | Payload,
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

    REGISTRY: dict[str, Action] = {}

    def __init__(self, nodes: xr.DataArray, yields: Optional[Coord] = None):
        if yields:
            ydim, ycoords = yields
            nodes = xr.apply_ufunc(
                lambda x: np.asarray([x.get_output(out) for out in x.outputs]),
                nodes,
                output_core_dims=[[ydim]],
                vectorize=True,
            )
            nodes.coords[ydim] = ycoords
        assert not np.any(nodes.isnull()), "Array of nodes can not contain NaNs"
        self.nodes = nodes

    def graph(self) -> Graph:
        """Creates graph from the nodes of the action.

        Return
        ------
        Graph instance constructed from list of nodes

        """
        nodes = list(self.nodes.data.flatten())
        sinks = set()
        for node in nodes:
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

        cls.REGISTRY[name] = obj  # type: ignore

    @classmethod
    def flush_registry(cls):
        """Flush the registry of all registered actions"""
        cls.REGISTRY = {}

    def as_action(self, other: ActionType) -> ActionType:
        """Parse action into another action class"""
        return other(self.nodes)

    def __getattr__(self, attr):
        if attr in Action.REGISTRY:
            return RegisteredAction(
                attr, Action.REGISTRY[attr], self
            )  # When the attr is a registered action class
        raise AttributeError(f"{self.__class__.__name__} has no attribute {attr!r}")

    def join(
        self,
        other_action: "Action",
        dim: str | Coord,
        match_coord_values: bool = False,
    ) -> "Action":
        if match_coord_values:
            for coord, values in self.nodes.coords.items():
                if coord in other_action.nodes.coords:
                    other_action.nodes = other_action.nodes.assign_coords(
                        **{str(coord): values}
                    )
        new_nodes = xr.concat(
            [self.nodes, other_action.nodes],
            dim if isinstance(dim, str) else xr.DataArray(dim[1], name=dim[0]),
            combine_attrs="no_conflicts",
            coords="minimal",
            join="exact",
        )
        ret = type(self)(new_nodes)
        return ret

    def transform(
        self, func: Callable, params: list, dim: str | Coord, axis: int = 0
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
            new_res = func(self, *param)
            if dim_name not in new_res.nodes.coords:
                new_res._add_dimension(dim_name, dim_values[index], axis)
            if res is None:
                res = new_res
            else:
                res = res.join(new_res, dim_name)

        if not res:
            raise ValueError
        # Remove expanded dimension if only a single element
        res._squeeze_dimension(dim_name)
        return res

    def broadcast(
        self, other_action: "Action", exclude: list[str] | None = None
    ) -> "Action":
        """Broadcast nodes against nodes in other_action

        Parameters
        ----------
        other_action: Action containing nodes to broadcast against
        exclude: List of str, dimension names to exclude from broadcasting

        Return
        ------
        Action
        """
        # Ensure coordinates in existing dimensions match, otherwise obtain NaNs
        for key, values in other_action.nodes.coords.items():
            if key in self.nodes.coords and (exclude is None or key not in exclude):
                assert np.all(
                    values.data == self.nodes.coords[key].data
                ), f"Existing coordinates must match for broadcast. Found mismatch in {key}!"

        broadcasted_nodes = self.nodes.broadcast_like(
            other_action.nodes, exclude=exclude
        )
        new_nodes = np.empty(broadcasted_nodes.shape, dtype=object)
        it = np.nditer(
            self.nodes.transpose(*broadcasted_nodes.dims, missing_dims="ignore"),
            flags=["multi_index", "refs_ok"],
        )
        for node in it:
            new_nodes[it.multi_index] = Node(Payload(backends.trivial), node[()])  # type: ignore

        new_nodes_xa = xr.DataArray(
            new_nodes,
            coords=broadcasted_nodes.coords,
            dims=broadcasted_nodes.dims,
            attrs=self.nodes.attrs,
        )
        return type(self)(new_nodes_xa)

    def expand(
        self,
        dim: str | Coord,
        internal_dim: int | str | Coord,
        dim_size: int | None = None,
        axis: int = 0,
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
        return self.transform(_expand_transform, params, dim, axis=axis)

    def map(
        self,
        payload: Callable | Payload | np.ndarray[Any, Any],
        yields: Coord | None = None,
    ) -> "Action":
        """Apply specified payload on all nodes. If argument is an array of payloads,
        this must be the same size as the array of nodes and each node gets a
        unique payload from the array

        Parameters
        ----------
        payload: function or array of functions
        yields: Coord | None, name and coords of dimension yielded by payload, if generator

        Return
        ------
        MultiAction where nodes are a result of applying the same
        payload to all nodes, or in the case where payload is an array,
        applying a different payload to each node

        Raises
        ------
        AssertionError if the shape of the payload array does not match the shape of the
        array of nodes
        """
        # NOTE this method is really not mypy friendly, just ignore everything
        if not isinstance(payload, Callable | Payload):  # type: ignore
            payload = np.asarray(payload)
            assert payload.shape == self.nodes.shape, (
                f"For unique payloads for each node, payload shape {payload.shape}"
                f"must match node array shape {self.nodes.shape}"
            )

        # Applies operation to every node, keeping node array structure
        new_nodes = np.empty(self.nodes.shape, dtype=object)

        it = np.nditer(self.nodes, flags=["multi_index", "refs_ok"])
        node_payload = payload
        for node in it:
            if not isinstance(payload, Callable | Payload):  # type: ignore
                node_payload = payload[it.multi_index]  # type: ignore
            new_nodes[it.multi_index] = Node(
                node_payload,  # type: ignore
                node[()],  # type: ignore
                num_outputs=len(yields[1]) if yields else 1,
            )

        new_nodes_xr = xr.DataArray(
            new_nodes,
            coords=self.nodes.coords,
            dims=self.nodes.dims,
            attrs=self.nodes.attrs,
        )

        return type(self)(new_nodes_xr, yields)

    def reduce(
        self,
        payload: Callable | Payload,
        yields: Coord | None = None,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
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

        Return
        ------
        Action

        Raises
        ------
        ValueError if payload function is not batchable and batch_size is not 0
        """

        if len(dim) == 0:
            dim = str(self.nodes.dims[0])

        batched = self
        level = 0
        if not isinstance(payload, Payload):
            payload = Payload(payload)
        if yields and batch_size != 0:
            raise ValueError("Can not batch the execution of a generator")
        if batch_size > 1 and batch_size < batched.nodes.sizes[dim]:
            if not getattr(payload.func, "batchable", False):
                raise ValueError(
                    f"Function {payload.func.__name__} is not batchable, but batch_size {batch_size} is specified"
                )

            while batch_size < batched.nodes.sizes[dim]:
                lst = batched.nodes.coords[dim].data
                batched = batched.transform(
                    _batch_transform,
                    [
                        ({dim: lst[i : i + batch_size]}, payload)  # noqa: E203
                        for i in range(0, len(lst), batch_size)
                    ],
                    f"batch.{level}.{dim}",
                )
                dim = f"batch.{level}.{dim}"
                level += 1

        new_dims = [x for x in batched.nodes.dims if x != dim]
        transposed_nodes = batched.nodes.transpose(dim, *new_dims)
        new_nodes = np.empty(transposed_nodes.shape[1:], dtype=object)
        it = np.nditer(new_nodes, flags=["multi_index", "refs_ok"])
        for _ in it:
            inputs = transposed_nodes[(slice(None, None, 1), *it.multi_index)].data
            new_nodes[it.multi_index] = Node(
                payload, inputs, num_outputs=len(yields[1]) if yields else 1
            )

        new_coords = {key: batched.nodes.coords[key] for key in new_dims}
        # Propagate scalar coords
        new_coords.update(
            {
                k: v
                for k, v in batched.nodes.coords.items()
                if k not in batched.nodes.dims
            }
        )
        nodes = xr.DataArray(
            new_nodes,
            coords=new_coords,
            dims=new_dims,
            attrs=batched.nodes.attrs,
        )
        result = type(batched)(nodes, yields)
        if keep_dim:
            axis = self.nodes.dims.index(dim)
            result._add_dimension(
                dim,
                f"{self.nodes.coords[dim][0]}-{self.nodes.coords[dim][-1]}",
                axis,
            )
        return result

    @capture_payload_metadata
    def flatten(
        self, dim: str = "", axis: int = 0, backend_kwargs: dict = {}
    ) -> "Action":
        """Flattens the array of nodes along specified dimension by creating new
        nodes from stacking internal data of nodes along that dimension.

        Parameters
        ----------
        dim: str, name of dimension to flatten along
        axis: int, axis of new dimension in internal data
        kwargs: kwargs for the underlying array module stack method
        Return
        ------
        Action
        """
        return self.reduce(
            Payload(backends.stack, kwargs={"axis": axis, **backend_kwargs}), dim=dim
        )

    def _validate_criteria(self, criteria: dict):
        keys = list(criteria.keys())
        for key in keys:
            if key not in self.nodes.dims:
                if self.nodes.coords.get(key, None) == criteria[key]:
                    criteria.pop(key)
                else:
                    raise NotImplementedError(
                        f"Unknown dim in criteria {criteria}. Existing dimensions {self.nodes.dims}"
                        + f"and coords {self.nodes.coords}"
                    )
        return criteria

    @capture_payload_metadata
    def select(
        self, criteria: dict | None = None, drop: bool = False, **kwargs
    ) -> "Action":
        """Create action contaning nodes match selection criteria

        Parameters
        ----------
        criteria: dict, key-value pairs specifying selection criteria
        drop: bool, drop coord variables in criteria if True

        Return
        ------
        Action
        """
        crit: dict = criteria or {}
        crit.update(kwargs)

        crit = self._validate_criteria(crit)

        if len(crit) == 0:
            return self
        selected_nodes = self.nodes.sel(**crit, drop=drop)
        return type(self)(selected_nodes)

    sel = select

    @capture_payload_metadata
    def iselect(
        self, criteria: dict | None = None, drop: bool = False, **kwargs
    ) -> "Action":
        """Create action contaning nodes match index selection criteria

        Parameters
        ----------
        criteria: dict, key-value pairs specifying selection criteria
        drop: bool, drop coord variables in criteria if True

        Return
        ------
        Action
        """
        crit: dict = criteria or {}
        crit.update(kwargs)

        crit = self._validate_criteria(crit)

        if len(crit) == 0:
            return self
        selected_nodes = self.nodes.isel(**crit, drop=drop)
        return type(self)(selected_nodes)

    isel = iselect

    @capture_payload_metadata
    def concatenate(
        self,
        dim: str,
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
    ) -> "Action":
        return _combine_nodes(self, "concat", dim, batch_size, keep_dim, backend_kwargs)

    @capture_payload_metadata
    def stack(
        self,
        dim: str,
        batch_size: int = 0,
        keep_dim: bool = False,
        axis: int = 0,
        backend_kwargs: dict = {},
    ) -> "Action":
        return _combine_nodes(
            self,
            "stack",
            dim,
            batch_size,
            keep_dim,
            backend_kwargs={"axis": axis, **backend_kwargs},
        )

    @capture_payload_metadata
    def sum(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.reduce(
            Payload(backends.sum, kwargs=backend_kwargs),
            dim=dim,
            batch_size=batch_size,
            keep_dim=keep_dim,
        )

    @capture_payload_metadata
    def mean(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
    ) -> "Action":
        if len(dim) == 0:
            dim = str(self.nodes.dims[0])

        if batch_size <= 1 or batch_size >= self.nodes.sizes[dim]:
            return self.reduce(
                Payload(backends.mean, kwargs=backend_kwargs),
                dim=dim,
                keep_dim=keep_dim,
            )

        return self.sum(
            dim=dim, batch_size=batch_size, keep_dim=keep_dim, **backend_kwargs
        ).divide(self.nodes.sizes[dim])

    @capture_payload_metadata
    def std(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
    ) -> "Action":
        if len(dim) == 0:
            dim = str(self.nodes.dims[0])

        if batch_size <= 1 or batch_size >= self.nodes.sizes[dim]:
            return self.reduce(Payload(backends.std, kwargs=backend_kwargs), dim=dim)

        mean_sq = self.mean(
            dim=dim, batch_size=batch_size, keep_dim=keep_dim, **backend_kwargs
        ).power(2)
        norm = (
            self.power(2)
            .sum(dim=dim, batch_size=batch_size, keep_dim=keep_dim, **backend_kwargs)
            .divide(self.nodes.sizes[dim])
        )
        return norm.subtract(mean_sq).power(0.5)

    @capture_payload_metadata
    def max(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.reduce(
            Payload(backends.max, kwargs=backend_kwargs),
            dim=dim,
            batch_size=batch_size,
            keep_dim=keep_dim,
        )

    @capture_payload_metadata
    def min(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.reduce(
            Payload(backends.min, kwargs=backend_kwargs),
            dim=dim,
            batch_size=batch_size,
            keep_dim=keep_dim,
        )

    @capture_payload_metadata
    def prod(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
    ) -> "Action":
        return self.reduce(
            Payload(backends.prod, kwargs=backend_kwargs),
            dim=dim,
            batch_size=batch_size,
            keep_dim=keep_dim,
        )

    def __two_arg_method(
        self, method: Callable, other: "Action | float", kwargs
    ) -> "Action":
        if isinstance(other, Action):
            return self.join(other, "**datatype**", match_coord_values=True).reduce(
                Payload(method, kwargs=kwargs)
            )
        return self.map(
            Payload(method, args=(Node.input_name(0), other), kwargs=kwargs)
        )

    @capture_payload_metadata
    def subtract(self, other: "Action | float", backend_kwargs: dict = {}) -> "Action":
        return self.__two_arg_method(backends.subtract, other, backend_kwargs)

    @capture_payload_metadata
    def divide(self, other: "Action | float", backend_kwargs: dict = {}) -> "Action":
        return self.__two_arg_method(backends.divide, other, backend_kwargs)

    @capture_payload_metadata
    def add(self, other: "Action | float", backend_kwargs: dict = {}) -> "Action":
        return self.__two_arg_method(backends.add, other, backend_kwargs)

    @capture_payload_metadata
    def multiply(self, other: "Action | float", backend_kwargs: dict = {}) -> "Action":
        return self.__two_arg_method(backends.multiply, other, backend_kwargs)

    @capture_payload_metadata
    def power(self, other: "Action | float", backend_kwargs: dict = {}) -> "Action":
        return self.__two_arg_method(backends.pow, other, backend_kwargs)

    def add_attributes(self, attrs: dict):
        self.nodes.attrs.update(attrs)

    def _add_dimension(self, name: str, value: Any, axis: int = 0):
        self.nodes = self.nodes.expand_dims({name: [value]}, axis)

    def _squeeze_dimension(self, dim_name: str, drop: bool = False):
        if dim_name in self.nodes.coords and len(self.nodes.coords[dim_name]) == 1:
            self.nodes = self.nodes.squeeze(dim_name, drop=drop)


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
    action: Action, selection: dict, payload: Callable | Payload
) -> Action:
    selected = action.select(selection, drop=True)
    dim = list(selection.keys())[0]
    if dim not in selected.nodes.dims:
        return selected
    if selected.nodes.sizes[dim] == 1:
        selected._squeeze_dimension(dim, drop=True)
        return selected

    reduced = selected.reduce(payload, dim=dim)
    return reduced


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
    backend_kwargs: dict = {},
) -> Action:
    if action.nodes.sizes[dim] == 1:
        # no-op
        if not keep_dim:
            action._squeeze_dimension(dim)
        return action
    return action.reduce(
        Payload(getattr(backends, backend_method), kwargs=backend_kwargs),
        dim=dim,
        batch_size=batch_size,
        keep_dim=keep_dim,
    )


def from_source(
    payloads_list: np.ndarray[Any, Any],  # values are Callables
    yields: Coord | None = None,
    dims: list | None = None,
    coords: dict | None = None,
    action=Action,
) -> Action:
    payloads = xr.DataArray(payloads_list, dims=dims, coords=coords)
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

    return action(nodes, yields)


Action.register("default", Action)

__all__ = [
    "Action",
    "Payload",
    "Node",
    "from_source",
]
