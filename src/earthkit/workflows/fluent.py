# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import functools
from typing import Any, Callable, Hashable, Optional, Union
import numpy as np

from qubed import Qube

from . import backends
from ._qubed import expand_as_qube
from .graph import Graph, Output
from .nodeqube import create_task_instance, Coord, Node, NodeQube, Payload
from .metadata import NodeMetadata
from .utils import expand

class Action:
    REGISTRY: dict[str, type[Action]] = {}

    def __init__(self, nodeqube: NodeQube, yields: Optional[Coord] = None):
        self.nodeqube = nodeqube.expand_outputs(yields)

    def graph(self) -> Graph:
        """Creates graph from the nodes of the action.

        Return
        ------
        Graph instance constructed from list of nodes

        """
        sinks = set()
        for node in self.nodeqube.nodes.values():
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
            raise ValueError(f"Action class {obj} already has an attribute {name}, will not override")

        cls.REGISTRY[name] = obj

    @classmethod
    def flush_registry(cls):
        """Flush the registry of all registered actions"""
        cls.REGISTRY = {}

    def as_action(self, other) -> Action:
        """Parse action into another action class"""
        return other(self.nodeqube)

    def join(
        self,
        other_action: Action,
    ) -> Action:
        return type(self)(self.nodeqube.append(other_action.nodeqube))

    def transform(
        self,
        func: Callable[..., Action],
        params: list,
        dim: str | Coord,
    ) -> Action:
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
        nodeqube = NodeQube.empty()
        dim_values: list[int]
        if isinstance(dim, str):
            dim_name = dim
            dim_values = list(range(len(params)))
        else:
            dim_name = dim[0]
            dim_values = dim[1]

        for index, param in enumerate(params):
            new_res = func(self, *param)
            if dim_name not in new_res.nodeqube.dimensions():
                new_res.add_scalar_dimension(dim_name, dim_values[index], override=True)
            nodeqube.append(new_res.nodeqube)

        if nodeqube.is_empty():
            raise ValueError("No new actions generated from transform")
        return type(self)(nodeqube)

    def broadcast(
        self,
        other_action: Action,
        exclude: list[str] | None = None,
    ) -> Action:
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
        exclude = exclude or []
        existing_dimensions = self.nodeqube.dimensions()
        nodeqube = NodeQube.empty()
        for key in other_action.nodeqube.nodes.keys():
            datacube = NodeQube.datacube(key)
            select_criteria = {k: v for k, v in datacube.items() if k in existing_dimensions}
            unique_nodeqube = self.nodeqube.select(select_criteria)
            new_datacube = {k: v for k, v in datacube.items() if k not in exclude}
            new_key = NodeQube.key(new_datacube)
            new_node = Node(
                create_task_instance(
                    backends.method,
                    static_input_ps=["trivial"],
                ),
                unique_nodeqube.node(),
            )
            nodeqube.append(NodeQube(Qube.from_datacube(new_datacube), {new_key: new_node}))
        return type(self)(nodeqube)

    def expand(
        self,
        dim: str | Coord,
        internal_dim: int | str | Coord,
        dim_size: int | None = None,
        backend_kwargs: dict = {},
    ) -> Action:
        """Create new dimension in array of nodes of specified size by
        taking elements of internal data in each node. Indexing is taken along the specified axis
        dimension of internal data and graph execution will fail if
        dim_size exceeds the dimension size of this axis in the internal data.

        Parameters
        ----------
        dim: str or `Coord`, name of dimension or `Coord` specifying new dimension name and
        coordinate values
        internal_dim: int, str or DataArray, index or name of internal dimension to expand, or
        `Coord` specifying dimension name and list of selection criteria.
        dim_size: int | None, size of new dimension. If not given `internal_dim` must be `Coord`
        backend_kwargs: dict, kwargs for the underlying backend take method

        Return
        ------
        Action
        """
        if isinstance(internal_dim, (int, str)):
            if dim_size is None:
                raise TypeError("If `internal_dim` is str or int, then `dim_size` must be provided")
            params = [(i, internal_dim, backend_kwargs) for i in range(dim_size)]
        else:
            params = [(x, internal_dim[0], backend_kwargs) for x in internal_dim[1]]
            if isinstance(dim, str):
                dim = (dim, internal_dim[1])

        if not isinstance(dim, str) and len(params) != len(dim[1]):
            raise ValueError("Length of values in `dim` must match `dim_size` or length of values in `internal_dim`")
        return self.transform(_expand_transform, params, dim)

    expand_as_qube = expand_as_qube

    def map(
        self,
        payload: Payload | dict[str, Payload],
        yields: Coord | None = None,
        node_metadata: Optional[NodeMetadata] = None,
    ) -> Action:
        """Apply specified payload on all nodes. If argument is an dictionary of payloads,
        this must be the same size as the array of nodes and each node gets a
        unique payload from the array

        Parameters
        ----------
        payload: function or dictionary of functions
        yields: Coord | None, name and coords of dimension yielded by payload, if generator
        node_metadata: Optional[NodeMetadata] = None

        Return
        ------
        Action where nodes are a result of applying the same
        payload to all nodes, or in the case where payload is an dictionary,
        applying a different payload to each node

        Raises
        ------
        ValueError if the shape of the payload array does not match the shape of the
        array of nodes
        """
        new_nodes = {}
        if isinstance(payload, dict) and len(payload) != len(self.nodeqube.nodes):
            raise ValueError(
                f"Length of payload dict {len(payload)} does not match number of nodes {len(self.nodeqube.nodes)}"
            )
        for key, node in self.nodeqube.nodes.items():
            new_nodes[key] = Node(
                payload[key] if isinstance(payload, dict) else payload,
                node,
                num_outputs=len(yields[1]) if yields else 1,
                metadata=node_metadata,
            )
        return type(self)(NodeQube(self.nodeqube.qube, new_nodes), yields)

    def reduce(
        self,
        payload: Payload,
        dim: str,
        yields: Coord | None = None,
        batch_size: int = 0,
        keep_dim: bool = False,
        node_metadata: Optional[NodeMetadata] = None,
    ) -> Action:
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
        node_metadata: NodeMetadata, metadata to attach to the new nodes created by the reduction

        Return
        ------
        Action

        Raises
        ------
        ValueError if payload function is not batchable and batch_size is not 0
        """
        payload = create_task_instance(payload)
        if yields and batch_size != 0:
            raise ValueError("Can not batch the execution of a generator")
        payload_func = payload.definition.func
        if payload_func is not None and not getattr(payload_func, "batchable", False):
            raise ValueError(
                f"Function {payload_func} is not batchable, but batch_size {batch_size} is specified"  # type: ignore[union-attr]
            )
        
        nodeqube = NodeQube.empty()
        for datacube in self.nodeqube.datacubes():
            if dim not in datacube:
                nodeqube.append(self.nodeqube.select(datacube))
                continue
            if np.ndim(datacube[dim]) == 0:
                selection = self.nodeqube.select(datacube)
                if not keep_dim:
                    selection = selection.drop_scalar_dimension(dim)
                nodeqube.append(selection)
                continue
            if batch_size > 1 and batch_size < np.ndim(datacube[dim]):
                level = 0
                batched = self.select(datacube)
                while batch_size < len(batched.nodeqube.axes()[dim]):
                    lst = sorted(batched.nodeqube.axes()[dim])
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
                batched_nodeqube = batched.nodeqube
            else:
                batched_nodeqube = self.select(datacube).nodeqube

            for unique_datacube in batched_nodeqube.datacubes(expand=True):
                input_qube = Qube.from_datacube(unique_datacube)
                unique_datacube.pop(dim)
                new_key = NodeQube.key(unique_datacube)
                new_node = Node(
                    payload, 
                    list(batched_nodeqube.get_nodes(input_qube).values()),
                    num_outputs=len(yields[1]) if yields else 1, 
                    metadata=node_metadata,
                )
                nodeqube.append(NodeQube(Qube.from_datacube(unique_datacube), {new_key: new_node}))

            if keep_dim:
                coords = sorted(nodeqube.axes()[dim])
                nodeqube = nodeqube.add_scalar_dimension(dim, f"{coords[0]}-{coords[-1]}")
        return type(self)(nodeqube, yields)

    def flatten(
        self,
        new_dim: str,
        keep_dims: list[str] = [],
    ) -> Action:
        """Restructures node arrays by flattening arrays along all dims, except keep_dims

        Parameters
        ----------
        keep_dims: str, name of dimensions not to flatten
        new_dim: str, name of new dimension containing flattened dims

        Return
        ------
        Action
        """
        return type(self)(self.nodeqube.flatten(new_dim, keep_dims=set(keep_dims)))

    def select(
        self,
        criteria: dict | None = None,
        **kwargs,
    ) -> Action:
        """Create action contaning nodes match selection criteria

        Parameters
        ----------
        criteria: dict, key-value pairs specifying selection criteria

        Return
        ------
        Action
        """
        criteria = criteria or {}
        criteria.update(kwargs)
        return type(self)(self.nodeqube.select(criteria))

    sel = select

    def iselect(
        self,
        criteria: dict | None = None,
        **kwargs,
    ) -> Action:
        """Create action contaning nodes match index selection criteria

        Parameters
        ----------
        criteria: dict, key-value pairs specifying selection criteria
        drop: bool, drop coord variables in criteria if True
        path: str, path to select subset of nodes to operate on, if provided
        expand: bool, whether to expand the selection criteria into all possible combinations

        Return
        ------
        Action
        """
        criteria = criteria or {}
        criteria.update(kwargs)
        return type(self)(self.nodeqube.iselect(criteria))

    isel = iselect

    def concatenate(
        self,
        dim: str,
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        return _combine_nodes(self, "concat", dim, batch_size, keep_dim, node_metadata=node_metadata, backend_kwargs=backend_kwargs)

    def stack(
        self,
        dim: str,
        batch_size: int = 0,
        keep_dim: bool = False,
        axis: int = 0,
        backend_kwargs: dict = {},
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        return _combine_nodes(
            self,
            "stack",
            dim,
            batch_size,
            keep_dim,
            node_metadata=node_metadata,
            backend_kwargs={"axis": axis, **backend_kwargs},
        )

    def sum(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        return self.reduce(
            create_task_instance(
                backends.method,
                static_input_ps=["sum"],
                static_input_kw={"backend_kwargs": backend_kwargs},
            ),
            dim=dim,
            batch_size=batch_size,
            keep_dim=keep_dim,
            node_metadata=node_metadata,
        )

    def mean(
        self,
        dim: str,
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        size = len(self.nodeqube.axes()[dim])
        if batch_size <= 1 or batch_size >= size:
            action = self.reduce(
                create_task_instance(
                    backends.method,
                    static_input_ps=["mean"],
                    static_input_kw={"backend_kwargs": backend_kwargs},
                    node_metadata=node_metadata,
                ),
                dim=dim,
                keep_dim=keep_dim,
                node_metadata=node_metadata,
            )
        else:
            action = self.sum(
                dim=dim,
                batch_size=batch_size,
                keep_dim=keep_dim,
                backend_kwargs=backend_kwargs,
                node_metadata=node_metadata,
            ).divide(size, node_metadata=node_metadata)
        return action

    def std(
        self,
        dim: str,
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        size = len(self.nodeqube.axes()[dim])
        if batch_size <= 1 or batch_size >= size:
            action = self.reduce(
                create_task_instance(
                    backends.method,
                    static_input_ps=["std"],
                    static_input_kw={"backend_kwargs": backend_kwargs},
                    node_metadata=node_metadata,
                ),
                dim=dim,
            )

        else:
            mean_sq = self.mean(
                dim=dim,
                batch_size=batch_size,
                keep_dim=keep_dim,
                backend_kwargs=backend_kwargs,
                node_metadata=node_metadata,
            ).power(2, node_metadata=node_metadata)
            norm = (
                self.power(2, node_metadata=node_metadata)
                .sum(dim=dim, batch_size=batch_size, keep_dim=keep_dim, backend_kwargs=backend_kwargs, node_metadata=node_metadata)
                .divide(size, node_metadata=node_metadata)
            )
            action = norm.subtract(mean_sq, node_metadata=node_metadata).power(
                0.5, node_metadata=node_metadata
            )
        return action

    def max(
        self,
        dim: str,
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        return self.reduce(
            create_task_instance(
                backends.method,
                static_input_ps=["max"],
                static_input_kw={"backend_kwargs": backend_kwargs},
            ),
            dim=dim,
            batch_size=batch_size,
            keep_dim=keep_dim,
            node_metadata=node_metadata,
        )

    def min(
        self,
        dim: str,
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        return self.reduce(
            create_task_instance(
                backends.method,
                static_input_ps=["min"],
                static_input_kw={"backend_kwargs": backend_kwargs},
            ),
            dim=dim,
            batch_size=batch_size,
            keep_dim=keep_dim,
            node_metadata=node_metadata,
        )

    def prod(
        self,
        dim: str,
        batch_size: int = 0,
        keep_dim: bool = False,
        backend_kwargs: dict = {},
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        return self.reduce(
            create_task_instance(
                backends.method,
                static_input_ps=["prod"],
                static_input_kw={"backend_kwargs": backend_kwargs},
            ),
            dim=dim,
            batch_size=batch_size,
            keep_dim=keep_dim,
            node_metadata=node_metadata,
        )

    def __two_arg_method(
        self,
        method: str,
        other: Union[Action, float],
        node_metadata: NodeMetadata | None = None,
        backend_kwargs: Optional[dict] = None,
    ) -> Action:
        if isinstance(other, Action):
            other = other.add_scalar_dimension("**datatype**", 1)
            return (
                self.add_scalar_dimension("**datatype**", 0)
                .join(other)
                .reduce(
                    create_task_instance(
                        backends.method,
                        static_input_ps=[method],
                        static_input_kw={"backend_kwargs": backend_kwargs},
                        node_metadata=node_metadata,
                    ),
                    dim="**datatype**",
                )
            )
        return self.map(
            create_task_instance(
                backends.method,
                static_input_ps=[method, Node.Index(0), other],
                static_input_kw={"backend_kwargs": backend_kwargs},
            ),
            node_metadata=node_metadata,
        )

    def subtract(
        self,
        other: Union[Action, float],
        backend_kwargs: Optional[dict] = None,
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        return self.__two_arg_method("subtract", other, node_metadata=node_metadata, backend_kwargs=backend_kwargs)

    def divide(
        self,
        other: Union[Action, float],
        backend_kwargs: Optional[dict] = None,
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        return self.__two_arg_method("divide", other, node_metadata=node_metadata, backend_kwargs=backend_kwargs)

    def add(
        self,
        other: Union[Action, float],
        backend_kwargs: Optional[dict] = None,
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        return self.__two_arg_method("add", other, node_metadata=node_metadata, backend_kwargs=backend_kwargs)

    def multiply(
        self,
        other: Union[Action, float],
        backend_kwargs: Optional[dict] = None,
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        return self.__two_arg_method("multiply", other, node_metadata=node_metadata, backend_kwargs=backend_kwargs)

    def power(
        self,
        other: Union[Action, float],
        backend_kwargs: Optional[dict] = None,
        node_metadata: NodeMetadata | None = None,
    ) -> Action:
        return self.__two_arg_method("pow", other, node_metadata=node_metadata, backend_kwargs=backend_kwargs)

    def add_scalar_dimension(self, name: str, value: Any, override: bool = False):
        nodeqube = self.nodeqube
        if override:
            nodeqube = self.drop_scalar_dimension(name).nodeqube
        return type(self)(nodeqube.add_scalar_dimension(name, value))

    def drop_scalar_dimension(self, dim: str) -> Action:
        return type(self)(self.nodeqube.drop_scalar_dimension(dim))

    def __getattr__(self, attr):
        if attr in Action.REGISTRY:
            return RegisteredAction(attr, Action.REGISTRY[attr], self)  # When the attr is a registered action class
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


def _batch_transform(action: Action, selection: dict, payload: Payload) -> Action:
    selected = action.select(selection, drop=True)
    dim = list(selection.keys())[0]
    return selected.reduce(payload, dim=dim)


def _expand_transform(action: Action, index: int | Hashable, dim: int | str, backend_kwargs: dict = {}) -> Action:
    ret = action.map(
        create_task_instance(
            payload=backends.method,
            static_input_ps=["take", Node.Index(0), index],
            static_input_kw={"dim": dim, "backend_kwargs": backend_kwargs},
        ),
    )
    return ret


def _combine_nodes(
    action: Action,
    backend_method: str,
    dim: str,
    batch_size: int = 0,
    keep_dim: bool = False,
    backend_kwargs: Optional[dict] = None,
    node_metadata: NodeMetadata | None = None,
) -> Action:
    if backend_method not in ["stack", "concat"]:
        raise ValueError(f"Unknown method {backend_method} for combining nodes")
    return action.reduce(
        create_task_instance(
            payload=backends.method,
            static_input_ps=[backend_method],
            static_input_kw=backend_kwargs,
        ),
        dim=dim,
        batch_size=batch_size,
        keep_dim=keep_dim,
        node_metadata=node_metadata,
    )


def from_source(
    payloads: Payload | dict[str, Payload],
    yields: Coord | None = None,
    datacubes: Optional[list[dict]] = None,
    node_metadata: NodeMetadata | None = None,
    action=Action,
) -> Action:
    qube = Qube.empty()
    if datacubes is None:
        if not isinstance(payloads, dict):
            raise ValueError("If datacubes is None, payloads must be a dict of payloads.")
        for key in payloads.keys():
            datacube = NodeQube.datacube(str(key))
            qube.append_datacube(datacube)
    else:
        for datacube in datacubes:
            qube.append_datacube(datacube)

    nodes = {}
    for index, unique_datacube in enumerate(expand(qube.datacubes())):
        key = NodeQube.key(unique_datacube)
        nodes[key] = Node(payloads[key] if isinstance(payloads, dict) else payloads, num_outputs=len(yields[1]) if yields else 1, name=str(index), metadata=node_metadata)

    return action(
        NodeQube(qube, nodes),
        yields,
    )


def merge(*actions) -> Action:
    """Merge nodequbes in actions.


    Return
    ------
    Action
    """
    final_action = actions[0]
    for action in actions[1:]:
        final_action = final_action.join(action)
    return final_action

Action.register("default", Action)

__all__ = [
    "Action",
    "Payload",
    "Node",
    "from_source",
    "merge",
]
