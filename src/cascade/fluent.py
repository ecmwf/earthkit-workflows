import numpy as np
import xarray as xr
import functools
import hashlib
from dataclasses import dataclass, field

from .graph import Graph
from .graph import Node as BaseNode
from .backends.arrayapi import ArrayApiBackend


@dataclass
class Payload:
    """
    Class for detailing function, args and kwargs to be computing in a graph node
    """

    func: callable
    args: list = None
    kwargs: dict = field(default_factory=dict)

    def has_args(self) -> bool:
        """
        Return
        ------
        bool, specifying if arguments have been set
        """
        return self.args is not None

    def set_args(self, args: list):
        """
        Parameters
        ----------
        args: list, arguments to pass to the function

        Raises
        ------
        AssertionError if arguments already exist
        """
        assert not self.has_args()
        self.args = args

    def to_tuple(self) -> tuple:
        """
        Return
        ------
        tuple, containing function, arguments and kwargs
        """
        assert self.has_args()
        return (self.func, self.args, self.kwargs)

    def name(self) -> str:
        """
        Return
        ------
        str, name of function, or if a partial function, the function name and partial
        arguments
        """
        if hasattr(self.func, "__name__"):
            return self.func.__name__
        if isinstance(self.func, functools.partial):
            return f"{self.func.func.__name__}@{'/'.join(map(str, self.func.args))}"
        return ""

    def __str__(self) -> str:
        return f"{self.name()}{self.args}{self.kwargs}"

    def copy(self) -> "Payload":
        return Payload(self.func, self.args, self.kwargs)


def custom_hash(string: str) -> str:
    ret = hashlib.sha256()
    ret.update(string.encode())
    return ret.hexdigest()


class Node(BaseNode):
    def __init__(
        self,
        payload: Payload,
        inputs: BaseNode | tuple[BaseNode] = (),
        name: str = None,
    ):
        payload = payload.copy()
        if isinstance(inputs, BaseNode):
            inputs = [inputs]
        # If payload doesn't have inputs, assume inputs to the function are the inputs
        # to the node, in the order provided
        if not payload.has_args():
            payload.set_args([self.input_name(x) for x in range(len(inputs))])
        else:
            # All inputs into Node should also feature in payload - no dangling inputs
            assert np.all(
                [
                    x in payload.args
                    for x in [self.input_name(x) for x in range(len(inputs))]
                ]
            ), f"Payload {payload} does not use all node inputs {len(inputs)}"

        if name is None:
            name = payload.name()
        name += f":{custom_hash(f'{payload}{[x.name for x in inputs]}')}"

        super().__init__(
            name,
            payload=payload.to_tuple(),
            **{self.input_name(x): node for x, node in enumerate(inputs)},
        )
        self.attributes = {}

    @staticmethod
    def input_name(index: int):
        return f"input{index}"


class Action:
    def __init__(self, previous: "Action", nodes: xr.DataArray, backend):
        self.previous = previous
        assert not np.any(nodes.isnull()), "Array of nodes can not contain NaNs"
        self.nodes = nodes
        self.sinks = [] if previous is None else previous.sinks.copy()
        self.backend = backend

    def graph(self) -> Graph:
        """
        Creates graph from sinks. If list of sinks is empty then uses the
        list of nodes.

        Return
        ------
        Graph instance constructed from list of sinks, or if empty, list
        of nodes

        """
        sinks = self.sinks
        if len(sinks) == 0:
            sinks = list(self.nodes.data.flatten())

        for index in range(len(sinks)):
            sinks[index] = sinks[index].copy()
            sinks[index].outputs = []  # Ensures they are recognised as sinks
        return Graph(sinks)

    def join(
        self,
        other_action: "Action",
        dim: str | xr.DataArray,
        match_coord_values: bool = False,
    ) -> "Action":
        if match_coord_values:
            for coord, values in self.nodes.coords.items():
                if coord in other_action.nodes.coords:
                    other_action.nodes = other_action.nodes.assign_coords(
                        **{coord: values}
                    )
        new_nodes = xr.concat(
            [self.nodes, other_action.nodes],
            dim,
            combine_attrs="no_conflicts",
            coords="minimal",
            join="exact",
        )
        if hasattr(self, "to_multi"):
            ret = self.to_multi(new_nodes)
        else:
            ret = type(self)(self, new_nodes, self.backend)
        ret.sinks = self.sinks + other_action.sinks
        return ret

    def transform(
        self, func: callable, params: list, dim: str | xr.DataArray
    ) -> "Action":
        """
        Create new nodes by applying function on action with different
        parameters. The result actions from applying function are joined
        along the specified dimension.

        Parameters
        ----------
        func: function with signature func(Action, *args) -> Action
        params: list, containing different arguments to pass into func
        for generating new nodes
        dim: str or DataArray, name of dimension to join actions or xr.DataArray specifying new dimension name and
        coordinate values

        Return
        ------
        SingleAction or MultiAction
        """
        res = None
        if isinstance(dim, str):
            dim_name = dim
            dim_values = list(range(len(params)))
        else:
            dim_name = dim.name
            dim_values = dim.values

        for index, param in enumerate(params):
            new_res = func(self, *param)
            if dim_name not in new_res.nodes.coords:
                new_res._add_dimension(dim_name, dim_values[index])
            if res is None:
                res = new_res
            else:
                res = res.join(new_res, dim_name)

        # Remove expanded dimension if only a single element
        res._squeeze_dimension(dim_name)
        return res

    def broadcast(self, other_action: "Action", exclude: list[str] = None) -> "Action":
        """
        Broadcast nodes against nodes in other_action

        Parameters
        ----------
        other_action: Action containing nodes to broadcast against
        exclude: List of str, dimension names to exclude from broadcasting

        Return
        ------
        MultiAction
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
            new_nodes[it.multi_index] = Node(Payload(self.backend.trivial), node[()])

        new_nodes = xr.DataArray(
            new_nodes,
            coords=broadcasted_nodes.coords,
            dims=broadcasted_nodes.dims,
            attrs=self.nodes.attrs,
        )
        if hasattr(self, "to_multi"):
            return self.to_multi(new_nodes)
        return type(self)(self, new_nodes, self.backend)

    def expand(
        self, dim: str, dim_size: int, axis: int = 0, new_axis: int = 0
    ) -> "Action":
        """
        Create new dimension in array of nodes of specified size by
        taking elements of internal data in each node. Indexing is taken along the specified axis
        dimension of internal data and graph execution will fail if
        dim_size exceeds the dimension size of this axis in the internal data.

        Parameters
        ----------
        dim: str, name of new dimension
        dim_size: int, size of new dimension
        axis: int, axis to take values from in internal data of node
        new_axis: int, position to insert new dimension


        Return
        ------
        MultiAction
        """
        params = [(x, dim, axis, new_axis) for x in range(dim_size)]
        return self.transform(_expand_transform, params, dim)

    def __two_arg_method(
        self, method: callable, other, **method_kwargs
    ) -> "SingleAction | MultiAction":
        if isinstance(other, Action):
            return self.join(other, "**datatype**", match_coord_values=True).reduce(
                Payload(method, kwargs=method_kwargs)
            )
        return self.map(
            Payload(method, args=(Node.input_name(0), other), kwargs=method_kwargs)
        )

    def subtract(self, other, **method_kwargs) -> "SingleAction | MultiAction":
        return self.__two_arg_method(self.backend.subtract, other, **method_kwargs)

    def divide(self, other, **method_kwargs) -> "SingleAction | MultiAction":
        return self.__two_arg_method(self.backend.divide, other, **method_kwargs)

    def add(self, other, **method_kwargs) -> "SingleAction | MultiAction":
        return self.__two_arg_method(self.backend.add, other, **method_kwargs)

    def multiply(self, other, **method_kwargs) -> "SingleAction | MultiAction":
        return self.__two_arg_method(self.backend.multiply, other, **method_kwargs)

    def power(self, other, **method_kwargs) -> "SingleAction | MultiAction":
        return self.__two_arg_method(self.backend.pow, other, **method_kwargs)

    def add_attributes(self, attrs: dict):
        self.nodes.attrs.update(attrs)

    def _add_dimension(self, name: str, value: float, axis: int = 0):
        self.nodes = self.nodes.expand_dims({name: [value]}, axis)

    def _squeeze_dimension(self, dim_name: str, drop: bool = False):
        if dim_name in self.nodes.coords and len(self.nodes.coords[dim_name]) == 1:
            self.nodes = self.nodes.squeeze(dim_name, drop=drop)


class SingleAction(Action):
    def __init__(self, previous: Action, node: xr.DataArray, backend):
        assert node.size == 1, f"Expected node size 1, got {node.size}"
        super().__init__(previous, node, backend)

    def to_multi(self, nodes: xr.DataArray) -> "MultiAction":
        """
        Conversion from SingleAction to MultiAction

        Parameters
        ----------
        nodes: xr.DataArray[Node], new nodes for constructing MultiAction

        Return
        ------
        MultiAction
        """
        return MultiAction(self, nodes, self.backend)

    def map(self, payload: Payload) -> "SingleAction":
        """
        Create new action from applying payload to node

        Parameters
        ----------
        payload: Payload

        Return
        ------
        SingleAction
        """
        return type(self).from_payload(self, payload, self.backend)

    def node(self) -> Node:
        return self.nodes.data[()]

    @classmethod
    def from_payload(
        cls, previous: Action, payload: Payload, backend
    ) -> "SingleAction":
        """
        Factory for SingleAction from previous action and payload

        Parameters
        ----------
        previous: Action that precedes action to be constructed
        payload: Payload for node in new action

        Return
        ------
        SingleAction
        """
        if previous is None:
            node = xr.DataArray(Node(payload))
        else:
            node = xr.DataArray(
                Node(
                    payload,
                    previous.nodes.data.flatten(),
                ),
                coords={
                    k: v
                    for k, v in previous.nodes.coords.items()
                    if k not in previous.nodes.dims
                },
                attrs=previous.nodes.attrs,
            )
        return cls(previous, node, backend)


class MultiAction(Action):
    def __init__(self, previous: xr.DataArray, nodes: xr.DataArray, backend):
        super().__init__(previous, nodes, backend)

    def to_single(self, payload_or_node: Payload | xr.DataArray) -> SingleAction:
        """
        Conversion from MultiAction to SingleAction

        Parameters
        ----------
        payload_or_node: Payload or xr.DataArray[Node] for constructing
        SingleAction

        Return
        ------
        SingleAction
        """
        if isinstance(payload_or_node, Payload):
            return SingleAction.from_payload(self, payload_or_node, self.backend)
        return SingleAction(self, payload_or_node, self.backend)

    def node(self, criteria: dict) -> Node | np.ndarray[Node]:
        """
        Get nodes matching selection criteria from action

        Parameters
        ----------
        criteria: dict, key-value pairs specifying selection criteria

        Return
        ------
        Node or np.ndarray[Node]
        """
        return self.nodes.sel(**criteria, drop=True).data[()]

    def map(self, payload: Payload | np.ndarray[Payload]) -> "MultiAction":
        """
        Apply specified payload on all nodes. If argument is an array of payloads,
        this must be the same size as the array of nodes and each node gets a
        unique payload from the array

        Parameters
        ----------
        payload: Payload or array of Payloads

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
        if not isinstance(payload, Payload):
            payload = np.asarray(payload)
            assert (
                payload.shape == self.nodes.shape
            ), f"For unique payloads for each node, payload shape {payload.shape} must match node array shape {self.nodes.shape}"

        # Applies operation to every node, keeping node array structure
        new_nodes = np.empty(self.nodes.shape, dtype=object)
        it = np.nditer(self.nodes, flags=["multi_index", "refs_ok"])
        node_payload = payload
        for node in it:
            if not isinstance(payload, Payload):
                node_payload = payload[it.multi_index]
            new_nodes[it.multi_index] = Node(node_payload, node[()])
        return type(self)(
            self,
            xr.DataArray(
                new_nodes,
                coords=self.nodes.coords,
                dims=self.nodes.dims,
                attrs=self.nodes.attrs,
            ),
            self.backend,
        )

    def reduce(
        self, payload: Payload, dim: str = "", batch_size: int = 0
    ) -> "SingleAction | MultiAction":
        """
        Reduction operation across the named dimension using the provided
        function in the payload. If batch_size > 1 and less than the size
        of the named dimension, the reduction will be computed first in
        batches and then aggregated, otherwise no batching will be performed.

        Parameters
        ----------
        payload: Payload, payload specifying function for performing the reduction
        dim: str, name of dimension along which to reduce
        batch_size: int, size of batches to split reduction into. If 0,
        computation is not batched

        Return
        ------
        SingleAction or MultiAction

        Raises
        ------
        ValueError if payload function is not batchable and batch_size is not 0
        """

        if len(dim) == 0:
            dim = self.nodes.dims[0]

        batched = self
        level = 0
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
                        ({dim: lst[i : i + batch_size]}, payload)
                        for i in range(0, len(lst), batch_size)
                    ],
                    f"batch.{level}.{dim}",
                )
                dim = f"batch.{level}.{dim}"
                level += 1

        if batched.nodes.ndim == 1:
            return batched.to_single(payload)

        new_dims = [x for x in batched.nodes.dims if x != dim]
        transposed_nodes = batched.nodes.transpose(dim, *new_dims)
        new_nodes = np.empty(transposed_nodes.shape[1:], dtype=object)
        it = np.nditer(new_nodes, flags=["multi_index", "refs_ok"])
        for _ in it:
            inputs = transposed_nodes[(slice(None, None, 1), *it.multi_index)].data
            new_nodes[it.multi_index] = Node(payload, inputs)

        new_coords = {key: batched.nodes.coords[key] for key in new_dims}
        # Propagate scalar coords
        new_coords.update(
            {
                k: v
                for k, v in batched.nodes.coords.items()
                if k not in batched.nodes.dims
            }
        )
        return type(batched)(
            batched,
            xr.DataArray(
                new_nodes,
                coords=new_coords,
                dims=new_dims,
                attrs=batched.nodes.attrs,
            ),
            batched.backend,
        )

    def flatten(
        self, dim: str = "", axis: int = 0, **method_kwargs
    ) -> "SingleAction | MultiAction":
        """
        Flattens the array of nodes along specified dimension by creating new
        nodes from stacking internal data of nodes along that dimension.

        Parameters
        ----------
        dim: str, name of dimension to flatten along
        axis: int, axis of new dimension in internal data
        method_kwargs: dict, kwargs for the underlying array module stack method
        Return
        ------
        SingleAction or MultiAction
        """
        return self.reduce(
            Payload(self.backend.stack, kwargs={"axis": axis, **method_kwargs}), dim
        )

    def select(
        self, criteria: dict, drop: bool = False
    ) -> "SingleAction | MultiAction":
        """
        Create action contaning nodes match selection criteria

        Parameters
        ----------
        criteria: dict, key-value pairs specifying selection criteria
        drop: bool, drop coord variables in criteria if True

        Return
        ------
        SingleAction or MultiAction
        """
        if any([key not in self.nodes.dims for key in criteria.keys()]):
            raise NotImplementedError(
                f"Unknown coordinate in criteria {criteria}. Existing dimensions {self.nodes.dims}"
            )

        selected_nodes = self.nodes.sel(**criteria, drop=drop)
        if selected_nodes.size == 1:
            return self.to_single(selected_nodes)
        return type(self)(self, selected_nodes, self.backend)

    def concatenate(
        self, dim: str, batch_size: int = 0, **method_kwargs
    ) -> "SingleAction | MultiAction":
        return self.reduce(
            Payload(self.backend.concat, kwargs=method_kwargs), dim, batch_size
        )

    def sum(
        self, dim: str = "", batch_size: int = 0, **method_kwargs
    ) -> "SingleAction | MultiAction":
        return self.reduce(
            Payload(self.backend.sum, kwargs=method_kwargs), dim, batch_size
        )

    def mean(
        self, dim: str = "", batch_size: int = 0, **method_kwargs
    ) -> "SingleAction | MultiAction":
        if len(dim) == 0:
            dim = self.nodes.dims[0]

        if batch_size <= 1 or batch_size >= self.nodes.sizes[dim]:
            return self.reduce(Payload(self.backend.mean, kwargs=method_kwargs), dim)

        return self.sum(dim, batch_size, **method_kwargs).divide(self.nodes.sizes[dim])

    def std(
        self, dim: str = "", batch_size: int = 0, **method_kwargs
    ) -> "SingleAction | MultiAction":
        if len(dim) == 0:
            dim = self.nodes.dims[0]

        if batch_size <= 1 or batch_size >= self.nodes.sizes[dim]:
            return self.reduce(Payload(self.backend.std, kwargs=method_kwargs), dim)

        mean_sq = self.mean(dim, batch_size, **method_kwargs).power(2)
        norm = self.power(2).sum(dim, batch_size).divide(self.nodes.sizes[dim])
        return norm.subtract(mean_sq).power(0.5)

    def max(
        self, dim: str = "", batch_size: int = 0, **method_kwargs
    ) -> "SingleAction | MultiAction":
        return self.reduce(
            Payload(self.backend.max, kwargs=method_kwargs), dim, batch_size
        )

    def min(
        self, dim: str = "", batch_size: int = 0, **method_kwargs
    ) -> "SingleAction | MultiAction":
        return self.reduce(
            Payload(self.backend.min, kwargs=method_kwargs), dim, batch_size
        )

    def prod(
        self, dim: str = "", batch_size: int = 0, **method_kwargs
    ) -> "SingleAction | MultiAction":
        return self.reduce(
            Payload(self.backend.prod, kwargs=method_kwargs), dim, batch_size
        )


def _batch_transform(action: Action, selection: dict, payload: Payload) -> Action:
    selected = action.select(selection, drop=True)
    dim = list(selection.keys())[0]
    if isinstance(selected, SingleAction):
        selected._squeeze_dimension(dim, drop=True)
        return selected

    reduced = selected.reduce(payload, dim=dim)
    return reduced


def _expand_transform(
    action: Action, index: int, dim: str, axis: int = 0, new_axis: int = 0
) -> Action:
    ret = action.map(
        Payload(action.backend.take, [Node.input_name(0), index], {"axis": axis})
    )
    ret._add_dimension(dim, index, new_axis)
    return ret


class Fluent:
    def __init__(
        self,
        single_action=SingleAction,
        multi_action=MultiAction,
        backend=ArrayApiBackend,
    ):
        self.single_action = single_action
        self.multi_action = multi_action
        self.backend = backend

    def source(
        self, func: callable, args: tuple, kwargs: dict = {}
    ) -> "SingleAction | MultiAction":
        """
        Create source nodes in graph from an dataarray of payloads, containing
        payload for each source node. If none of func, args and kwargs are a xr.DataArray
        then returns SingleAction with Payload(func, args, kwargs). If any of func, args
        or kwargs is a xr.DataArray then creates node array with the same shape is created.

        Parameters
        ----------
        func: callable or xr.DataArray[callable], function or functions to apply in
        each node
        args: tuple or xr.DataArray[tuple], container for args or array of
        arguments for each node
        kwargs: dict or xr.DataArray[dict], kwargs or array of kwargs for function
        in each node

        Return
        ------
        SingleAction or MultiAction

        Raises
        ------
        ValueError, if func, args or kwargs are DataArrays with different shapes
        """
        if not any([isinstance(x, xr.DataArray) for x in [func, args, kwargs]]):
            payload = Payload(func, args, kwargs)
            return self.single_action.from_payload(None, payload, self.backend)

        shape = None
        ufunc_args = []
        for x in [func, args, kwargs]:
            if isinstance(x, xr.DataArray):
                if shape is None:
                    shape = (x.shape, x.coords, x.dims)
                elif shape != (x.shape, x.coords, x.dims):
                    raise ValueError(
                        "Shape, dims or coords of data arrays do not match"
                    )
                ufunc_args.append(x)
            else:
                ufunc_args.append(xr.DataArray(x))
        payloads = xr.apply_ufunc(Payload, *ufunc_args, vectorize=True)
        nodes = np.empty(payloads.shape, dtype=object)
        it = np.nditer(payloads, flags=["multi_index", "refs_ok"])
        # Ensure all source nodes have a unique name
        node_names = set()
        for payload in it:
            name = payload[()].name()
            if name in node_names:
                name += str(it.multi_index)
            node_names.add(name)
            nodes[it.multi_index] = Node(payload[()], name=name)
        return self.multi_action(
            None,
            xr.DataArray(
                nodes,
                dims=payloads.dims,
                coords=payloads.coords,
            ),
            self.backend,
        )
