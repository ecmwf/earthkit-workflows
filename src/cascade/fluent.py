import numpy as np
import xarray as xr
import functools
import hashlib
import copy
from typing import Callable

from .graph import Graph
from .graph import Node as BaseNode
from . import backends


class Payload:
    """
    Class for detailing function, args and kwargs to be computing in a graph node
    """

    def __init__(
        self, func: Callable, args: list | None = None, kwargs: dict | None = None
    ):
        if isinstance(func, functools.partial):
            if args is not None or kwargs is not None:
                raise ValueError("Partial function should not have args or kwargs")
            self.func = func.func
            self.args = func.args
            self.kwargs = func.keywords
        else:
            self.func = func
            self.args = args
            if kwargs is None:
                self.kwargs = {}
            else:
                self.kwargs = copy.deepcopy(kwargs)

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
        if self.has_args():
            raise ValueError("Arguments already set")
        self.args = args

    def to_tuple(self) -> tuple:
        """
        Return
        ------
        tuple, containing function, arguments and kwargs
        """
        if not self.has_args():
            raise ValueError("Arguments not set")
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
        payload: Callable | Payload,
        inputs: BaseNode | tuple[BaseNode] = (),
        name: str = None,
    ):
        if not isinstance(payload, Payload):
            payload = Payload(payload)
        else:
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

    def __str__(self) -> str:
        return f"Node {self.name}, inputs: {[x.parent.name for x in self.inputs.values()]}, payload: {self.payload}"


class Action:
    def __init__(self, nodes: xr.DataArray):
        assert not np.any(nodes.isnull()), "Array of nodes can not contain NaNs"
        self.nodes = nodes

    def graph(self) -> Graph:
        """
        Creates graph from the nodes of the action.

        Return
        ------
        Graph instance constructed from list of nodes

        """
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
        ret = type(self)(new_nodes)
        return ret

    def transform(
        self, func: Callable, params: list, dim: str | xr.DataArray
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
        Action
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
            new_nodes[it.multi_index] = Node(Payload(backends.trivial), node[()])

        new_nodes = xr.DataArray(
            new_nodes,
            coords=broadcasted_nodes.coords,
            dims=broadcasted_nodes.dims,
            attrs=self.nodes.attrs,
        )
        return type(self)(new_nodes)

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
        Action
        """
        params = [(x, dim, axis, new_axis) for x in range(dim_size)]
        return self.transform(_expand_transform, params, dim)

    def map(self, payload: Callable | np.ndarray[Callable]) -> "Action":
        """
        Apply specified payload on all nodes. If argument is an array of payloads,
        this must be the same size as the array of nodes and each node gets a
        unique payload from the array

        Parameters
        ----------
        payload: function or array of functions

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
        if not isinstance(payload, (Callable, Payload)):
            payload = np.asarray(payload)
            assert (
                payload.shape == self.nodes.shape
            ), f"For unique payloads for each node, payload shape {payload.shape} must match node array shape {self.nodes.shape}"

        # Applies operation to every node, keeping node array structure
        new_nodes = np.empty(self.nodes.shape, dtype=object)
        it = np.nditer(self.nodes, flags=["multi_index", "refs_ok"])
        node_payload = payload
        for node in it:
            if not isinstance(payload, (Callable, Payload)):
                node_payload = payload[it.multi_index]
            new_nodes[it.multi_index] = Node(node_payload, node[()])
        return type(self)(
            xr.DataArray(
                new_nodes,
                coords=self.nodes.coords,
                dims=self.nodes.dims,
                attrs=self.nodes.attrs,
            ),
        )

    def reduce(
        self,
        payload: Callable,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
    ) -> "Action":
        """
        Reduction operation across the named dimension using the provided
        function in the payload. If batch_size > 1 and less than the size
        of the named dimension, the reduction will be computed first in
        batches and then aggregated, otherwise no batching will be performed.

        Parameters
        ----------
        payload: function for performing the reduction
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
            dim = self.nodes.dims[0]

        batched = self
        level = 0
        if not isinstance(payload, Payload):
            payload = Payload(payload)
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
        result = type(batched)(
            xr.DataArray(
                new_nodes,
                coords=new_coords,
                dims=new_dims,
                attrs=batched.nodes.attrs,
            ),
        )
        if keep_dim:
            axis = self.nodes.dims.index(dim)
            result._add_dimension(
                dim, f"{self.nodes.coords[dim][0]}-{self.nodes.coords[dim][-1]}", axis
            )
        return result

    def flatten(self, dim: str = "", axis: int = 0, **kwargs) -> "Action":
        """
        Flattens the array of nodes along specified dimension by creating new
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
            Payload(backends.stack, kwargs={"axis": axis, **kwargs}), dim
        )

    def select(self, criteria: dict, drop: bool = False) -> "Action":
        """
        Create action contaning nodes match selection criteria

        Parameters
        ----------
        criteria: dict, key-value pairs specifying selection criteria
        drop: bool, drop coord variables in criteria if True

        Return
        ------
        Action
        """
        keys = list(criteria.keys())
        for key in keys:
            if key not in self.nodes.dims:
                if self.nodes.coords.get(key, None) == criteria[key]:
                    criteria.pop(key)
                else:
                    raise NotImplementedError(
                        f"Unknown dim in criteria {criteria}. Existing dimensions {self.nodes.dims} and coords {self.nodes.coords}"
                    )
        if len(criteria) == 0:
            return self
        selected_nodes = self.nodes.sel(**criteria, drop=drop)
        return type(self)(selected_nodes)

    def concatenate(
        self, dim: str, batch_size: int = 0, keep_dim: bool = False, **kwargs
    ) -> "Action":
        if self.nodes.sizes[dim] == 1:
            # no-op
            if not keep_dim:
                self._squeeze_dimension(dim)
            return self
        return self.reduce(
            Payload(backends.concat, kwargs=kwargs), dim, batch_size, keep_dim
        )

    def sum(
        self, dim: str = "", batch_size: int = 0, keep_dim: bool = False, **kwargs
    ) -> "Action":
        return self.reduce(
            Payload(backends.sum, kwargs=kwargs), dim, batch_size, keep_dim
        )

    def mean(
        self, dim: str = "", batch_size: int = 0, keep_dim: bool = False, **kwargs
    ) -> "Action":
        if len(dim) == 0:
            dim = self.nodes.dims[0]

        if batch_size <= 1 or batch_size >= self.nodes.sizes[dim]:
            return self.reduce(Payload(backends.mean, kwargs=kwargs), dim, keep_dim)

        return self.sum(dim, batch_size, keep_dim, **kwargs).divide(
            self.nodes.sizes[dim]
        )

    def std(
        self, dim: str = "", batch_size: int = 0, keep_dim: bool = False, **kwargs
    ) -> "Action":
        if len(dim) == 0:
            dim = self.nodes.dims[0]

        if batch_size <= 1 or batch_size >= self.nodes.sizes[dim]:
            return self.reduce(Payload(backends.std, kwargs=kwargs), dim)

        mean_sq = self.mean(dim, batch_size, keep_dim, **kwargs).power(2)
        norm = (
            self.power(2)
            .sum(dim, batch_size, keep_dim, **kwargs)
            .divide(self.nodes.sizes[dim])
        )
        return norm.subtract(mean_sq).power(0.5)

    def max(
        self, dim: str = "", batch_size: int = 0, keep_dim: bool = False, **kwargs
    ) -> "Action":
        return self.reduce(
            Payload(backends.max, kwargs=kwargs), dim, batch_size, keep_dim
        )

    def min(
        self, dim: str = "", batch_size: int = 0, keep_dim: bool = False, **kwargs
    ) -> "Action":
        return self.reduce(
            Payload(backends.min, kwargs=kwargs), dim, batch_size, keep_dim
        )

    def prod(
        self, dim: str = "", batch_size: int = 0, keep_dim: bool = False, **kwargs
    ) -> "Action":
        return self.reduce(
            Payload(backends.prod, kwargs=kwargs), dim, batch_size, keep_dim
        )

    def __two_arg_method(
        self, method: Callable, other: "Action | float", **kwargs
    ) -> "Action":
        if isinstance(other, Action):
            return self.join(other, "**datatype**", match_coord_values=True).reduce(
                Payload(method, kwargs=kwargs)
            )
        return self.map(
            Payload(method, args=(Node.input_name(0), other), kwargs=kwargs)
        )

    def subtract(self, other: "Action | float", **kwargs) -> "Action":
        return self.__two_arg_method(backends.subtract, other, **kwargs)

    def divide(self, other: "Action | float", **kwargs) -> "Action":
        return self.__two_arg_method(backends.divide, other, **kwargs)

    def add(self, other: "Action | float", **kwargs) -> "Action":
        return self.__two_arg_method(backends.add, other, **kwargs)

    def multiply(self, other: "Action | float", **kwargs) -> "Action":
        return self.__two_arg_method(backends.multiply, other, **kwargs)

    def power(self, other: "Action | float", **kwargs) -> "Action":
        return self.__two_arg_method(backends.pow, other, **kwargs)

    def add_attributes(self, attrs: dict):
        self.nodes.attrs.update(attrs)

    def _add_dimension(self, name: str, value: float, axis: int = 0):
        self.nodes = self.nodes.expand_dims({name: [value]}, axis)

    def _squeeze_dimension(self, dim_name: str, drop: bool = False):
        if dim_name in self.nodes.coords and len(self.nodes.coords[dim_name]) == 1:
            self.nodes = self.nodes.squeeze(dim_name, drop=drop)

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
    action: Action, index: int, dim: str, axis: int = 0, new_axis: int = 0
) -> Action:
    ret = action.map(
        Payload(backends.take, [Node.input_name(0), index], {"axis": axis})
    )
    ret._add_dimension(dim, index, new_axis)
    return ret


def from_source(
    payloads_list: np.ndarray[Callable],
    dims: list = None,
    coords: dict = None,
    action=Action,
) -> Action:

    payloads = xr.DataArray(payloads_list, dims=dims, coords=coords)
    nodes = np.empty(payloads.shape, dtype=object)
    it = np.nditer(payloads, flags=["multi_index", "refs_ok"])
    # Ensure all source nodes have a unique name
    node_names = set()
    for item in it:
        if not isinstance(item[()], Payload):
            payload = Payload(item[()])
        else:
            payload = item[()]
        name = payload.name()
        if name in node_names:
            name += str(it.multi_index)
        node_names.add(name)
        nodes[it.multi_index] = Node(payload, name=name)
    return action(
        xr.DataArray(
            nodes,
            dims=payloads.dims,
            coords=payloads.coords,
        ),
    )
