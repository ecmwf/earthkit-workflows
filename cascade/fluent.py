import numpy as np
import xarray as xr
import functools
import hashlib
from dataclasses import dataclass, field

from .graph import Graph
from .graph import Node as BaseNode
from . import backends


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
        return str(self.to_tuple())


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
        name += f":{custom_hash(f'{[payload.to_tuple()] + [x.name for x in inputs]}')}"

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
    def __init__(self, previous: "Action", nodes: xr.DataArray):
        self.previous = previous
        assert not np.any(nodes.isnull()), "Array of nodes can not contain NaNs"
        self.nodes = nodes
        self.sinks = [] if previous is None else previous.sinks.copy()

    def graph(self) -> Graph:
        """
        Creates graph from sinks. If list of sinks is empty then uses the
        list of nodes.

        Return
        ------
        Graph instance constructed from list of sinks, or if empty, list
        of nodes

        """
        if len(self.sinks) == 0:
            return Graph(list(self.nodes.data.flatten()))
        return Graph(self.sinks)

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
            ret = type(self)(self, new_nodes)
        ret.sinks = self.sinks + other_action.sinks
        return ret

    def transform(self, func: callable, params: list, dim: str) -> "Action":
        """
        Create new nodes by applying function on action with different
        parameters. The result actions from applying function are joined
        along the specified dimension.

        Parameters
        ----------
        func: function with signature func(Action, arg) -> Action
        params: list, containing different arguments to pass into func
        for generating new nodes
        dim: str, name of dimension to join actions resulting from applying
        function on

        Return
        ------
        SingleAction or MultiAction
        """
        res = None
        for param in params:
            new_res = func(self, param)
            if res is None:
                res = new_res
            else:
                res = res.join(new_res, dim)
        # Remove expanded dimension if only a single element in param list
        res._squeeze_dimension(dim)
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
            new_nodes[it.multi_index] = Node(Payload(backends.trivial), node[()])

        new_nodes = xr.DataArray(
            new_nodes,
            coords=broadcasted_nodes.coords,
            dims=broadcasted_nodes.dims,
            attrs=self.nodes.attrs,
        )
        if hasattr(self, "to_multi"):
            return self.to_multi(new_nodes)
        return type(self)(self, new_nodes)

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

        def _expand(action: Action, index: int) -> Action:
            ret = action.map(
                Payload(backends.take, [Node.input_name(0), index], {"axis": axis})
            )
            ret._add_dimension(dim, index, new_axis)
            return ret

        return self.transform(_expand, np.arange(dim_size), dim)

    def add_attributes(self, attrs: dict):
        self.nodes.attrs.update(attrs)

    def _add_dimension(self, name: str, value: float, axis: int = 0):
        self.nodes = self.nodes.expand_dims({name: [value]}, axis)

    def _squeeze_dimension(self, dim_name: str):
        if dim_name in self.nodes.coords and len(self.nodes.coords[dim_name]) == 1:
            self.nodes = self.nodes.squeeze(dim_name)


class SingleAction(Action):
    def __init__(self, previous: Action, node):
        assert node.size == 1
        super().__init__(previous, node)

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
        return MultiAction(self, nodes)

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
        return type(self).from_payload(self, payload)

    def node(self) -> Node:
        return self.nodes.data[()]

    @classmethod
    def from_payload(cls, previous: Action, payload: Payload) -> "SingleAction":
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
                attrs=previous.nodes.attrs,
            )
        return cls(previous, node)


class MultiAction(Action):
    def __init__(self, previous, nodes):
        super().__init__(previous, nodes)

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
            return SingleAction.from_payload(self, payload_or_node)
        return SingleAction(self, payload_or_node)

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
        )

    def reduce(self, payload: Payload, dim: str = "") -> "SingleAction | MultiAction":
        """
        Reduction operation across the named dimension using the provided
        function in the payload

        Parameters
        ----------
        payload: Payload, payload specifying function for performing the reduction
        dim: str, name of dimension along which to return

        Return
        ------
        SingleAction or MultiAction
        """
        if self.nodes.ndim == 1:
            return self.to_single(payload)

        if len(dim) == 0:
            dim = self.nodes.dims[0]

        new_dims = [x for x in self.nodes.dims if x != dim]
        transposed_nodes = self.nodes.transpose(dim, *new_dims)
        new_nodes = np.empty(transposed_nodes.shape[1:], dtype=object)
        it = np.nditer(new_nodes, flags=["multi_index", "refs_ok"])
        for _ in it:
            inputs = transposed_nodes[(slice(None, None, 1), *it.multi_index)].data
            new_nodes[it.multi_index] = Node(payload, inputs)
        return type(self)(
            self,
            xr.DataArray(
                new_nodes,
                coords={key: self.nodes.coords[key] for key in new_dims},
                dims=new_dims,
                attrs=self.nodes.attrs,
            ),
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
            Payload(backends.stack, kwargs={"axis": axis, **method_kwargs}), dim
        )

    def select(self, criteria: dict) -> "SingleAction | MultiAction":
        """
        Create action contaning nodes match selection criteria

        Parameters
        ----------
        criteria: dict, key-value pairs specifying selection criteria

        Return
        ------
        SingleAction or MultiAction
        """
        if any([key not in self.nodes.dims for key in criteria.keys()]):
            raise NotImplementedError(
                f"Unknown coordinate in criteria {criteria}. Existing dimensions {self.nodes.dims}"
            )

        selected_nodes = self.nodes.sel(**criteria, drop=True)
        if selected_nodes.size == 1:
            return self.to_single(selected_nodes)
        return type(self)(self, selected_nodes)

    def concatenate(self, dim: str, **method_kwargs) -> "SingleAction | MultiAction":
        return self.reduce(Payload(backends.concat, kwargs=method_kwargs), dim)

    def mean(self, dim: str = "", **method_kwargs) -> "SingleAction | MultiAction":
        return self.reduce(Payload(backends.mean, kwargs=method_kwargs), dim)

    def std(self, dim: str = "", **method_kwargs) -> "SingleAction | MultiAction":
        return self.reduce(Payload(backends.std, kwargs=method_kwargs), dim)

    def maximum(self, dim: str = "", **method_kwargs) -> "SingleAction | MultiAction":
        return self.reduce(Payload(backends.max, kwargs=method_kwargs), dim)

    def minimum(self, dim: str = "", **method_kwargs) -> "SingleAction | MultiAction":
        return self.reduce(Payload(backends.min, kwargs=method_kwargs), dim)

    def subtract(self, dim: str = "", **method_kwargs) -> "SingleAction | MultiAction":
        return self.reduce(Payload(backends.subtract, kwargs=method_kwargs), dim)

    def add(self, dim: str = "", **method_kwargs) -> "SingleAction | MultiAction":
        return self.reduce(Payload(backends.add, kwargs=method_kwargs), dim)

    def divide(self, dim: str = "", **method_kwargs) -> "SingleAction | MultiAction":
        return self.reduce(Payload(backends.divide, kwargs=method_kwargs), dim)

    def multiply(self, dim: str = "", **method_kwargs) -> "SingleAction | MultiAction":
        return self.reduce(Payload(backends.multiply, kwargs=method_kwargs), dim)


class Fluent:
    single_action: type = SingleAction
    multi_action: type = MultiAction

    @classmethod
    def source(
        cls, payloads: np.ndarray[Payload], dims: list | dict, name: str = "source"
    ) -> "SingleAction | MultiAction":
        """
        Create source nodes in graph from an array of payloads, containing
        payload for each source node

        Parameters
        ----------
        payloads: np.ndarray[Payload], containing payload for each source node
        dims: list or dict, specifying dimension names. If dict, then used
        as coords in xarray.DataArray of nodes. If list, then values of coordinates
        are integers up to the dimension size
        name: str, common string to appear in the name of all source nodes

        Return
        ------
        SingleAction or MultiAction
        """
        payloads = np.asarray(payloads)
        if len(dims) == 0:
            return cls.single_action.from_payload(None, payloads[()])

        assert len(dims) == len(payloads.shape)
        if isinstance(dims, dict):
            coords = dims
            dims = list(coords.keys())
        else:
            coords = {x: np.arange(payloads.shape[i]) for i, x in enumerate(dims)}

        nodes = np.empty(payloads.shape, dtype=object)
        it = np.nditer(payloads, flags=["multi_index", "refs_ok"])
        for payload in it:
            nodes[it.multi_index] = Node(payload[()], name=f"{name}{it.multi_index}")
        return cls.multi_action(
            None,
            xr.DataArray(
                nodes,
                dims=dims,
                coords=coords,
            ),
        )
