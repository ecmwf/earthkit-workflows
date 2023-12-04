import numpy as np
import xarray as xr

import functools
import hashlib

from .graph import Graph
from .graph import Node as BaseNode
from . import functions


class Payload:
    def __init__(self, func, args: list | tuple = None, kwargs: dict = {}):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def has_args(self) -> bool:
        return self.args is not None

    def set_args(self, args: list | tuple):
        assert not self.has_args()
        self.args = args

    def to_tuple(self) -> tuple:
        assert self.has_args()
        return (self.func, self.args, self.kwargs)

    def name(self) -> str:
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
        self, payload: Payload, inputs: BaseNode | tuple[BaseNode] = (), name=None
    ):
        if isinstance(inputs, BaseNode):
            inputs = [inputs]
        # If payload doesn't have inputs, assume inputs to the function are the inputs
        # to the node, in the order provided
        if not payload.has_args():
            payload.set_args([f"input{x}" for x in range(len(inputs))])
        else:
            # All inputs into Node should also feature in payload - no dangling inputs
            assert np.all(
                [x in payload.args for x in [f"input{x}" for x in range(len(inputs))]]
            ), f"Payload {payload} does not use all node inputs {len(inputs)}"

        if name is None:
            name = payload.name()
            name += (
                f":{custom_hash(f'{[payload.to_tuple()] + [x.name for x in inputs]}')}"
            )

        super().__init__(
            name,
            payload=payload.to_tuple(),
            **{f"input{x}": node for x, node in enumerate(inputs)},
        )
        self.attributes = {}


class Action:
    def __init__(self, previous: "Action", nodes: xr.DataArray):
        self.previous = previous
        self.nodes = nodes
        self.sinks = [] if previous is None else previous.sinks.copy()

    def graph(self) -> Graph:
        """
        Creates graph from sinks. If list of sinks is empty then uses the
        list of nodes.

        Returns
        -------
        Graph instance constructed from list of sinks, or if empty, list
        of nodes

        """
        if len(self.sinks) == 0:
            return Graph(list(self.nodes.data.flatten()))
        return Graph(self.sinks)

    def join(
        self,
        other_action: "Action",
        dim_name: str | xr.DataArray,
        match_coord_values: bool = False,
    ) -> "MultiAction":
        if match_coord_values:
            for coord, values in self.nodes.coords.items():
                if coord in other_action.nodes.coords:
                    other_action.nodes = other_action.nodes.assign_coords(
                        **{coord: values}
                    )
        new_nodes = xr.concat(
            [self.nodes, other_action.nodes],
            dim_name,
            combine_attrs="no_conflicts",
            coords="minimal",
        )
        if hasattr(self, "to_multi"):
            ret = self.to_multi(new_nodes)
        else:
            ret = type(self)(self, new_nodes)
        ret.sinks = self.sinks + other_action.sinks
        return ret

    def transform(self, func, params: list, key: str):
        res = None
        for param in params:
            new_res = func(self, param)
            if res is None:
                res = new_res
            else:
                res = res.join(new_res, key)
        # Remove expanded dimension if only a single element in param list
        res._squeeze_dimension(key)
        return res

    def broadcast(self, other_action: "Action", exclude: list[str] = None) -> "Action":
        """
        Broadcast the nodes against nodes in other_action

        Params
        ------
        other_action: Action containings nodes to broadcast against
        exclude: List of dimension names to exclude from broadcasting

        Return
        ------
        Action with broadcasted set of nodes
        """
        # Ensure coordinates in existing dimensions match, otherwise obtain NaNs
        for key, values in other_action.nodes.coords.items():
            if key in self.nodes.coords and (exclude is None or key not in exclude):
                assert np.all(
                    values.data == self.nodes.coords[key].data
                ), f"Existing coordinates must match for broadcast. Found mismatch in {key}!"

        broadcasted_nodes = self.nodes.broadcast_like(other_action.nodes, exclude)
        new_nodes = np.empty(broadcasted_nodes.shape, dtype=object)
        it = np.nditer(
            self.nodes.transpose(*broadcasted_nodes.dims, missing_dims="ignore"),
            flags=["multi_index", "refs_ok"],
        )
        for node in it:
            new_nodes[it.multi_index] = Node(Payload(functions.trivial), node[()])
        return MultiAction(
            self,
            xr.DataArray(
                new_nodes,
                coords=broadcasted_nodes.coords,
                dims=broadcasted_nodes.dims,
                attrs=self.nodes.attrs,
            ),
        )

    def add_attributes(self, attrs: dict):
        self.nodes.attrs.update(attrs)

    def _add_dimension(self, name, value):
        self.nodes = self.nodes.expand_dims({name: [value]})

    def _squeeze_dimension(self, dim_name: str):
        if dim_name in self.nodes.coords and len(self.nodes.coords[dim_name]) == 1:
            self.nodes = self.nodes.squeeze(dim_name)


class SingleAction(Action):
    def __init__(self, payload: Payload, previous, node=None):
        if node is None:
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
        assert node.size == 1
        super().__init__(previous, node)

    def to_multi(self, nodes):
        return MultiAction(self, nodes)

    def map(self, payload: Payload):
        return type(self)(payload, self)

    def node(self):
        return self.nodes.data[()]


class MultiAction(Action):
    def __init__(self, previous, nodes):
        super().__init__(previous, nodes)

    def to_single(self, payload: Payload, node=None):
        return SingleAction(payload, self, node)

    def map(self, payload: Payload | np.ndarray[Payload]):
        """
        Parameters
        ----------
        payload: Payload or array of Payloads

        Returns
        -------
        MultiAction where nodes are a result of applying the same
        payload to all nodes, or in the case where payload is an array,
        applying a different payload to each node
        """
        if not isinstance(payload, Payload):
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

    def reduce(self, payload: Payload, key: str = ""):
        if self.nodes.ndim == 1:
            return self.to_single(payload)

        if len(key) == 0:
            key = self.nodes.dims[0]

        new_dims = [x for x in self.nodes.dims if x != key]
        transposed_nodes = self.nodes.transpose(key, *new_dims)
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

    def select(self, criteria: dict):
        if any([key not in self.nodes.dims for key in criteria.keys()]):
            raise NotImplementedError(
                f"Unknown coordinate in criteria {criteria}. Existing dimensions {self.nodes.dims}"
            )

        selected_nodes = self.nodes.sel(**criteria, drop=True)
        if selected_nodes.size == 1:
            return self.to_single(None, selected_nodes)
        return type(self)(self, selected_nodes)

    def node(self, criteria: dict):
        return self.nodes.sel(**criteria, drop=True).data[()]
