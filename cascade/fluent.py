import numpy as np
import xarray as xr
import itertools
from enum import Enum, auto
import functools

from ppgraph import Graph
from ppgraph import Node as PPNode

from .io import write as write_grib
from . import functions


def create_payload(func, args: list | tuple, kwargs: dict = {}):
    return (func, args, kwargs)


class Node(PPNode):
    def __init__(self, payload, inputs: PPNode | tuple[PPNode] = (), name=None):
        if isinstance(inputs, PPNode):
            inputs = [inputs]
        # If payload is just a function, assume inputs to the function are the inputs
        # to the node, in the order provided
        if not isinstance(payload, tuple) or len(payload) == 1:
            payload = create_payload(payload, [f"input{x}" for x in range(len(inputs))])
        else:
            payload = create_payload(*payload)
        assert len(payload) == 3

        if name is None:
            name = ""
            if hasattr(payload[0], "__name__"):
                name += payload[0].__name__
            name += str(hash(f"{[payload] + [x.name for x in inputs]}"))

        super().__init__(
            name,
            payload=payload,
            **{f"input{x}": node for x, node in enumerate(inputs)},
        )
        self.attributes = {}


class Action:
    def __init__(self, previous: "Action", nodes: xr.DataArray):
        self.previous = previous
        self.nodes = nodes
        self.sinks = [] if previous is None else previous.sinks.copy()

    def graph(self) -> Graph:
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

    def add_attributes(self, attrs: dict):
        self.nodes.attrs.update(attrs)

    def _add_dimension(self, name, value):
        self.nodes = self.nodes.expand_dims({name: [value]})

    def _squeeze_dimension(self, dim_name: str):
        if dim_name in self.nodes.coords and len(self.nodes.coords[dim_name]) == 1:
            self.nodes = self.nodes.squeeze(dim_name)


class SingleAction(Action):
    def __init__(self, payload, previous, node=None):
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

    def then(self, payload):
        return type(self)(payload, self)

    def write(self, target, config_grib_sets: dict):
        if target != "null:":
            grib_sets = config_grib_sets.copy()
            grib_sets.update(self.nodes.attrs)
            for name, values in self.nodes.coords.items():
                if values.data.ndim == 0:
                    grib_sets[name] = values.data
                else:
                    assert values.data.ndim == 1
                    grib_sets[name] = values.data[0]
            payload = (write_grib, (target, "input0", grib_sets))
            self.sinks.append(Node(payload, self.node()))
        return self

    def node(self):
        return self.nodes.data[()]


class MultiAction(Action):
    def __init__(self, previous, nodes):
        super().__init__(previous, nodes)

    def to_single(self, payload, node=None):
        return SingleAction(payload, self, node)

    def foreach(self, payload):
        # Applies operation to every node, keeping node array structure
        new_nodes = np.empty(self.nodes.shape, dtype=object)
        it = np.nditer(self.nodes, flags=["multi_index", "refs_ok"])
        for node in it:
            new_nodes[it.multi_index] = Node(payload, node[()])
        return type(self)(
            self,
            xr.DataArray(
                new_nodes,
                coords=self.nodes.coords,
                dims=self.nodes.dims,
                attrs=self.nodes.attrs,
            ),
        )

    def reduce(self, payload, key: str = ""):
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

    def transform(self, func, params: list, key: str):
        res = None
        for param in params:
            new_res = func(self, param)
            if res is None:
                res = new_res
            else:
                res = res.join(new_res, key)
        # Remove expanded dimension if only a single threshold in list
        res._squeeze_dimension(key)
        return res

    def write(self, target, config_grib_sets: dict):
        if target != "null:":
            coords = list(self.nodes.coords.keys())
            for node_attrs in itertools.product(
                *[self.nodes.coords[key].data for key in coords]
            ):
                node_coords = {
                    key: node_attrs[index] for index, key in enumerate(coords)
                }
                node = self.node(node_coords)

                grib_sets = config_grib_sets.copy()
                grib_sets.update(self.nodes.attrs)
                grib_sets.update(node_coords)
                self.sinks.append(
                    Node((write_grib, (target, "input0", grib_sets)), [node])
                )
        return self

    def node(self, criteria: dict):
        return self.nodes.sel(**criteria, drop=True).data[()]

    def concatenate(self, key: str):
        return self.reduce(functions._concatenate, key)

    def mean(self, key: str = ""):
        return self.reduce(functions._mean, key)

    def std(self, key: str = ""):
        return self.reduce(functions._std, key)

    def maximum(self, key: str = ""):
        return self.reduce(functions._maximum, key)

    def minimum(self, key: str = ""):
        return self.reduce(functions._minimum, key)

    def norm(self, key: str = ""):
        return self.reduce(functions._norm, key)

    def diff(self, key: str = ""):
        return self.reduce((functions._subtract, ("input1", "input0")), key)

    def subtract(self, key: str = "", extract_keys: tuple = ()):
        return self.reduce(
            (functions._subtract, ("input0", "input1", extract_keys)), key
        )

    def add(self, key: str = ""):
        return self.reduce(functions._add, key)

    def divide(self, key: str = ""):
        return self.reduce(functions._divide, key)

    def multiply(self, key: str = ""):
        return self.reduce(functions._multiply, key)
