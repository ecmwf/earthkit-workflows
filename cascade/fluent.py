import randomname
import numpy as np
import xarray as xr
import itertools
from enum import Enum, auto

from ppgraph import Graph
from ppgraph import Node as PPNode

from .io import write as write_grib


class Node(PPNode):
    class Attributes(Enum):
        GRIB_KEYS = auto()

    def __init__(self, payload, inputs=(), name=None):
        if name is None:
            name = randomname.generate()
        if isinstance(inputs, PPNode):
            super().__init__(name, payload=payload, input=inputs)
        else:
            super().__init__(
                name,
                payload=payload,
                **{f"input{x}": node for x, node in enumerate(inputs)},
            )
        self.attributes = {}

    def add_attribute(self, key: Attributes, value):
        assert key in Node.Attributes, f"Unknown attribute {key}"
        if key == Node.Attributes.GRIB_KEYS:
            self.attributes.setdefault(key, {}).update(value)
        else:
            self.attributes[key] = value

    def get_attribute(self, key):
        if key == Node.Attributes.GRIB_KEYS:
            return self.attributes.get(key, {})
        return self.attributes.get(key, 0)


class Action:
    def __init__(self, previous: "Action", nodes: xr.DataArray):
        self.previous = previous
        self.nodes = nodes

    def graph(self) -> Graph:
        return Graph(list(self.nodes.data.flatten()))

    def join(
        self,
        other_action: "Action",
        dim_name: str | xr.DataArray = None,
        match_coord_values: bool = False,
    ) -> "MultiAction":
        if dim_name is None:
            dim_name = randomname.generate()
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
            return self.to_multi(new_nodes)
        return type(self)(self, new_nodes)

    def add_attributes(self, attrs: dict):
        self.nodes.attrs.update(attrs)

    def _add_dimension(self, name, value):
        self.nodes = self.nodes.expand_dims({name: [value]})

    def _squeeze_dimension(self, dim_name: str):
        if dim_name in self.nodes.coords and len(self.nodes.coords[dim_name]) == 1:
            self.nodes = self.nodes.squeeze(dim_name)


class SingleAction(Action):
    def __init__(self, func, previous, node=None):
        if node is None:
            if previous is None:
                node = xr.DataArray(Node(func))
            else:
                node = xr.DataArray(
                    Node(
                        func,
                        previous.nodes.data.flatten(),
                    ),
                    attrs=previous.nodes.attrs,
                )
        assert node.size == 1
        super().__init__(previous, node)

    def to_multi(self, nodes):
        return MultiAction(self, nodes)

    def add_node_attributes(self, key: Node.Attributes, value):
        node = self.node()
        node.add_attribute(key, value)
        self.nodes = xr.DataArray(node, attrs=self.nodes.attrs)

    def then(self, func):
        return type(self)(func, self)

    def write(self, target, config_grib_sets: dict):
        grib_sets = config_grib_sets.copy()
        grib_sets.update(self.nodes.attrs)
        grib_sets.update(self.node().get_attribute(Node.Attributes.GRIB_KEYS))
        payload = (write_grib, target, "input", grib_sets)
        return type(self)(
            payload,
            self,
            node=xr.DataArray(Node(payload, self.node())),
        )

    def node(self):
        return self.nodes.data[()]


class MultiAction(Action):
    def __init__(self, previous, nodes):
        super().__init__(previous, nodes)

    def to_single(self, func, node=None):
        return SingleAction(func, self, node)

    def add_node_attributes(self, key: Node.Attributes, value, criteria: dict):
        node = self.node(criteria)
        node.add_attribute(key, value)
        self.nodes.loc[criteria] = node

    def foreach(self, func):
        # Applies operation to every node, keeping node array structure
        new_nodes = np.empty(self.nodes.shape, dtype=object)
        it = np.nditer(self.nodes, flags=["multi_index", "refs_ok"])
        for node in it:
            new_nodes[it.multi_index] = Node(func, node[()])
        return type(self)(
            self,
            xr.DataArray(
                new_nodes,
                coords=self.nodes.coords,
                dims=self.nodes.dims,
                attrs=self.nodes.attrs,
            ),
        )

    def reduce(self, func, key: str = ""):
        if self.nodes.ndim == 1:
            return self.to_single(func)

        if len(key) == 0:
            key = self.nodes.dims[0]

        new_dims = [x for x in self.nodes.dims if x != key]
        transposed_nodes = self.nodes.transpose(key, *new_dims)
        new_nodes = np.empty(transposed_nodes.shape[1:], dtype=object)
        it = np.nditer(new_nodes, flags=["multi_index", "refs_ok"])
        for _ in it:
            inputs = transposed_nodes[(slice(None, None, 1), *it.multi_index)].data
            new_nodes[it.multi_index] = Node(func, inputs)
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
        coords = list(self.nodes.coords.keys())
        new_nodes = []
        for node_attrs in itertools.product(
            *[self.nodes.coords[key].data for key in coords]
        ):
            node_coords = {key: node_attrs[index] for index, key in enumerate(coords)}
            node = self.node(node_coords)

            grib_sets = config_grib_sets.copy()
            grib_sets.update(self.nodes.attrs)
            grib_sets.update(node_coords)
            grib_sets.update(node.get_attribute(Node.Attributes.GRIB_KEYS))
            new_nodes.append(Node((write_grib, target, "input0", grib_sets), [node]))
        return type(self)(
            self,
            xr.DataArray(new_nodes),
        )

    def node(self, criteria: dict):
        return self.nodes.sel(**criteria, drop=True).data[()]

    def concatenate(self, key: str):
        return self.reduce(_concatenate, key)

    def mean(self, key: str = ""):
        def _mean(*arrays):
            concat = _concatenate(*arrays)
            return concat.mean(dim=concat.dims[0], keep_attrs=True)

        return self.reduce(_mean, key)

    def std(self, key: str = ""):
        def _std(*arrays):
            concat = _concatenate(*arrays)
            return concat.std(dim=concat.dims[0], keep_attrs=True)

        return self.reduce(_std, key)

    def maximum(self, key: str = ""):
        def _maximum(*arrays):
            concat = _concatenate(*arrays)
            return concat.max(dim=concat.dims[0], keep_attrs=True)

        return self.reduce(_maximum, key)

    def minimum(self, key: str = ""):
        def _minimum(*arrays):
            concat = _concatenate(*arrays)
            return concat.min(dim=concat.dims[0], keep_attrs=True)

        return self.reduce(_minimum, key)

    # Should perform dimension check for these as they only expect two arguments

    def diff(self, key: str = ""):
        return self.reduce((np.subtract, "input1", "input0"), key)

    def norm(self, key: str = ""):
        return self.reduce(
            lambda x, y: xr.DataArray(np.sqrt(x**2 + y**2), attrs=x.attrs), key
        )

    def subtract(self, key: str = ""):
        return self.reduce(np.subtract, key)

    def add(self, key: str = ""):
        return self.reduce(np.add, key)

    def divide(self, key: str = ""):
        return self.reduce(np.divide, key)

    def multiply(self, key: str = ""):
        return self.reduce(np.multiply, key)


def _concatenate(*arrays):
    newdim = arrays[0].ndim + 1
    return xr.concat(tuple(arr for arr in arrays), dim=f"dim{newdim}")
