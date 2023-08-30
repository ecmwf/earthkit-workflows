import randomname
import numpy as np
import xarray as xr
import itertools

from ppgraph import Node as PPNode
from ppgraph import Graph


class Node(PPNode):
    def __init__(self, payload, inputs=(), name=None):
        print(payload, inputs)
        if name is None:
            name = randomname.generate()
        if isinstance(inputs, PPNode):
            super().__init__(name, payload=payload, input=inputs)
        else:
            super().__init__(
                name,
                payload=payload,
                **{f"inputs{x}": node for x, node in enumerate(inputs)},
            )
        self.attrs = {}

    def add_attributes(self, attrs: dict):
        self.attrs.update(attrs)


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

    def then(self, func):
        return type(self)(func, self)

    def write(self):
        grib_sets = self.nodes.attrs.copy()
        grib_sets.update(self.nodes.data[()].attrs)
        return type(self)(
            f"write_{grib_sets}",
            self,
            node=xr.DataArray(Node(f"write_{grib_sets}", self.nodes.data[()])),
        )

    def add_node_attributes(self, attrs: dict):
        node = self.nodes.data[()]
        node.add_attributes(attrs)
        self.nodes = xr.DataArray(node, attrs=self.nodes.attrs)


class MultiAction(Action):
    def __init__(self, previous, nodes):
        super().__init__(previous, nodes)

    def to_single(self, func, node=None):
        return SingleAction(func, self, node)

    # def add_node_attributes(self, node_index: int, attrs: dict):
    #     try:
    #         self.nodes[()] = self.nodes[()].add_attributes(attrs)
    #     except:
    #         print(self.nodes.data[node_index])
    #         raise

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

    def select(self, coord: str, value):
        if coord not in self.nodes.dims:
            raise NotImplementedError(
                f"Unknown coordinate {coord}. Existing dimensions {self.nodes.dims}"
            )

        selected_nodes = self.nodes.sel(**{coord: value}, drop=True)
        if selected_nodes.size == 1:
            return self.to_single(None, selected_nodes)
        return type(self)(self, selected_nodes)

    def write(self):
        coords = list(self.nodes.coords.keys())
        new_nodes = []
        for node_attrs in itertools.product(
            *[self.nodes.coords[key].data for key in coords]
        ):
            node_coords = {key: node_attrs[index] for index, key in enumerate(coords)}
            node = self.nodes.sel(**node_coords).data[()]
            grib_sets = self.nodes.attrs.copy()
            print(node)
            grib_sets.update(node.attrs)
            grib_sets.update(node_coords)
            new_nodes.append(Node(f"write_{grib_sets}", [node]))
        return type(self)(
            self,
            xr.DataArray(new_nodes),
        )
