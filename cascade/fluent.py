import randomname
from ppgraph import Node, Graph
import numpy as np
import xarray as xr


class Action:
    def __init__(self, previous: "Action", nodes: xr.DataArray):
        self.previous = previous
        self.nodes = nodes

    def graph(self) -> Graph:
        return Graph(list(self.nodes.data.flatten()))


class SingleAction(Action):
    def __init__(self, func, previous, nodes=None):
        if nodes is None:
            if previous is None:
                nodes = xr.DataArray([Node(randomname.generate(), payload=func)])
            else:
                nodes = xr.DataArray(
                    [
                        Node(
                            randomname.generate(),
                            payload=func,
                            **{
                                f"input{x}": node
                                for x, node in enumerate(previous.nodes.data.flatten())
                            },
                        )
                    ]
                )
        assert nodes.size == 1
        super().__init__(previous, nodes)

    def then(self, func):
        return SingleAction(func, self)

    def join(self, other_action: "Action", dim_name: str = "", 
             match_coord_values: bool = False):
        if len(dim_name) == 0:
            dim_name = randomname.generate()
        if match_coord_values:
            for coord, values in self.nodes.coords.items():
                if coord in other_action.nodes.coords:
                    other_action.nodes = other_action.nodes.assign_coords({coord: values})
        return MultiAction(
            self, xr.concat([self.nodes, other_action.nodes], dim_name)
        )

    def write(self):
        return SingleAction(lambda x: print(x), self)


class MultiAction(Action):
    def __init__(self, previous, nodes, internal_dims=0):
        super().__init__(previous, nodes)
        self.internal_dims = internal_dims

    def map(self, func):
        # Applies operation to every node, keeping node array structure
        new_nodes = np.empty(self.nodes.shape, dtype=object)
        it = np.nditer(self.nodes, flags=["multi_index", "refs_ok"])
        for node in it:
            new_nodes[it.multi_index] = Node(
                randomname.generate(), payload=func, input=node[()]
            )
        return MultiAction(
            self,
            xr.DataArray(new_nodes, coords=self.nodes.coords, dims=self.nodes.dims),
            self.internal_dims,
        )

    def groupby(self, key):
        grouped_nodes = self.nodes.groupby(key)
        if len(self.nodes.dims) > 2:
            print(
                f"Warning: groupby along {key} flattens across remaining dimensions {grouped_nodes.dims}"
            )
        new_nodes = np.empty(len(grouped_nodes), dtype=object)
        for index, group in enumerate(grouped_nodes):
            _, nodes = group
            inputs = {f"input{x}": node for x, node in enumerate(nodes.data.flatten())}
            new_nodes[index] = Node(
                randomname.generate(), payload=np.concatenate, **inputs
            )
        new_nodes = xr.DataArray(
            new_nodes, dims=key, coords={key: self.nodes.coords[key]}
        )
        return MultiAction(self, new_nodes, self.internal_dims + 1)

    def reduce(self, func, key: str = ""):
        if (self.nodes.size == 1 and self.internal_dims == 1) or (
            self.internal_dims == 0 and self.nodes.ndim == 1
        ):
            return SingleAction(func, self)

        # If reduction operation acts on internal array, then need to reduce internal dimensions
        if self.nodes.ndim == 1:
            new_dims = self.internal_dims
            if self.nodes.size == 1:
                new_dims -= 1
            assert new_dims >= 0
            new_nodes = np.array(
                [
                    Node(
                        randomname.generate(),
                        payload=func,
                        **{f"input{x}": node for x, node in enumerate(self.nodes.data)},
                    )
                ]
            )
            return MultiAction(self, xr.DataArray(new_nodes), new_dims)

        if len(key) == 0:
            key = self.nodes.dims[0]

        new_dims = [x for x in self.nodes.dims if x != key]
        transposed_nodes = self.nodes.transpose(key, *new_dims)
        new_nodes = np.empty(transposed_nodes.shape[1:], dtype=object)
        it = np.nditer(new_nodes, flags=["multi_index", "refs_ok"])
        for _ in it:
            inputs = {
                f"input{x}": node
                for x, node in enumerate(
                    transposed_nodes[(slice(None, None, 1), *it.multi_index)].data
                )
            }
            new_nodes[it.multi_index] = Node(
                randomname.generate(), payload=func, **inputs
            )
        return MultiAction(
            self,
            xr.DataArray(
                new_nodes,
                coords={key: self.nodes.coords[key] for key in new_dims},
                dims=new_dims,
            ),
            self.internal_dims,
        )

    def join(self, other_action: "Action", dim_name: str = "", 
            match_coord_values: bool = False):
        if len(dim_name) == 0:
            dim_name = randomname.generate()
        if match_coord_values:
            for coord, values in self.nodes.coords.items():
                if coord in other_action.nodes.coords:
                    other_action.nodes = other_action.nodes.assign_coords(**{coord: values})
        return MultiAction(
            self,
            xr.concat([self.nodes, other_action.nodes], dim_name),
            self.internal_dims,
        )

    def select(self, coord: str, value):
        if coord in self.nodes.dims:
            selected_nodes = self.nodes.sel(**{coord: value})
            if selected_nodes.size == 1:
                return SingleAction(None, self, selected_nodes.expand_dims("dim_0"))
            return MultiAction(self, selected_nodes, self.internal_dims)

        raise NotImplementedError("Selecting on internal dimensions")

    def write(self):
        return self.map(lambda x: print(x))
