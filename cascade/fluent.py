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
        new_nodes = xr.concat([self.nodes, other_action.nodes], dim_name)
        if hasattr(self, "to_multi"):
            return self.to_multi(new_nodes)
        return type(self)(self, new_nodes)


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

    def to_multi(self, nodes):
        return MultiAction(self, nodes)

    def then(self, func):
        return type(self)(func, self)

    def write(self):
        return type(self)(lambda x: print(x), self)


class MultiAction(Action):
    def __init__(self, previous, nodes):
        super().__init__(previous, nodes)

    def to_single(self, func, nodes=None):
        return SingleAction(func, self, nodes)

    def foreach(self, func):
        # Applies operation to every node, keeping node array structure
        new_nodes = np.empty(self.nodes.shape, dtype=object)
        it = np.nditer(self.nodes, flags=["multi_index", "refs_ok"])
        for node in it:
            new_nodes[it.multi_index] = Node(
                randomname.generate(), payload=func, input=node[()]
            )
        return type(self)(
            self,
            xr.DataArray(new_nodes, coords=self.nodes.coords, dims=self.nodes.dims),
        )

    def groupby(self, key):
        assert key in self.nodes.dims
        new_dims = [key] + [x for x in self.nodes.dims if x != key]
        transposed_nodes = self.nodes.transpose(*new_dims)
        new_nodes = xr.DataArray(
            transposed_nodes, dims=new_dims, coords=self.nodes.coords
        )
        return type(self)(self, new_nodes)

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
            inputs = {
                f"input{x}": node
                for x, node in enumerate(
                    transposed_nodes[(slice(None, None, 1), *it.multi_index)].data
                )
            }
            new_nodes[it.multi_index] = Node(
                randomname.generate(), payload=func, **inputs
            )
        return type(self)(
            self,
            xr.DataArray(
                new_nodes,
                coords={key: self.nodes.coords[key] for key in new_dims},
                dims=new_dims,
            ),
        )

    def select(self, coord: str, value):
        if coord not in self.nodes.dims:
            raise NotImplementedError(
                f"Unknown coordinate {coord}. Existing dimensions {self.nodes.dims}"
            )

        selected_nodes = self.nodes.sel(**{coord: value})
        if selected_nodes.size == 1:
            return self.to_single(None, selected_nodes.expand_dims({coord: [value]}))
        return type(self)(self, selected_nodes)

    def _concatenate(self, key):
        assert self.nodes.size == len(self.nodes.coords[key])

        return type(self)(
            self,
            xr.DataArray(
                [
                    Node(
                        randomname.generate(),
                        payload=np.concatenate,
                        **{
                            f"input{x}": node
                            for x, node in enumerate(self.nodes.data.flatten())
                        },
                    )
                ]
            ),
        )

    def write(self):
        return self.foreach(lambda x: print(x))
