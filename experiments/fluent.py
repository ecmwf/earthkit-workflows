import randomname
import functools
from ppgraph import Node, Graph
from typing import List, Dict
from ppgraph import pyvis
import numpy as np
import itertools
import xarray as xr
from collections import OrderedDict


def read(request: Dict):
    print(request)


class Cascade:
    def read(self, request: Dict):
        # Need to carry data about the request to know which node corresponds to
        # what meta data
        dims = OrderedDict()
        for key, values in request.items():
            if hasattr(values, "__iter__") and not isinstance(values, str):
                dims[key] = len(values)

        if len(dims) == 0:
            return SingleAction(functools.partial(read, request), None)

        nodes = np.empty(tuple(dims.values()), dtype=object)
        for params in itertools.product(*[request[x] for x in dims.keys()]):
            new_request = request.copy()
            indices = []
            for index, expand_param in enumerate(dims.keys()):
                try:
                    new_request[expand_param] = params[index]
                except:
                    print(params, index)
                    raise
                indices.append(list(request[expand_param]).index(params[index]))
            read_with_request = functools.partial(read, new_request)
            try:
                nodes[tuple(indices)] = Node(
                    randomname.generate(), payload=read_with_request
                )
            except:
                print(indices, dims, nodes.shape)
                print(nodes)
                raise
        # Currently using xarray for keeping track of dimensions but these should
        # belong in node attributes on the graph?
        return MultiAction(
            None,
            xr.DataArray(
                nodes,
                dims=dims.keys(),
                coords={key: list(request[key]) for key in dims.keys()},
            ),
        )


class SingleAction:
    def __init__(self, func, previous):
        self.previous = previous
        if self.previous is None:
            self.nodes = xr.DataArray([Node(randomname.generate(), payload=func)])
        else:
            self.nodes = xr.DataArray(
                [
                    Node(
                        randomname.generate(),
                        payload=func,
                        **{
                            f"input{x}": node
                            for x, node in enumerate(self.previous.nodes.data.flatten())
                        },
                    )
                ]
            )

    def then(self, func):
        return SingleAction(func, self)

    def write(self):
        return SingleAction(lambda x: print(x), self)

    def graph(self):
        return Graph(self.nodes.data)


class MultiAction:
    def __init__(self, previous, nodes, internal_dims=0):
        self.previous = previous
        self.nodes = nodes
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
        new_nodes = np.empty(len(grouped_nodes), dtype=object)
        for index, group in enumerate(grouped_nodes):
            group_name, nodes = group
            inputs = {f"input{x}": node.data[()] for x, node in enumerate(nodes)}
            new_nodes[index] = Node(
                f"{key}:{group_name}", payload=np.concatenate, **inputs
            )
        new_nodes = xr.DataArray(
            new_nodes, dims=key, coords={key: self.nodes.coords[key]}
        )
        return MultiAction(self, new_nodes, self.internal_dims + 1)

    def reduce(self, func, key: str = ""):
        if self.nodes.size == 1 and self.internal_dims == 1:
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
        self.nodes.transpose(key, *new_dims)
        new_nodes = np.empty(self.nodes.shape[1:], dtype=object)
        it = np.nditer(new_nodes, flags=["multi_index", "refs_ok"])
        for _ in it:
            inputs = {
                f"input{x}": node
                for x, node in enumerate(
                    self.nodes[(slice(None, None, 1), *it.multi_index)].data
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

    def join(self, other_action: "MultiAction", dim_name: str = ""):
        if len(dim_name) == 0:
            dim_name = randomname.generate()
        return MultiAction(
            self,
            xr.concat([self.nodes, other_action.nodes], dim_name),
            self.internal_dims,
        )

    def graph(self):
        return Graph(self.nodes.data)


window_ranges = [[120, 144], [168, 240]]
window_cascades = []
total_graph = Graph([])

for window in window_ranges:
    start = window[0]
    end = window[1]

    climatology = Cascade().read(
        {
            "stream": "efhs",
            "levtype": "pl",
            "level": 850,
            "step": range(start, end + 1, 12),
        }
    )

    t850 = (
        Cascade()
        .read(
            {
                "stream": "enfo",
                "levtype": "pl",
                "level": 850,
                "number": range(1, 5),
                "step": range(start, end + 1, 12),
            }
        )
        .groupby("step")
        .join(climatology)
        .reduce(np.subtract)
        .reduce(np.mean)
        .map(lambda x: 1 if x > -2 else 0)
        .reduce(np.mean)
        .write()
    )

    window_cascades.append(t850)

    # draw graph
    g = t850.graph()
    pyvis.to_pyvis(g, notebook=True, cdn_resources="remote").show(
        f"graph_{window}.html"
    )
    total_graph += g

if False:
    # concatenate not yet supported
    Cascade().concatenate(*total_graph).schedule()

if False:
    # some other examples of fluent queries
    take10 = Cascade().orderby(lambda x: x.step).take(10)
    other = Cascade().read()

    # could implement forking (not yet supported)
    take10.foreach(lambda x: x + 1).mean().then(lambda x: x + 1).write()
    take10.mean().join(other).sum().write()


# orderby
# groupby
# take
# where
# sum/max/min/mean
# join (into tuples)

# Cascade::read().joinread()


pyvis.to_pyvis(g, notebook=True, cdn_resources="remote").show("index.html")


# # -----------------------

# Dask notes:

# cant go into nested functions
# cant do foreach, we have to code it explicitly
# have to wrap every function in a dask.delayed(func)(args) call
# annotations using with:
