import functools
from ppgraph import Node, Graph
from ppgraph import pyvis
import numpy as np
import itertools
import xarray as xr
from collections import OrderedDict

from cascade.fluent import SingleAction, MultiAction


def read(request: dict):
    print(request)


class Cascade:
    def read(self, request: dict):
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
                new_request[expand_param] = params[index]
                indices.append(list(request[expand_param]).index(params[index]))
            read_with_request = functools.partial(read, new_request)
            nodes[tuple(indices)] = Node(
                ",".join([f"{key}={new_request[key]}" for key in dims]),
                payload=read_with_request,
            )
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


def test_t850():
    num_ensembles = 4
    window_ranges = [[120, 240], [168, 240]]

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
                    "number": range(1, num_ensembles + 1),
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

        total_graph += t850.graph()
    pyvis.to_pyvis(total_graph, notebook=True, cdn_resources="remote").show(
            f"t850_graph.html"
        )


def test_wind():
    num_ensembles = 4
    start = 0
    end = 24
    by = 3
    cf = Cascade().read(
        {
            "stream": "enfo",
            "levtype": "pl",
            "levelist": [250, 850],
            "param": [138, 155],
            "type": "cf",
            "number": range(0, 1),
            "step": range(start, end + 1, by),
        }
    )
    wind_speed = (
        Cascade()
        .read(
            {
                "stream": "enfo",
                "levtype": "pl",
                "levelist": [250, 850],
                "param": [138, 155],
                "type": "pf",
                "number": range(1, num_ensembles + 1),
                "step": range(start, end + 1, by),
            }
        )
        .join(cf, "number")
        .reduce(lambda x, y: np.sqrt(x**2 + y**2), "param")
    )

    mean = wind_speed.reduce(np.mean, "number").write()
    std = wind_speed.reduce(np.std, "number").write()

    # draw graph
    wind_graph = mean.graph() + std.graph()
    pyvis.to_pyvis(wind_graph, notebook=True, cdn_resources="remote").show(
        f"wind_graph.html"
    )

def test_extreme():
    num_ensembles = 4
    window_ranges = [[120, 144], [132, 156]]

    for window in window_ranges:
        start, end = window
        climatology = Cascade().read(
            {
                "stream": "efhs",
                "levtype": "sfc",
                "param": 228044,
                "step": f"{start}-{end}",
            }
        )

        capeshear_cf = Cascade().read(
            {
                "stream": "enfo",
                "levtype": "sfc",
                "param": [228036, 228035],
                "type": "cf",
                "number": range(0, 1),
                "step": range(start, end + 1, 12),
            }
        )
        capeshear = (
            Cascade()
            .read(
                {
                    "stream": "enfo",
                    "levtype": "sfc",
                    "param": [228036, 228035],
                    "type": "pf",
                    "number": range(1, num_ensembles + 1),
                    "step": range(start, end + 1, 12),
                }
            )
            .join(capeshear_cf, "number")
            .reduce(lambda x, y: x if y <= 0 else 0, "param")
        )
        efi_control = (
            capeshear.select("number", 0)
            .reduce(np.maximum, "step")
            .join(climatology, "data_type")
            .reduce("efi")
            .write()
        )
        efi_sot = (
            capeshear.groupby("step")
            .reduce("maximum")
            .join(climatology, "data_type")
        )

        efi = efi_sot.reduce("efi").write()
        sot = efi_sot.reduce("sot").write()

        extreme_graph = efi_control.graph() + efi.graph() + sot.graph()
        pyvis.to_pyvis(extreme_graph, notebook=True, cdn_resources="remote").show(
            f"extreme_graph.html"
        )