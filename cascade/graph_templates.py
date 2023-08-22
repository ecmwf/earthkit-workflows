import numpy as np
import functools
import xarray as xr

from ppgraph import Graph, Node

from .fluent import SingleAction, MultiAction


def read(request: dict):
    print(request)


def read(requests: list, join_key: str = ""):
    all_actions = None
    for request in requests:
        if len(request.dims) == 0:
            new_action = SingleAction(functools.partial(read, request), None)
        else:
            nodes = np.empty(tuple(request.dims.values()), dtype=object)
            for indices, new_request in request.expand():
                read_with_request = functools.partial(read, new_request)
                nodes[indices] = Node(
                    f"{new_request}",
                    payload=read_with_request,
                )
            new_action = MultiAction(
                None,
                xr.DataArray(
                    nodes,
                    dims=request.dims.keys(),
                    coords={key: list(request[key]) for key in request.dims.keys()},
                ),
            )

        if all_actions is None:
            all_actions = new_action
        else:
            assert len(join_key) != 0
            all_actions = all_actions.join(new_action, join_key)
    # Currently using xarray for keeping track of dimensions but these should
    # belong in node attributes on the graph?
    return all_actions


def anomaly_prob(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        climatology = read(
            param_config.clim_request(window)
        )  # Will contain type em and es

        total_graph += (
            read(param_config.forecast_request(window), "number")
            .groupby("step")
            .join(climatology.select("type", "em"), match_coord_values=True)
            .reduce(np.subtract)
            .applyif(
                window.options.get("std_anomaly", False),
                "join",
                climatology.select("type", "es"),
                match_coord_values=True,
            )
            .applyif(window.options.get("std_anomaly", False), "reduce", np.divide)
            .applyif(window.operation is not None, "reduce", window.operation)
            .fork("map", [(threshold,) for threshold in window.thresholds()])
            .reduce(np.mean)
            .write()
            .graph()
        )

    return total_graph


def prob(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window))
            .applyif(
                param_config.param_operation is not None,
                "reduce",
                param_config.param_operation,
                "param",
            )
            .groupby("step")
            .applyif(window.operation is not None, "reduce", window.operation)
            .fork("map", [(threshold,) for threshold in window.thresholds()])
            .reduce(np.mean)
            .write()
            .graph()
        )

    return total_graph


def wind(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window), "number")
            .reduce(lambda x, y: np.sqrt(x**2 + y**2), "param")
            .fork("reduce", [(np.mean, "number"), (np.std, "number")])
            .write()
            .graph()
        )
    return total_graph


def ensms(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window), "number")
            .applyif(window.operation is not None, "reduce", window.operation, "step")
            .fork("reduce", [(np.mean, "number"), (np.std, "number")])
            .write()
            .graph()
        )
    return total_graph


def extreme(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        climatology = read(param_config.clim_request(window, True))
        parameter = read(param_config.forecast_request(window), "number").applyif(
            param_config.param_operation is not None,
            "reduce",
            param_config.param_operation,
            "param",
        )

        # EFI Control
        if param_config.options.get("efi_control", False):
            total_graph += (
                parameter.select("number", 0)
                .reduce(window.operation, "step")
                .join(climatology, "data_type")
                .reduce("efi")
                .write()
                .graph()
            )

        total_graph += (
            parameter.groupby("step")
            .reduce(window.operation)
            .join(climatology, "data_type")
            .fork(
                "reduce",
                [("efi",), *[(f"sot_{perc}",) for perc in param_config.options["sot"]]],
            )
            .write()
            .graph()
        )

    return total_graph
