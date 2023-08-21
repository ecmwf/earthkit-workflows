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
        clim_em = climatology.select("type", "em")
        clim_es = climatology.select("type", "es")

        anomaly = (
            read(param_config.forecast_request(window), "number")
            .groupby("step")
            .join(clim_em, match_coord_values=True)
            .reduce(np.subtract)
        )

        if window.options.get("std_anomaly", False):
            anomaly = anomaly.join(clim_es, match_coord_values=True).reduce(np.divide)
        if window.operation is not None:
            anomaly = anomaly.reduce(window.operation)
        for threshold in window.thresholds():
            total_graph += anomaly.map(threshold).reduce(np.mean).write().graph()

        return total_graph


def prob(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        prob = read(param_config.forecast_request(window))

        # Multi-parameter processing operation
        if param_config.param_operation is not None:
            prob = prob.reduce(param_config.param_operation, "param")

        prob = prob.groupby("step")
        if window.operation is not None:
            prob = prob.reduce(window.operation)
        for threshold in window.thresholds():
            total_graph += prob.map(threshold).reduce(np.mean).write().graph()

    return total_graph


def wind(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        wind_speed = read(param_config.forecast_request(window), "number").reduce(
            lambda x, y: np.sqrt(x**2 + y**2), "param"
        )
        mean = wind_speed.reduce(np.mean, "number").write()
        std = wind_speed.reduce(np.std, "number").write()
        total_graph += mean.graph() + std.graph()
    return total_graph


def ensms(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        ensms = read(param_config.forecast_request(window), "number")
        if window.operation is not None:
            ensms = ensms.reduce(window.operation, "step")
        mean = ensms.reduce(np.mean, "number").write()
        std = ensms.reduce(np.std, "number").write()
        total_graph += mean.graph() + std.graph()
    return total_graph


def extreme(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        climatology = read(param_config.clim_request(window, True))
        parameter = read(param_config.forecast_request(window), "number")

        # Multi-parameter processing operation
        if param_config.param_operation is not None:
            parameter = parameter.reduce(param_config.param_operation, "param")

        total_graph += (
            parameter.select("number", 0)
            .reduce(window.operation, "step")
            .join(climatology, "data_type")
            .reduce("efi")
            .write()
            .graph()
        )

        efi_sot = (
            parameter.groupby("step")
            .reduce(window.operation)
            .join(climatology, "data_type")
        )
        total_graph += efi_sot.reduce("efi").write().graph()

        for perc in param_config.options["sot"]:
            total_graph += efi_sot.reduce(f"sot_{perc}").write().graph()

    return total_graph
