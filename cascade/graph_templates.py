import numpy as np
import xarray as xr
import numexpr

from ppgraph import Graph, Node, deduplicate_nodes

from .fluent import Action
from .fluent import SingleAction as BaseSingleAction
from .fluent import MultiAction as BaseMultiAction


class SingleAction(BaseSingleAction):
    def to_multi(self, nodes):
        return MultiAction(self, nodes)

    def extreme(self, climatology: Action, sot: list):
        # Join with climatology and reduce efi/sot
        with_clim = self.join(climatology, "datatype")

        efi_functions = ["efi"] + [f"sot_{perc}" for perc in sot]
        res = None
        for func in efi_functions:
            new_extreme = with_clim.reduce(func)
            new_extreme.nodes.expand_dims({"type": [func]})
            if res is None:
                res = new_extreme
            else:
                res = res.join(new_extreme, "type")
        return res


class MultiAction(BaseMultiAction):
    def to_single(self, func, nodes=None):
        return SingleAction(func, self, nodes)

    def extreme(self, climatology: Action, sot: list):
        # First concatenate across ensemble, and then join
        # with climatology and reduce efi/sot
        with_clim = self._concatenate("number").join(climatology, "datatype")

        efi_functions = ["efi"] + [f"sot_{perc}" for perc in sot]
        res = None
        for func in efi_functions:
            new_extreme = with_clim.reduce(func)
            new_extreme.nodes.expand_dims({"type": [func]})
            if res is None:
                res = new_extreme
            else:
                res = res.join(new_extreme, "type")
        return res

    def ensms(self):
        mean = self.reduce(np.mean, "number")
        std = self.reduce(np.std, "number")
        res = mean.join(std, xr.DataArray(["em", "es"], name="type"))
        return res

    def threshold_prob(self, thresholds: list):
        res = None
        for threshold in thresholds:
            threshold_func = lambda x: numexpr.evaluate(
                "data " + threshold["comparison"] + str(threshold["value"]),
                local_dict={"data": x},
            )
            # Also need to multiply by 100
            new_threshold_action = self.foreach(threshold_func).reduce(
                np.mean, "number"
            )
            new_threshold_action.nodes.expand_dims(
                {"threshold": [f"{threshold['comparison']}{threshold['value']}"]}
            )
            if res is None:
                res = new_threshold_action
            else:
                res = res.join(new_threshold_action, "threshold")
        return res

    def anomaly(self, climatology: Action, standardised: bool):
        anomaly = self.join(
            climatology.select("type", "em"), match_coord_values=True
        ).reduce(np.subtract)

        if standardised:
            anomaly = anomaly.join(
                climatology.select("type", "es"), match_coord_values=True
            ).reduce(np.divide)
        return anomaly

    def param_operation(self, operation: str):
        if operation is None:
            return self
        return self.reduce(operation, "param")

    def window_operation(self, operation: str):
        if operation is None:
            return self
        return self.reduce(operation, "step")


def read(requests: list, join_key: str = "number"):
    all_actions = None
    for request in requests:
        if len(request.dims) == 0:
            new_action = SingleAction(f"read:{request}", None)
        else:
            nodes = np.empty(tuple(request.dims.values()), dtype=object)
            for indices, new_request in request.expand():
                nodes[indices] = Node(
                    f"{new_request}",
                    payload=f"read:{new_request}",
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
            read(param_config.forecast_request(window))
            .groupby("step")
            .anomaly(climatology, window.options.get("std_anomaly", False))
            .window_operation(window.operation)
            .threshold_prob(window.options.get("thresholds", []))
            .write()
            .graph()
        )

    return total_graph


def prob(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window))
            .param_operation(param_config.param_operation)
            .groupby("step")
            .window_operation(window.operation)
            .threshold_prob(window.options.get("thresholds", []))
            .write()
            .graph()
        )

    return deduplicate_nodes(total_graph)


def wind(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window))
            .param_operation(lambda x, y: np.sqrt(x**2 + y**2))
            .ensms()
            .write()
            .graph()
        )
    return deduplicate_nodes(total_graph)


def ensms(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window))
            .window_operation(window.operation)
            .ensms()
            .write()
            .graph()
        )
    return deduplicate_nodes(total_graph)


def extreme(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        climatology = read(param_config.clim_request(window, True))
        parameter = read(param_config.forecast_request(window)).param_operation(
            param_config.param_operation
        )

        # EFI Control
        if param_config.options.get("efi_control", False):
            total_graph += (
                parameter.select("number", 0)
                .window_operation(window.operation)
                .extreme(climatology, [])
                .write()
                .graph()
            )

        total_graph += (
            parameter.groupby("step")
            .window_operation(window.operation)
            .extreme(climatology, param_config.options["sot"])
            .write()
            .graph()
        )

    return deduplicate_nodes(total_graph)


def quantiles(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window), "number")
            .param_operation(param_config.param_operation)
            .groupby("step")
            .window_operation(window.operation)
            .transform("iter_quantiles", 1)
            .write()
            .graph()
        )
    return deduplicate_nodes(total_graph)
