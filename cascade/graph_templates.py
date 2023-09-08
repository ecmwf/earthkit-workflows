import numpy as np
import xarray as xr

from ppgraph import Graph, deduplicate_nodes

from .fluent import Action, Node
from .fluent import SingleAction as BaseSingleAction
from .fluent import MultiAction as BaseMultiAction
from .graph_config import threshold_config, extreme_config
from .io import retrieve


class SingleAction(BaseSingleAction):
    def to_multi(self, nodes):
        return MultiAction(self, nodes)

    def extreme(self, climatology: Action, eps: float):
        # Join with climatology and compute efi control
        ret = self.join(climatology, "datatype").reduce(
            ("meteokit.extreme.efi", "input1", "input0", eps)
        )
        ret.add_attributes(
            {"marsType": "efic", "efiOrder": 0, "totalNumber": 1, "number": 0}
        )
        return ret


class MultiAction(BaseMultiAction):
    def to_single(self, func, node=None):
        return SingleAction(func, self, node)

    def extreme(self, climatology: Action, sot: list, eps: float):
        # First concatenate across ensemble, and then join
        # with climatology and reduce efi/sot
        def _extreme(action, number):
            extreme_type, efi_keys = extreme_config(number)
            if extreme_type == "efi":
                new_extreme = action.reduce(
                    ("meteokit.extreme.efi", "input1", "input0", eps)
                )
            else:
                new_extreme = action.reduce(
                    ("meteokit.extreme.sot", "input1", "input0", number, eps)
                )
            new_extreme.add_node_attributes(Node.Attributes.GRIB_KEYS, efi_keys)
            new_extreme._add_dimension("number", number)
            return new_extreme

        return (
            self.concatenate("number")
            .join(climatology, "datatype")
            .transform(_extreme, [0] + sot, "number")
        )

    def ensms(self):
        mean = self.mean("number")
        mean._add_dimension("type", "em")
        std = self.std("number")
        std._add_dimension("type", "es")
        res = mean.join(std, "type")
        return res

    def threshold_prob(self, thresholds: list):
        def _threshold_prob(action, threshold):
            threshold_func, threshold_keys = threshold_config(threshold)
            new_threshold_action = (
                action.foreach(threshold_func)
                .foreach(lambda x: xr.DataArray(x * 100, attrs=x.attrs))
                .mean("number")
            )
            new_threshold_action.add_node_attributes(
                Node.Attributes.GRIB_KEYS, threshold_keys
            )
            new_threshold_action._add_dimension("paramId", threshold["out_paramid"])
            return new_threshold_action

        return self.transform(_threshold_prob, thresholds, "paramId")

    def anomaly(self, climatology: Action, standardised: bool):
        anomaly = self.join(
            climatology.select({"type": "em"}), match_coord_values=True
        ).subtract()

        if standardised:
            anomaly = anomaly.join(
                climatology.select({"type": "es"}), match_coord_values=True
            ).divide()
        return anomaly

    def quantiles(self, n: int = 100):
        def _quantiles(action, quantile):
            new_quantile = action.then(("meteokit.stats.quantiles", "input0", quantile))
            new_quantile._add_dimension("pertubationNumber", quantile)
            return new_quantile

        return self.concatenate("number").transform(
            _quantiles, range(n), "perturbationNumber"
        )

    def param_operation(self, operation: str):
        if operation is None:
            return self
        if isinstance(operation, str):
            return getattr(self, operation)("param")
        return self.reduce(operation, "param")

    def window_operation(self, window):
        if window.operation is None:
            self._squeeze_dimension("step")
            return self
        ret = getattr(self, window.operation)("step")
        ret.add_attributes({"stepRange": window.name})
        return ret


def read(requests: list, join_key: str = "number"):
    all_actions = None
    for request in requests:
        if len(request.dims) == 0:
            new_action = SingleAction(f"read:{request}", None)
        else:
            nodes = np.empty(tuple(request.dims.values()), dtype=object)
            for indices, new_request in request.expand():
                nodes[indices] = Node(
                    payload=(
                        retrieve,
                        new_request,
                    ),
                    name=f"{new_request}",
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
    return all_actions


def anomaly_prob(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        climatology = read(
            param_config.clim_request(window)
        )  # Will contain type em and es

        total_graph += (
            read(param_config.forecast_request(window))
            .anomaly(climatology, window.options.get("std_anomaly", False))
            .window_operation(window)
            .threshold_prob(window.options.get("thresholds", []))
            .write(param_config.target, window.options.get("grib_set", {}))
            .graph()
        )

    return deduplicate_nodes(total_graph)


def prob(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window))
            .param_operation(param_config.param_operation)
            .window_operation(window)
            .threshold_prob(window.options.get("thresholds", []))
            .write(param_config.target, window.options.get("grib_set", {}))
            .graph()
        )

    return deduplicate_nodes(total_graph)


def wind(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window))
            .param_operation("norm")
            .ensms()
            .write(param_config.target, window.options.get("grib_set", {}))
            .graph()
        )
    return deduplicate_nodes(total_graph)


def ensms(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window))
            .window_operation(window)
            .ensms()
            .write(param_config.target, window.options.get("grib_set", {}))
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
        eps = float(param_config.options["eps"])
        grib_sets = window.options.get("grib_set", {})

        # EFI Control
        if param_config.options.get("efi_control", False):
            total_graph += (
                parameter.select({"number": 0})
                .window_operation(window)
                .extreme(climatology, eps)
                .write(param_config.target, grib_sets)
                .graph()
            )

        total_graph += (
            parameter.window_operation(window)
            .extreme(climatology, list(map(int, param_config.options["sot"])), eps)
            .write(param_config.target, grib_sets)
            .graph()
        )

    return deduplicate_nodes(total_graph)


def quantiles(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window), "number")
            .param_operation(param_config.param_operation)
            .window_operation(window)
            .quantiles()
            .write(param_config.target, window.options.get("grib_set", {}))
            .graph()
        )
    return deduplicate_nodes(total_graph)
