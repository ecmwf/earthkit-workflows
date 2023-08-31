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

    def extreme(self, climatology: Action):
        # Join with climatology and compute efi control
        ret = self.join(climatology, "datatype").reduce("efi")
        ret.add_attributes(
            {"marsType": "efic", "efiOrder": 0, "totalNumber": 1, "number": 0}
        )
        return ret


class MultiAction(BaseMultiAction):
    def to_single(self, func, node=None):
        return SingleAction(func, self, node)

    def extreme(self, climatology: Action, sot: list):
        # First concatenate across ensemble, and then join
        # with climatology and reduce efi/sot
        with_clim = self.reduce(np.concatenate, "number").join(climatology, "datatype")
        res = None
        for number in [0] + sot:
            if number == 0:
                func = "efi"
                efi_order = 0
            else:
                func = "sot"
                if number == 90:
                    efi_order = 99
                elif number == 10:
                    efi_order == 1
                else:
                    raise Exception(
                        "SOT value '{sot}' not supported in template! Only accepting 10 and 90"
                    )
            new_extreme = with_clim.reduce(func)
            new_extreme.add_node_attributes({"marsType": func, "efiOrder": efi_order})
            new_extreme._add_dimension("number", number)
            if res is None:
                res = new_extreme
            else:
                res = res.join(new_extreme, "number")
        return res

    def ensms(self):
        mean = self.reduce(np.mean, "number")
        mean._add_dimension("type", "em")
        std = self.reduce(np.std, "number")
        std._add_dimension("type", "es")
        res = mean.join(std, "type")
        return res

    def threshold_prob(self, thresholds: list):
        res = None
        for threshold in thresholds:
            threshold_attrs = {}
            threshold_value = threshold["value"]
            if "localDecimalScaleFactor" in threshold:
                scale_factor = threshold["localDecimalScaleFactor"]
                threshold_attrs["localDecimalScaleFactor"] = scale_factor
                threshold_value = round(threshold["value"] * 10**scale_factor, 0)

            comparison = threshold["comparison"]
            if "<" in comparison:
                threshold_attrs["thresholdIndicator"] = 2
                threshold_attrs["upperThreshold"] = threshold_value
            else:
                threshold_attrs["thresholdIndicator"] = 1
                threshold_attrs["lowerThreshold"] = threshold_value

            threshold_func = lambda x: numexpr.evaluate(
                "data " + comparison + str(threshold["value"]),
                local_dict={"data": x},
            )
            # Also need to multiply by 100
            new_threshold_action = self.foreach(threshold_func).reduce(
                np.mean, "number"
            )

            new_threshold_action.add_node_attributes(threshold_attrs)
            new_threshold_action._add_dimension("paramId", threshold["out_paramid"])
            if res is None:
                res = new_threshold_action
            else:
                res = res.join(new_threshold_action, "paramId")

        # Remove expanded dimension if only a single threshold in list
        res._squeeze_dimension("paramId")
        return res

    def anomaly(self, climatology: Action, standardised: bool):
        anomaly = self.join(
            climatology.select({"type": "em"}), match_coord_values=True
        ).reduce(np.subtract)

        if standardised:
            anomaly = anomaly.join(
                climatology.select({"type": "es"}), match_coord_values=True
            ).reduce(np.divide)
        return anomaly

    def quantiles(self, n: int = 100):
        all_ens = self.reduce(np.concatenate, "number")
        res = None
        for index in range(n):
            new_quantile = all_ens.foreach(f"quantiles{index}")
            new_quantile._add_dimension("pertubationNumber", index)
            if res is None:
                res = new_quantile
            else:
                res = res.join(new_quantile, "pertubationNumber")
        return res

    def param_operation(self, operation: str):
        if operation is None:
            return self
        return self.reduce(operation, "param")

    def window_operation(self, window_name: str, operation: str):
        if operation is None:
            self._squeeze_dimension("step")
            return self
        ret = self.reduce(operation, "step")
        ret.add_attributes({"step": window_name})
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
            .window_operation(window.name, window.operation)
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
            .window_operation(window.name, window.operation)
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
            .window_operation(window.name, window.operation)
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
                parameter.select({"number": 0})
                .window_operation(window.name, window.operation)
                .extreme(climatology)
                .write()
                .graph()
            )

        total_graph += (
            parameter.window_operation(window.name, window.operation)
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
            .window_operation(window.name, window.operation)
            .quantiles()
            .write()
            .graph()
        )
    return deduplicate_nodes(total_graph)
