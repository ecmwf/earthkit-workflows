import numpy as np
import xarray as xr

from ppgraph import Graph, deduplicate_nodes
from earthkit.data import FieldList

from .fluent import Action, Node
from .fluent import SingleAction as BaseSingleAction
from .fluent import MultiAction as BaseMultiAction
from .graph_config import ThresholdConfig
from .io import retrieve
from .graph_config import WindConfig
from . import functions


class SingleAction(BaseSingleAction):
    def to_multi(self, nodes):
        return MultiAction(self, nodes)

    def extreme(
        self,
        climatology: Action,
        eps: float,
        target_efi: str = "null:",
        grib_sets: dict = {},
    ):
        # Join with climatology and compute efi control
        payload = (functions.efi, ("input1", "input0", eps), {"control": True})
        ret = self.join(climatology, "datatype").reduce(payload)
        return ret.write(target_efi, grib_sets)


class MultiAction(BaseMultiAction):
    def to_single(self, payload, node=None):
        return SingleAction(payload, self, node)

    def extreme(
        self,
        climatology: Action,
        sot: list,
        eps: float,
        target_efi: str = "null:",
        target_sot: str = "null:",
        grib_sets: dict = {},
    ):
        # First concatenate across ensemble, and then join
        # with climatology and reduce efi/sot
        def _extreme(action, number):
            if number == 0:
                payload = (functions.efi, ("input1", "input0", eps))
                target = target_efi
            else:
                payload = (functions.sot, ("input1", "input0", number, eps))
                target = target_sot
            new_extreme = action.reduce(payload)
            new_extreme._add_dimension("number", number)
            return new_extreme.write(target, grib_sets)

        return (
            self.concatenate("number")
            .join(climatology, "datatype")
            .transform(_extreme, [0] + sot, "number")
        )

    def ensms(
        self, target_mean: str = "null:", target_std: str = "null:", grib_sets={}
    ):
        mean = self.mean("number")
        mean._add_dimension("type", "em")
        mean.write(target_mean, grib_sets)
        std = self.std("number")
        std._add_dimension("type", "es")
        std.write(target_std, grib_sets)
        res = mean.join(std, "type")
        return res

    def threshold_prob(
        self, thresholds: list, target: str = "null:", grib_sets: dict = {}
    ):
        def _threshold_prob(action, threshold):
            threshold_config = ThresholdConfig(threshold)
            payload = (
                functions.threshold,
                (
                    threshold_config.comparison,
                    threshold_config.threshold,
                    "input0",
                    threshold_config.grib_keys,
                ),
            )
            new_threshold_action = (
                action.foreach(payload)
                .foreach(lambda x: FieldList.from_numpy(x.values * 100, x.metadata()))
                .mean("number")
            )
            new_threshold_action._add_dimension("paramId", threshold["out_paramid"])
            return new_threshold_action

        return self.transform(_threshold_prob, thresholds, "paramId").write(
            target, grib_sets
        )

    def anomaly(self, climatology: Action, standardised: bool):
        anomaly = self.join(
            climatology.select({"type": "em"}), "datatype", match_coord_values=True
        ).subtract()

        if standardised:
            anomaly = anomaly.join(
                climatology.select({"type": "es"}), "datatype", match_coord_values=True
            ).divide()
        return anomaly

    def quantiles(self, n: int = 100, target: str = "null:", grib_sets: dict = {}):
        def _quantiles(action, quantile):
            new_quantile = action.then(("meteokit.stats.quantiles", "input0", quantile))
            new_quantile._add_dimension("pertubationNumber", quantile)
            return new_quantile

        return (
            self.concatenate("number")
            .transform(_quantiles, range(n), "perturbationNumber")
            .write(target, grib_sets)
        )

    def wind_speed(self, vod2uv: bool, target: str = "null:", grib_sets={}):
        if vod2uv:
            ret = self.foreach((functions.wind_speed, ("input0",)))
        else:
            ret = self.param_operation("norm")
        return ret.write(target, grib_sets)

    def param_operation(self, operation: str):
        if operation is None:
            return self
        if isinstance(operation, str):
            return getattr(self, operation)("param")
        return self.reduce(operation, "param")

    def window_operation(self, window, target: str = "null:", grib_sets: dict = {}):
        if window.operation is None:
            self._squeeze_dimension("step")
            return self
        ret = getattr(self, window.operation)("step")
        ret.add_attributes({"stepRange": window.name})
        return ret.write(target, grib_sets)


def read(requests: list, join_key: str = "number", **kwargs):
    all_actions = None
    for request in requests:
        if len(request.dims) == 0:
            new_action = SingleAction(
                payload=(
                    retrieve,
                    (request.pop("source"), request.request),
                    kwargs,
                ),
                previous=None,
            )
        else:
            nodes = np.empty(tuple(request.dims.values()), dtype=object)
            for indices, new_request in request.expand():
                nodes[indices] = Node(
                    payload=(
                        retrieve,
                        (new_request.pop("source"), new_request),
                        kwargs,
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
            .window_operation(
                window, param_config.get_target("out_ensemble"), param_config.out_keys
            )
            .threshold_prob(
                window.options.get("thresholds", []),
                param_config.get_target("out_prob"),
                {**window.grib_set, **param_config.out_keys},
            )
            .graph()
        )

    return deduplicate_nodes(total_graph)


def prob(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window))
            .param_operation(param_config.param_operation)
            .window_operation(
                window, param_config.get_target("out_ensemble"), param_config.out_keys
            )
            .threshold_prob(
                window.options.get("thresholds", []),
                param_config.get_target("out_prob"),
                {**window.grib_set, **param_config.out_keys},
            )
            .graph()
        )

    return deduplicate_nodes(total_graph)


def wind(param_config: WindConfig):
    total_graph = Graph([])
    for window in param_config.windows:
        for source in param_config.sources:
            vod2uv = param_config.vod2uv(source)
            total_graph = (
                read(param_config.forecast_request(window, source), stream=(not vod2uv))
                .wind_speed(
                    vod2uv,
                    param_config.get_target(f"out_{source}_ws"),
                    param_config.out_keys,
                )
                .ensms(
                    param_config.get_target("out_mean"),
                    param_config.get_target("out_std"),
                    {**window.grib_set, **param_config.out_keys},
                )
                .graph()
            )

    return deduplicate_nodes(total_graph)


def ensms(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        total_graph += (
            read(param_config.forecast_request(window))
            .window_operation(window)
            .ensms(
                param_config.get_target("out_mean"),
                param_config.get_target("out_std"),
                {**window.grib_set, **param_config.out_keys},
            )
            .graph()
        )
    return deduplicate_nodes(total_graph)


def extreme(param_config):
    total_graph = Graph([])
    for window in param_config.windows:
        climatology = read(
            param_config.clim_request(window, True, no_expand=("quantile"))
        )
        parameter = read(param_config.forecast_request(window)).param_operation(
            param_config.param_operation
        )
        eps = float(param_config.options["eps"])
        grib_sets = {**window.grib_set, **param_config.out_keys}

        # EFI Control
        if param_config.options.get("efi_control", False):
            total_graph += (
                parameter.select({"number": 0})
                .window_operation(window)
                .extreme(
                    climatology, eps, param_config.get_target(f"out_efi"), grib_sets
                )
                .graph()
            )

        total_graph += (
            parameter.window_operation(window)
            .extreme(
                climatology,
                list(map(int, param_config.options["sot"])),
                eps,
                param_config.get_target(f"out_efi"),
                param_config.get_target(f"out_sot"),
                grib_sets,
            )
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
            .quantiles(
                target=param_config.get_target("out_quantiles"),
                grib_sets={**window.grib_set, **param_config.out_keys},
            )
            .graph()
        )
    return deduplicate_nodes(total_graph)
