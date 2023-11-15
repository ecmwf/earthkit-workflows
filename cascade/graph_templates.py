import numpy as np
import xarray as xr

from ppgraph import Graph, deduplicate_nodes
from earthkit.data import FieldList
from pproc.clustereps.config import FullClusterConfig

from .fluent import Action, Node, Payload
from .fluent import SingleAction as BaseSingleAction
from .fluent import MultiAction as BaseMultiAction
from .fluent import custom_hash
from .io import retrieve
from .graph_config import Config, Window, ParamConfig, WindConfig, ExtremeConfig
from . import functions


class SingleAction(BaseSingleAction):
    def to_multi(self, nodes: xr.DataArray):
        return MultiAction(self, nodes)

    def extreme(
        self,
        climatology: Action,
        eps: float,
        num_steps: int,
        target_efi: str = "null:",
        grib_sets: dict = {},
    ):
        # Join with climatology and compute efi control
        payload = Payload(
            functions.efi,
            ("input1", "input0", eps, num_steps),
            {"control": True},
        )
        ret = self.join(climatology, "datatype").reduce(payload)
        return ret.write(target_efi, grib_sets)


class MultiAction(BaseMultiAction):
    def to_single(self, payload: Payload, node: Node = None):
        return SingleAction(payload, self, node)

    def extreme(
        self,
        climatology: Action,
        sot: list,
        eps: float,
        num_steps: int,
        target_efi: str = "null:",
        target_sot: str = "null:",
        grib_sets: dict = {},
    ):
        # First concatenate across ensemble, and then join
        # with climatology and reduce efi/sot
        def _extreme(action, number):
            if number == 0:
                payload = Payload(functions.efi, ("input1", "input0", eps, num_steps))
                target = target_efi
            else:
                payload = Payload(
                    functions.sot, ("input1", "input0", number, eps, num_steps)
                )
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
        mean._add_dimension("marsType", "em")
        mean.write(target_mean, grib_sets)
        std = self.std("number")
        std._add_dimension("marsType", "es")
        std.write(target_std, grib_sets)
        res = mean.join(std, "marsType")
        return res

    def threshold_prob(
        self, thresholds: list, target: str = "null:", grib_sets: dict = {}
    ):
        def _threshold_prob(action, threshold):
            payload = Payload(
                functions.threshold,
                (threshold, "input0", grib_sets.get("edition", 1)),
            )
            new_threshold_action = (
                action.foreach(payload)
                .foreach(
                    Payload(
                        lambda x: FieldList.from_numpy(x.values * 100, x.metadata())
                    )
                )
                .mean("number")
            )
            new_threshold_action._add_dimension("paramId", threshold["out_paramid"])
            return new_threshold_action

        return self.transform(_threshold_prob, thresholds, "paramId").write(
            target, grib_sets
        )

    def anomaly(self, climatology: Action, window: Window):
        extract = (
            ("climateDateFrom", "climateDateTo", "referenceDate")
            if window.grib_set.get("edition", 1) == 2
            else ()
        )

        anom = self.join(
            climatology.select({"type": "em"}), "datatype", match_coord_values=True
        ).subtract(extract_keys=extract)

        if window.options.get("std_anomaly", False):
            anom = anom.join(
                climatology.select({"type": "es"}), "datatype", match_coord_values=True
            ).divide()
        return anom

    def quantiles(self, n: int = 100, target: str = "null:", grib_sets: dict = {}):
        def _quantiles(action, quantile):
            payload = Payload(functions.quantiles, ("input0", quantile))
            if isinstance(action, BaseSingleAction):
                new_quantile = action.then(payload)
            else:
                new_quantile = action.foreach(payload)
            new_quantile._add_dimension("perturbationNumber", quantile)
            return new_quantile

        return (
            self.concatenate("number")
            .transform(_quantiles, np.linspace(0.0, 1.0, n + 1), "perturbationNumber")
            .write(target, grib_sets)
        )

    def wind_speed(self, vod2uv: bool, target: str = "null:", grib_sets={}):
        if vod2uv:
            ret = self.foreach(Payload(functions.norm, ("input0",)))
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
            ret = self
        else:
            ret = getattr(self, window.operation)("step")
        if window.end - window.start == 0:
            ret.add_attributes({"step": window.name})
        else:
            ret.add_attributes({"stepRange": window.name})
        return ret.write(target, grib_sets)

    def pca(self, config, mask, target: str = None):
        return self.reduce(
            Payload(functions.pca, (config, "input0", "input1", mask, target))
        )

    def cluster(self, config, ncomp_file, indexes, deterministic):
        return self.foreach(
            Payload(
                functions.cluster,
                (config, "input0", ncomp_file, indexes, deterministic),
            )
        )

    def attribution(self, config, targets):
        def _attribution(action, scenario):
            payload = Payload(
                functions.attribution, (config, scenario, "input0", "input1")
            )
            if isinstance(action, BaseSingleAction):
                new_quantile = action.then(payload)
            else:
                new_quantile = action.foreach(payload)
            new_quantile._add_dimension("scenario", scenario)
            return new_quantile

        return self.transform(
            _attribution, ["centroids", "representatives"], "scenario"
        ).foreach(
            np.asarray(
                [
                    Payload(
                        functions.cluster_write,
                        (config, "centroids", "input0", targets["centroids"]),
                    ),
                    Payload(
                        functions.cluster_write,
                        (
                            config,
                            "representatives",
                            "input0",
                            targets["representatives"],
                        ),
                    ),
                ]
            )
        )


def read(requests: list, join_key: str = "number", **kwargs):
    all_actions = None
    for request in requests:
        nodes = np.empty(tuple(request.dims.values()), dtype=object)
        for indices, new_request in request.expand():
            payload = Payload(
                retrieve,
                (new_request.pop("source"), new_request),
                kwargs,
            )
            nodes[indices] = Node(
                payload=payload,
                name=f"retrieve@{new_request['type']}:{custom_hash(str(payload))}",
            )
        if len(request.dims) == 0:
            new_action = SingleAction(
                payload=None, previous=None, node=xr.DataArray(nodes[()])
            )
        else:
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


def anomaly_prob(args):
    config = Config(args)
    total_graph = Graph([])
    for _, cfg in config.options["parameters"].items():
        param_config = ParamConfig(config.members, cfg, config.in_keys, config.out_keys)
        for window in param_config.windows:
            climatology = read(
                param_config.clim_request(window)
            )  # Will contain type em and es

            total_graph += (
                read(param_config.forecast_request(window))
                .anomaly(climatology, window)
                .window_operation(
                    window,
                    param_config.get_target("out_ensemble"),
                    param_config.out_keys,
                )
                .threshold_prob(
                    window.options.get("thresholds", []),
                    param_config.get_target("out_prob"),
                    {**param_config.out_keys, **window.grib_set},
                )
                .graph()
            )

    return deduplicate_nodes(total_graph)


def prob(args):
    config = Config(args)
    total_graph = Graph([])
    for _, cfg in config.options["parameters"].items():
        param_config = ParamConfig(config.members, cfg, config.in_keys, config.out_keys)
        for window in param_config.windows:
            total_graph += (
                read(param_config.forecast_request(window))
                .param_operation(param_config.param_operation)
                .window_operation(
                    window,
                    param_config.get_target("out_ensemble"),
                    param_config.out_keys,
                )
                .threshold_prob(
                    window.options.get("thresholds", []),
                    param_config.get_target("out_prob"),
                    {**param_config.out_keys, **window.grib_set},
                )
                .graph()
            )

    return deduplicate_nodes(total_graph)


def wind(args):
    config = Config(args)
    total_graph = Graph([])
    for _, cfg in config.options["parameters"].items():
        param_config = WindConfig(config.members, cfg, config.in_keys, config.out_keys)
        for window in param_config.windows:
            for source in param_config.sources:
                vod2uv = param_config.vod2uv(source)
                total_graph = (
                    read(
                        param_config.forecast_request(window, source),
                        stream=(not vod2uv),
                    )
                    .wind_speed(
                        vod2uv,
                        param_config.get_target(f"out_{source}_ws"),
                        param_config.out_keys,
                    )
                    .ensms(
                        param_config.get_target("out_mean"),
                        param_config.get_target("out_std"),
                        {**param_config.out_keys, **window.grib_set},
                    )
                    .graph()
                )

    return deduplicate_nodes(total_graph)


def ensms(args):
    config = Config(args)
    total_graph = Graph([])
    for _, cfg in config.options["parameters"].items():
        param_config = ParamConfig(config.members, cfg, config.in_keys, config.out_keys)
        for window in param_config.windows:
            total_graph += (
                read(param_config.forecast_request(window))
                .window_operation(window)
                .ensms(
                    param_config.get_target("out_mean"),
                    param_config.get_target("out_std"),
                    {**param_config.out_keys, **window.grib_set},
                )
                .graph()
            )
    return deduplicate_nodes(total_graph)


def extreme(args):
    config = Config(args)
    total_graph = Graph([])
    for _, cfg in config.options["parameters"].items():
        param_config = ExtremeConfig(
            config.members, cfg, config.in_keys, config.out_keys
        )
        for window in param_config.windows:
            climatology = read(
                param_config.clim_request(window, True, no_expand=("quantile"))
            )
            parameter = read(param_config.forecast_request(window)).param_operation(
                param_config.param_operation
            )
            eps = float(param_config.options["eps"])
            grib_sets = {**param_config.out_keys, **window.grib_set}

            # EFI Control
            if param_config.options.get("efi_control", False):
                total_graph += (
                    parameter.select({"number": 0})
                    .window_operation(window)
                    .extreme(
                        climatology,
                        eps,
                        len(window.steps),
                        param_config.get_target(f"out_efi"),
                        grib_sets,
                    )
                    .graph()
                )

            total_graph += (
                parameter.window_operation(window)
                .extreme(
                    climatology,
                    list(map(int, param_config.options["sot"])),
                    eps,
                    len(window.steps),
                    param_config.get_target(f"out_efi"),
                    param_config.get_target(f"out_sot"),
                    grib_sets,
                )
                .graph()
            )

    return deduplicate_nodes(total_graph)


def quantiles(args):
    config = Config(args)
    total_graph = Graph([])
    for _, cfg in config.options["parameters"].items():
        param_config = ParamConfig(config.members, cfg, config.in_keys, config.out_keys)
        for window in param_config.windows:
            total_graph += (
                read(param_config.forecast_request(window), "number")
                .param_operation(param_config.param_operation)
                .window_operation(window)
                .quantiles(
                    param_config.options["num_quantiles"],
                    target=param_config.get_target("out_quantiles"),
                    grib_sets={**param_config.out_keys, **window.grib_set},
                )
                .graph()
            )
    return deduplicate_nodes(total_graph)


def clustereps(args):
    config = FullClusterConfig(args)
    spread = read(config.spread_request(args.spread))
    pca = read(config.forecast_request()).join(spread).pca(config, args.mask, args.pca)
    cluster = pca.cluster(config, args.ncomp_file, args.indexes, args.deterministic)
    total_graph += (
        pca.join(cluster)
        .attribution(
            config,
            {
                "centroids": (args.centroids, args.cen_anomalies),
                "representatives": (args.representative, args.rep_anomalies),
            },
        )
        .graph()
    )
