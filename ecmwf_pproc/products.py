import numpy as np
import xarray as xr

from ppgraph import Graph, deduplicate_nodes
from cascade.fluent import Node, Payload
from cascade.fluent import custom_hash

from .actions import SingleAction, MultiAction
from .io import retrieve
from .graph_config import (
    Config,
    ParamConfig,
    WindConfig,
    ExtremeConfig,
    ClusterConfig,
    Request,
    MultiSourceRequest,
)


def _read(
    requests: list[Request | MultiSourceRequest], join_key: str = "number", **kwargs
):
    all_actions = None
    for request in requests:
        nodes = np.empty(tuple(request.dims.values()), dtype=object)
        for indices, new_request, name in request.expand():
            payload = Payload(
                retrieve,
                (new_request,),
                kwargs,
            )
            nodes[indices] = Node(
                payload=payload,
                name=f"retrieve@{name}:{custom_hash(str(payload))}",
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
            climatology = _read(
                param_config.clim_request(window, args.climatology)
            )  # Will contain type em and es

            total_graph += (
                _read(param_config.forecast_request(window, args.ensemble))
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
                _read(param_config.forecast_request(window, args.ensemble))
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
            for loc, target in [
                (args.ensemble, f"out_ens_ws"),
                (args.deterministic, f"out_det_ws"),
            ]:
                if loc is None:
                    continue
                vod2uv, requests = param_config.forecast_request(window, loc)
                ws = _read(
                    requests,
                    stream=(not vod2uv),
                ).wind_speed(
                    vod2uv,
                    param_config.get_target(target),
                    param_config.out_keys,
                )
                if loc == args.ensemble:
                    ws = ws.ensms(
                        param_config.get_target("out_mean"),
                        param_config.get_target("out_std"),
                        {**param_config.out_keys, **window.grib_set},
                    )
                total_graph += ws.graph()

    return deduplicate_nodes(total_graph)


def ensms(args):
    config = Config(args)
    total_graph = Graph([])
    for _, cfg in config.options["parameters"].items():
        param_config = ParamConfig(config.members, cfg, config.in_keys, config.out_keys)
        for window in param_config.windows:
            total_graph += (
                _read(param_config.forecast_request(window, args.ensemble))
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
            climatology = _read(
                param_config.clim_request(
                    window, args.climatology, True, no_expand=("quantile")
                )
            )
            parameter = _read(
                param_config.forecast_request(window, args.ensemble)
            ).param_operation(param_config.param_operation)
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
                _read(param_config.forecast_request(window, args.ensemble), "number")
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
    config = ClusterConfig(args)
    if args.spread is not None:
        spread = _read(config.spread_request(args.spread, no_expand=("step",)))
    else:
        spread = _read(
            config.spread_compute_request(args.spread_compute, no_expand=("step",)),
            join_key="date",
        ).mean(key="date")

    pca = (
        _read(config.forecast_request(args.ensemble, no_expand="step"))
        .concatenate(key="number")
        .join(spread, dim_name="data_type")
        .pca(config, args.mask, args.pca)
    )
    cluster = pca.cluster(config, args.ncomp_file, args.indexes, args.deterministic)
    total_graph = (
        pca.join(cluster, dim_name="data_type")
        .attribution(
            config,
            {
                "centroids": (args.centroids, args.cen_anomalies),
                "representatives": (args.representative, args.rep_anomalies),
            },
        )
        .graph()
    )
    return deduplicate_nodes(total_graph)


GRAPHS = [anomaly_prob, prob, ensms, wind, extreme, clustereps, quantiles]
