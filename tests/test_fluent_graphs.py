import functools
import numpy as np
import itertools
import xarray as xr
from collections import OrderedDict
import datetime
import sys
import numexpr

from ppgraph import Node, Graph
from pproc import common

from cascade.fluent import SingleAction, MultiAction


def read(request: dict):
    print(request)


class Window(common.window.Window):
    def __init__(self, window_options, operation, include_init):
        super().__init__(window_options, include_init)
        self.operation = operation
        self.window_options = window_options

    def thresholds(self):
        for threshold_options in self.window_options["threshold"]:
            yield lambda x: numexpr.evaluate(
                "data "
                + threshold_options["comparison"]
                + str(threshold_options["threshold"]),
                local_dict={"data": x},
            )

    @property
    def is_std_anomaly(self):
        return "std_anomaly" in self.window_options


class Config(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.fc_date = datetime.strptime(str(self.options["fc_date"]), "%Y%m%d%H")
        if isinstance(self.options["members"], dict):
            members = range(
                self.options["members"]["start"], self.options["members"]["end"] + 1
            )
        else:
            members = range(1, int(self.options["members"]) + 1)
        self.parameters = {
            param: ParamConfig(self.fc_date, members, config)
            for param, config in self.options["parameters"]
        }


class ParamConfig:
    def __init__(self, date: datetime.datetime, members, param_config):
        self.steps = self._generate_steps(param_config.get("steps", []))
        self.interpolation_keys = param_config.get("interpolation_keys", None)
        self.base_request = param_config["base_request"]
        self.base_request["date"] = date.strftime("%Y%m%d")
        self.base_request["time"] = date.strftime("%H")
        self.members = members
        self.climatology = param_config.get("climatology", None)
        self.windows = self._generate_windows(param_config["windows"])
        self.param_operation = self._generate_param_operation(param_config)

    @classmethod
    def _generate_steps(cls, steps_config):
        steps = set()
        for steps in steps_config:
            start_step = steps["start_step"]
            end_step = steps["end_step"]
            interval = steps["interval"]
            range_len = steps.get("range", None)

            if range_len is None:
                for step in range(start_step, end_step + 1, interval):
                    if step not in steps:
                        steps.add(step)
            else:
                for sstep in range(start_step, end_step - range_len + 1, interval):
                    steps.add(common.steps.Step(sstep, sstep + range_len))
        return sorted(steps)

    @classmethod
    def _generate_windows(cls, windows_config):
        windows = []
        for window_type in windows_config:
            include_init = window_type.get("include_start_step", False)
            operation = window_type.get("window_operation", "none")
            for window in window_type["periods"]:
                windows.append(Window(window, operation, include_init))
            # Need to add thresholds and flag for standardised anomaly window
        return windows

    @classmethod
    def _generate_param_operation(param_config):
        if "input_filter_operation" in param_config:
            filter_configs = param_config["input_filter_operation"]
            return (
                lambda x, y, filter_configs=filter_configs: x
                if numexpr.evaluate(
                    "data "
                    + filter_configs["comparison"]
                    + str(filter_configs["threshold"]),
                    local_dict={"data": y},
                )
                else filter_configs["replacement"]
            )
        return param_config.get("input_combine_operation", None)

    def _request_steps(self, window):
        if len(self.steps) == 0:
            return window.steps
        return self.steps[
            self.steps.index(window.start) : self.steps.index(window.end) + 1
        ]

    def request(self, window):
        if isinstance(self.base_request, dict):
            req = self.base_request.copy()
            req["steps"] = self._request_steps(window)
            window_requests = [req]
        else:
            window_requests = []
            for request in self.base_request:
                req = request.copy()
                req["steps"] = self._request_steps(window)
                if request["type"] == "pf":
                    req["number"] = self.members
                window_requests.append(req)
        return window_requests, self.interpolation_keys

    def clim_request(self, window, accumulated: bool = False):
        clim_request = self.base_request.copy()
        clim_request.update(self.climatology["clim_keys"])
        if "quantile" in clim_request:
            num_quantiles = clim_request["quantile"]
            clim_request["quantile"] = [
                "{}:100".format(i) for i in range(num_quantiles)
            ]
        if accumulated:
            clim_request["step"] = self.climatology.get("steps", {}).get(
                window.name, window.name
            )
        else:
            clim_request["step"] = self._request_steps(window)
        return clim_request, self.interpolation_keys


class Cascade:
    def read(self, request: dict, join_key: str):
        # For loop over requests? But then how to combine at the end,
        # and maybe only relevant for cf and pf. For a number dimension on
        # cf?
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

    def graph(self, product: str, config: Config):
        total_graph = Graph()
        for _, param_config in config.parameters:
            total_graph += getattr(self, product)(config.members, param_config)

        return total_graph

    def anomaly_prob(self, param_config):
        total_graph = Graph([])
        for window in param_config.windows:
            climatology = self.read(
                param_config.clim_request(window)
            )  # Will contain type em and es
            clim_em = climatology.select("type", "em")
            clim_es = climatology.select("type", "es")

            anomaly = (
                self.read(param_config.request(window))
                .groupby("step")
                .join(clim_em)
                .reduce(np.subtract)
            )

            if window.is_std_anomaly:
                anomaly.join(clim_es).reduce(np.division)

            for threshold in window.thresholds():
                total_graph += (
                    anomaly.reduce(window.operation)
                    .map(threshold)
                    .reduce(np.mean)
                    .write()
                    .graph()
                )

        return total_graph

    def prob(self, param_config):
        total_graph = Graph([])
        for window in param_config.windows:
            prob = self.read(param_config.request(window))

            # Multi-parameter processing operation
            if param_config.param_operation is not None:
                prob = prob.reduce(param_config.param_operation, "param")

            for threshold in window.thresholds():
                total_graph += (
                    prob.groupby("step")
                    .reduce(window.operation)
                    .map(threshold)
                    .reduce(np.mean)
                    .write()
                    .graph()
                )

        return total_graph

    def wind(self, param_config):
        total_graph = Graph()
        for window in param_config.windows:
            wind_speed = self.read(param_config.request(window)).reduce(
                lambda x, y: np.sqrt(x**2 + y**2), "param"
            )
            mean = wind_speed.reduce(np.mean, "number").write()
            std = wind_speed.reduce(np.std, "number").write()
            total_graph += mean.graph() + std.graph()
        return total_graph

    def ensms(self, param_config):
        total_graph = Graph()
        for window in param_config.windows:
            ensms = self.read(param_config.request(window))
            mean = ensms.reduce(np.mean, "number").write()
            std = ensms.reduce(np.std, "number").write()
            total_graph += mean.graph() + std.graph()
        return total_graph

    def extreme(self, param_config):
        total_graph = Graph()
        for window in param_config.windows:
            climatology = Cascade().read(param_config.clim_request(window, True))
            parameter = Cascade().read(param_config.request(window))

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

            for perc in param_config.sot:
                total_graph += efi_sot.reduce(f"sot_{perc}").write().graph()

        return total_graph


def test_graph_construction(args):
    parser = common.default_parser()
    parser.add_argument(
        "--product",
        type=str,
        choices=["wind", "ensms", "extreme", "prob", "anomaly_prob"],
    )
    args = parser.parse_args(args)
    cfg = Config(args)

    Cascade.graph(args.product, cfg)


if __name__ == "__main__":
    test_graph_construction(sys.args[1:])
