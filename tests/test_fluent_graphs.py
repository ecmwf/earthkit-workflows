import functools
import numpy as np
import itertools
import xarray as xr
from collections import OrderedDict
import pytest
import numexpr
import yaml
import bisect

from ppgraph import Node, Graph
from ppgraph import pyvis

from cascade.fluent import SingleAction, MultiAction


def read(request: dict):
    print(request)


class Window:
    def __init__(self, window_range, operation, include_init, window_options):
        self.start = int(window_range[0])
        self.end = int(window_range[1])
        window_size = self.end - self.start
        self.include_init = include_init or (window_size == 0)
        if window_size == 0:
            self.name = str(self.end)
        else:
            self.name = f"{self.start}-{self.end}"

        self.step = (
            int(window_range[2]) if len(window_range) > 2 else 1
        )
        if self.include_init:
            self.steps = list(range(self.start, self.end + 1, self.step))
        else:
            self.steps = list(range(self.start + self.step, self.end + 1, self.step))

        self.operation = operation
        self.window_options = window_options

    def thresholds(self):
        for threshold_options in self.window_options["thresholds"]:
            yield lambda x: numexpr.evaluate(
                "data "
                + threshold_options["comparison"]
                + str(threshold_options["threshold"]),
                local_dict={"data": x},
            )

    @property
    def is_std_anomaly(self):
        return "std_anomaly" in self.window_options
    
class Request(dict):
    def __init__(self, request: dict):
        super().__init__()
        self.update(request)
        self.fake_dims = []

    @property
    def dims(self):
        dimensions = OrderedDict()
        for key, values in self.items():
            if key == "interpolate":
                continue
            if hasattr(values, "__iter__") and not isinstance(values, str):
                dimensions[key] = len(values)
        return dimensions
    
    def make_dim(self, key, value = None):
        if key in self:
            assert type(self[key], (str, int, float))
            self[key] = [self[key]]
        else:
            self[key] = [value]
            self.fake_dims.append(key)

    def expand(self):
        for params in itertools.product(*[self[x] for x in self.dims.keys()]):
            new_request = super().copy()
            indices = []
            for index, expand_param in enumerate(self.dims.keys()):
                new_request[expand_param] = params[index]
                indices.append(list(self[expand_param]).index(params[index]))
        
            # Remove fake dims from request
            for dim in self.fake_dims:
                new_request.pop(dim)
            yield tuple(indices), new_request

class Config:
    def __init__(self, config):
        print(config)
        with open(config, "r") as f:
            self.options = yaml.safe_load(f)

        if isinstance(self.options["members"], dict):
            members = range(
                self.options["members"]["start"], self.options["members"]["end"] + 1
            )
        else:
            members = range(1, int(self.options["members"]) + 1)
        self.parameters = {
            param: ParamConfig(members, cfg)
            for param, cfg in self.options["parameters"].items()
        }


class ParamConfig:
    def __init__(self, members, param_config):
        self.steps = self._generate_steps(param_config.get("steps", []))
        self.base_request = param_config["forecast"]
        self.members = members
        self.climatology = param_config.get("climatology", None)
        self.windows = self._generate_windows(param_config["windows"])
        self.param_operation = self._generate_param_operation(param_config)

    @classmethod
    def _generate_steps(cls, steps_config):
        unique_steps = set()
        for steps in steps_config:
            start_step = steps["start_step"]
            end_step = steps["end_step"]
            interval = steps["interval"]
            range_len = steps.get("range", None)

            if range_len is None:
                for step in range(start_step, end_step + 1, interval):
                    if step not in unique_steps:
                        unique_steps.add(step)
            else:
                raise NotImplementedError
                # for sstep in range(start_step, end_step - range_len + 1, interval):
                #     steps.add(Step(sstep, sstep + range_len))
        return sorted(unique_steps)

    @classmethod
    def _generate_windows(cls, windows_config):
        windows = []
        for window_type in windows_config:
            window_options = window_type.copy()
            include_init = window_options.pop("include_start_step", False)
            operation = window_options.pop("window_operation", None)
            for window in window_options.pop("periods"):
                windows.append(Window(window["range"], operation, include_init, window_options))
        return windows

    @classmethod
    def _generate_param_operation(cls, param_config):
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
        # Case when window.start not in steps
        if window.include_init:
            start_index = self.steps.index(window.start)
        else:
            start_index = bisect.bisect_right(self.steps, window.start)
        return self.steps[
            start_index : self.steps.index(window.end) + 1
        ]

    def forecast_request(self, window):
        requests = self.base_request
        if isinstance(self.base_request, dict):
            requests = [self.base_request]
            
        window_requests = []
        for request in requests:
            req = Request(request.copy())
            req["step"] = self._request_steps(window)
            if request["type"] == "pf":
                req["number"] = self.members
            elif request["type"] == "cf":
                req.make_dim("number", 0)
            window_requests.append(req)
        return window_requests

    def clim_request(self, window, accumulated: bool = False):
        clim_request = Request(self.climatology["clim_keys"].copy())
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
            window_steps = self._request_steps(window)
            clim_request["step"] = list(map(self.climatology.get("steps", {}).get, 
                                       window_steps, window_steps))
        print(f"Climatology Request {clim_request}")
        return [clim_request]


class Cascade:
    @classmethod
    def read(cls, requests: list, join_key: str = ""):
        all_actions = None
        for request in requests:
            if len(request.dims) == 0:
                new_action =  SingleAction(functools.partial(read, request), None)
            else:
                nodes = np.empty(tuple(request.dims.values()), dtype=object)
                for indices, new_request in request.expand():
                    read_with_request = functools.partial(read, new_request)
                    nodes[indices] = Node(f"{new_request}",
                        payload=read_with_request,
                    )
                new_action = MultiAction(
                    None,
                    xr.DataArray(
                        nodes,
                        dims=request.dims.keys(),
                        coords={key: list(request[key]) for key in request.dims.keys()},
                    ))
            
            if all_actions is None:
                all_actions = new_action
            else:
                assert len(join_key) != 0
                all_actions = all_actions.join(new_action, join_key)
        # Currently using xarray for keeping track of dimensions but these should
        # belong in node attributes on the graph?
        print("READ", all_actions.nodes)
        return all_actions

    @classmethod
    def graph(cls, product: str, config: Config):
        total_graph = Graph([])
        for param, param_config in config.parameters.items():
            print(param)
            total_graph += getattr(cls, product)(param_config)

        return total_graph

    @classmethod
    def anomaly_prob(cls, param_config):
        total_graph = Graph([])
        for window in param_config.windows:
            climatology = cls.read(
                param_config.clim_request(window)
            )  # Will contain type em and es
            clim_em = climatology.select("type", "em")
            clim_es = climatology.select("type", "es")

            anomaly = (
                cls.read(param_config.forecast_request(window), "number")
                .groupby("step") # TODO: Fix for when steps don't align
                .join(clim_em)
                .reduce(np.subtract)
            )

            if window.is_std_anomaly:
                anomaly.join(clim_es).reduce(np.division)
            if window.operation is not None:
                anomaly = anomaly.reduce(window.operation)
            for threshold in window.thresholds():
                total_graph += (
                    anomaly.map(threshold)
                    .reduce(np.mean)
                    .write()
                    .graph()
                )

        return total_graph

    @classmethod
    def prob(cls, param_config):
        total_graph = Graph([])
        for window in param_config.windows:
            prob = cls.read(param_config.forecast_request(window))

            # Multi-parameter processing operation
            if param_config.param_operation is not None:
                prob = prob.reduce(param_config.param_operation, "param")

            prob = prob.groupby("step")
            if window.operation is not None:
                prob = prob.reduce(window.operation)
            for threshold in window.thresholds():
                total_graph += (
                    prob.map(threshold)
                    .reduce(np.mean)
                    .write()
                    .graph()
                )

        return total_graph

    @classmethod
    def wind(cls, param_config):
        total_graph = Graph([])
        for window in param_config.windows:
            wind_speed = cls.read(param_config.forecast_request(window), "number").reduce(
                lambda x, y: np.sqrt(x**2 + y**2), "param"
            )
            mean = wind_speed.reduce(np.mean, "number").write()
            std = wind_speed.reduce(np.std, "number").write()
            total_graph += mean.graph() + std.graph()
        return total_graph

    @classmethod
    def ensms(cls, param_config):
        total_graph = Graph([])
        for window in param_config.windows:
            ensms = cls.read(param_config.forecast_request(window), "number")
            mean = ensms.reduce(np.mean, "number").write()
            std = ensms.reduce(np.std, "number").write()
            total_graph += mean.graph() + std.graph()
        return total_graph

    @classmethod
    def extreme(cls, param_config):
        total_graph = Graph([])
        for window in param_config.windows:
            climatology = cls.read(param_config.clim_request(window, True))
            parameter = cls.read(param_config.forecast_request(window), "number")

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


@pytest.mark.parametrize("product, config", 
[
    ["prob", "/etc/ecmwf/nfs/dh1_home_a/mawj/Documents/cascade/tests/templates/prob.yaml"], 
    ["anomaly_prob", "/etc/ecmwf/nfs/dh1_home_a/mawj/Documents/cascade/tests/templates/t850.yaml"]
])
def test_graph_construction(product, config):
    cfg = Config(config)
    pyvis.to_pyvis(Cascade.graph(product, cfg), notebook=True, cdn_resources="remote").show(f"/etc/ecmwf/nfs/dh1_home_a/mawj/Documents/cascade/{product}_graph.html")

