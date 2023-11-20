import itertools
from collections import OrderedDict
import bisect
from datetime import timedelta
import copy
import numpy as np

from pproc.common.config import Config as BaseConfig
from pproc.clustereps.config import FullClusterConfig

from . import functions
from .io import _source_from_location


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

        self.step = int(window_range[2]) if len(window_range) > 2 else 1
        if self.include_init:
            self.steps = list(range(self.start, self.end + 1, self.step))
        else:
            self.steps = list(range(self.start + self.step, self.end + 1, self.step))

        self.operation = operation
        self.options = window_options.copy()
        self.grib_set = self.options.pop("grib_set", {})


class Request:
    def __init__(self, request: dict, no_expand: tuple[str] = ()):
        self.request = request.copy()
        self.fake_dims = []
        self.no_expand = no_expand
        self.ignore = ["interpolate"]

    @property
    def dims(self) -> OrderedDict:
        dimensions = OrderedDict()
        for key, values in self.request.items():
            if key in self.ignore or key in self.no_expand:
                continue
            if hasattr(values, "__iter__") and not isinstance(values, str):
                dimensions[key] = len(values)
        return dimensions

    def __setitem__(self, key, value):
        self.request[key] = value

    def __getitem__(self, key):
        return self.request[key]

    def __contains__(self, key) -> bool:
        return key in self.request

    def update(self, **kwargs):
        self.request.update(**kwargs)

    def pop(self, key, default=None):
        if default is None:
            return self.request.pop(key)
        return self.request.pop(key, default)

    def make_dim(self, key, value=None):
        if key in self:
            assert type(self[key], (str, int, float))
            self[key] = [self[key]]
        else:
            self[key] = [value]
            self.fake_dims.append(key)

    def expand(self):
        for params in itertools.product(*[self.request[x] for x in self.dims.keys()]):
            new_request = self.request.copy()
            indices = []
            name = [f"type={self.request['type']}"]
            for index, expand_param in enumerate(self.dims.keys()):
                new_request[expand_param] = params[index]
                name.append(f"{expand_param}={params[index]}")
                indices.append(list(self.request[expand_param]).index(params[index]))

            # Remove fake dims from request
            for dim in self.fake_dims:
                new_request.pop(dim)
            yield tuple(indices), new_request, ",".join(name)


class MultiSourceRequest(Request):
    def __init__(self, requests: list[dict], no_expand: tuple[str] = ()):
        super().__init__(requests[0], no_expand)
        self.requests = requests

    def __setitem__(self, key, value):
        super().__setattr__(key, value)
        [x.__setitem__(key, value) for x in self.requests]

    def __getitem__(self, key):
        values = [x.__getitem__(key) for x in self.requests]
        # print("__getitem__", key, values[0], values[1], np.all([values[0] == values[x] for x in range(1, len(values))]))
        if np.all([values[0] == values[x] for x in range(1, len(values))]):
            return values[0]
        raise Exception(f"Requests {self.requests} differ on value for key {key}")

    def __contains__(self, key) -> bool:
        contains = [x.__contains__(key) for x in self.requests]
        if all([contains[0] == contains[x] for x in range(1, len(contains))]):
            return contains[0]
        raise Exception(f"Not all requests {self.requests} contain key {key}")

    def update(self, **kwargs):
        super().update(**kwargs)
        [x.update(**kwargs) for x in self.requests]

    def pop(self, key, default=None):
        contains = key in self
        if default is None or contains:
            value = self[key]
            super().pop(key)
            [x.pop(key) for x in self.requests]
            return value
        super().pop(key)
        [x.pop(key) for x in self.requests]
        return default

    def expand(self):
        for params in itertools.product(*[self.request[x] for x in self.dims.keys()]):
            indices = []
            name = [f"type={self.request['type']}"]
            new_requests = copy.deepcopy(self.requests)
            for index, expand_param in enumerate(self.dims.keys()):
                [x.__setitem__(expand_param, params[index]) for x in new_requests]
                name.append(f"{expand_param}={params[index]}")
                indices.append(list(self.request[expand_param]).index(params[index]))

            # Remove fake dims from request
            for dim in self.fake_dims:
                [x.pop(dim) for x in new_requests]
            yield tuple(indices), new_requests, ",".join(name)


class Config(BaseConfig):
    def __init__(self, args):
        super().__init__(args)

        if isinstance(self.options["members"], dict):
            self.members = range(
                self.options["members"]["start"], self.options["members"]["end"] + 1
            )
        else:
            self.members = range(1, int(self.options["members"]) + 1)
        self.out_keys = self.options.pop("out_keys", {})
        self.in_keys = self.options.pop("in_keys", {})


class ParamConfig:
    def __init__(self, members, param_config, in_keys, out_keys):
        param_options = param_config.copy()
        self.steps = self._generate_steps(param_options.pop("steps", []))
        self.sources = param_options.pop("sources")
        self.members = members
        self.windows = self._generate_windows(param_options.pop("windows"))
        self.param_operation = self._generate_param_operation(param_options)
        self.targets = param_options.pop("targets")
        self.out_keys = out_keys.copy()
        self.in_keys = in_keys.copy()
        self.options = param_options

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
        for window_options in windows_config:
            include_init = window_options.pop("include_start_step", False)
            operation = window_options.pop("window_operation", None)
            for window in window_options.pop("periods"):
                windows.append(
                    Window(window["range"], operation, include_init, window_options)
                )
        return windows

    @classmethod
    def _generate_param_operation(cls, param_config):
        if "input_filter_operation" in param_config:
            filter_configs = param_config.pop("input_filter_operation")
            return (
                functions.filter,
                (
                    filter_configs["comparison"],
                    float(filter_configs["threshold"]),
                    "input0",
                    "input1",
                    float(filter_configs.get("replacement", 0)),
                ),
            )
        return param_config.pop("input_combine_operation", None)

    def get_target(self, target: str) -> str:
        return self.targets.get(target, "null:")

    def _request_steps(self, window):
        if len(self.steps) == 0:
            return window.steps
        # Case when window.start not in steps
        if window.include_init:
            start_index = self.steps.index(window.start)
        else:
            start_index = bisect.bisect_right(self.steps, window.start)
        return self.steps[start_index : self.steps.index(window.end) + 1]

    def forecast_request(self, window, ens: str, no_expand: tuple[str] = ()):
        source, requests = _source_from_location(ens, self.sources)
        if isinstance(requests, dict):
            requests = [requests]

        window_requests = []
        for request in requests:
            req = Request({**request, **self.in_keys, "source": source}, no_expand)
            req["step"] = self._request_steps(window)
            if request["type"] == "pf":
                req["number"] = self.members
            elif request["type"] == "cf":
                req.make_dim("number", 0)
            window_requests.append(req)
        return window_requests

    def clim_request(
        self, window, clim: str, accumulated: bool = False, no_expand: tuple[str] = ()
    ):
        source, requests = _source_from_location(clim, self.sources)
        assert len(requests) == 1, f"Expected a single request, got {requests}"
        clim_req = Request(requests[0], no_expand)
        clim_req["source"] = source
        steps = clim_req.pop("step", {})
        if accumulated:
            clim_req["step"] = steps.get(window.name, window.name)
        else:
            window_steps = self._request_steps(window)
            clim_req["step"] = list(map(steps.get, window_steps, window_steps))
        return [clim_req]


class WindConfig(ParamConfig):
    def vod2uv(self, ens: str) -> bool:
        _, reqs = _source_from_location(ens, self.sources)
        return reqs[0].get("interpolate", {}).get("vod2uv", "0") == "1"

    def forecast_request(self, window: Window, ens: str):
        vod2uv = self.vod2uv(ens)
        no_expand = ("param") if vod2uv else ()
        return vod2uv, super().forecast_request(window, ens, no_expand)


class ExtremeConfig(ParamConfig):
    def clim_request(
        self, window, clim: str, accumulated: bool = False, no_expand: tuple[str] = ()
    ):
        clim_reqs = super().clim_request(window, clim, accumulated, no_expand)
        for req in clim_reqs:
            num_quantiles = int(req["quantile"])
            req["quantile"] = ["{}:100".format(i) for i in range(num_quantiles + 1)]
        return clim_reqs


class ClusterConfig(FullClusterConfig):
    def spread_request(self, spread: str, no_expand: tuple[str] = ()):
        source, reqs = _source_from_location(spread, self.sources)
        assert len(reqs) == 1, f"Expected a single request, got {reqs}"
        ret = Request(reqs[0], no_expand=no_expand)
        ret.update(source=source, step=self.steps)
        return [ret]

    def spread_compute_request(
        self, spread_compute: list[str], ndays: int = 31, no_expand: tuple[str] = ()
    ):
        ret = []
        dates = [
            (self.date - timedelta(days=diff)).strftime("%Y%m%d")
            for diff in range(ndays, 0, -1)
        ]
        for loc in spread_compute:
            source, reqs = _source_from_location(loc, self.sources)
            assert len(reqs) == 1, f"Expected a single request, got {reqs}"
            reqs[0].update(step=self.steps, date=dates, source=source)
            ret.append(reqs[0])
        return [MultiSourceRequest(ret, no_expand=no_expand)]

    def forecast_request(self, ensemble: str, no_expand: tuple[str] = ()):
        source, requests = _source_from_location(ensemble, self.sources)

        window_requests = []
        for request in requests:
            req = Request({**request, "source": source}, no_expand)
            req["step"] = self.steps
            if request["type"] == "pf":
                req["number"] = list(range(1, self.num_members))
            elif request["type"] == "cf":
                req.make_dim("number", 0)
            window_requests.append(req)
        return window_requests
