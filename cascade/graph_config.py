import itertools
from collections import OrderedDict
import bisect

from pproc.common.config import Config as BaseConfig

from . import functions


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

    @property
    def dims(self):
        dimensions = OrderedDict()
        for key, values in self.request.items():
            if key == "interpolate" or key in self.no_expand:
                continue
            if hasattr(values, "__iter__") and not isinstance(values, str):
                dimensions[key] = len(values)
        return dimensions

    def __setitem__(self, key, value):
        self.request[key] = value

    def __getitem__(self, key):
        return self.request[key]

    def __contains__(self, key):
        return key in self.request

    def pop(self, key, default=None):
        if default is None:
            return self.request.pop(key)
        return self.request.pop(key, default)

    def make_dim(self, key, value=None):
        if key in self:
            assert type(self.request[key], (str, int, float))
            self.request[key] = [self.request[key]]
        else:
            self.request[key] = [value]
            self.fake_dims.append(key)

    def expand(self):
        for params in itertools.product(*[self.request[x] for x in self.dims.keys()]):
            new_request = self.request.copy()
            indices = []
            for index, expand_param in enumerate(self.dims.keys()):
                new_request[expand_param] = params[index]
                indices.append(list(self.request[expand_param]).index(params[index]))

            # Remove fake dims from request
            for dim in self.fake_dims:
                new_request.pop(dim)
            yield tuple(indices), new_request


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

    def forecast_request(self, window, source: str = "ens", no_expand: tuple[str] = ()):
        requests = self.sources[source]
        if isinstance(requests, dict):
            requests = [requests]

        window_requests = []
        for request in requests:
            req = Request({**request, **self.in_keys}, no_expand)
            req["step"] = self._request_steps(window)
            if request["type"] == "pf":
                req["number"] = self.members
            elif request["type"] == "cf":
                req.make_dim("number", 0)
            window_requests.append(req)
        return window_requests

    def clim_request(
        self, window, accumulated: bool = False, no_expand: tuple[str] = ()
    ):
        clim_req = Request(self.sources["clim"], no_expand)
        steps = clim_req.pop("step", {})
        if accumulated:
            clim_req["step"] = steps.get(window.name, window.name)
        else:
            window_steps = self._request_steps(window)
            clim_req["step"] = list(map(steps.get, window_steps, window_steps))
        return [clim_req]


class WindConfig(ParamConfig):
    def vod2uv(self, source: str) -> bool:
        req = self.sources[source]
        if isinstance(req, list):
            req = req[0]
        return req.get("interpolate", {}).get("vod2uv", "0") == "1"

    def forecast_request(self, window: Window, source: str):
        no_expand = ("param") if self.vod2uv(source) else ()
        return super().forecast_request(window, source, no_expand)


class ExtremeConfig(ParamConfig):
    def clim_request(
        self, window, accumulated: bool = False, no_expand: tuple[str] = ()
    ):
        clim_reqs = super().clim_request(window, accumulated, no_expand)
        for req in clim_reqs:
            num_quantiles = int(req["quantile"])
            req["quantile"] = ["{}:100".format(i) for i in range(num_quantiles + 1)]
        return clim_reqs
