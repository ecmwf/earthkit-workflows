import itertools
from collections import OrderedDict
import yaml
import bisect
import numexpr
import xarray as xr
import numpy as np
import functools

from . import functions


def threshold_config(threshold: dict):
    threshold_keys = {}
    threshold_value = threshold["value"]
    if "localDecimalScaleFactor" in threshold:
        scale_factor = threshold["localDecimalScaleFactor"]
        threshold_keys["localDecimalScaleFactor"] = scale_factor
        threshold_value = round(threshold["value"] * 10**scale_factor, 0)

    comparison = threshold["comparison"]
    if "<" in comparison:
        threshold_keys["thresholdIndicator"] = 2
        threshold_keys["upperThreshold"] = threshold_value
    else:
        threshold_keys["thresholdIndicator"] = 1
        threshold_keys["lowerThreshold"] = threshold_value

    return (
        functions.threshold,
        (comparison, threshold["value"], "input0"),
    ), threshold_keys


def extreme_config(eps, number: int = 0, control: bool = False):
    if number == 0:
        payload = (functions.efi, "input1", "input0", eps)
        efi_keys = {"marsType": "efi", "efiOrder": 0}
        if control:
            efi_keys.update({"marsType": "efic", "totalNumber": 1, "number": 0})
    else:
        payload = (functions.sot, ("input1", "input0", number, eps))
        if number == 90:
            efi_order = 99
        elif number == 10:
            efi_order = 1
        else:
            raise Exception(
                "SOT value '{sot}' not supported in template! Only accepting 10 and 90"
            )
        efi_keys = {"marsType": "sot", "efiOrder": efi_order}
    return payload, efi_keys


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
        self.options = window_options


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


def param_config(product: str, members: int, cfg: dict):
    if product == "wind":
        return WindConfig(members, cfg)
    if product == "quantile":
        return QuantileConfig(members, cfg)
    return ParamConfig(members, cfg)


class Config:
    def __init__(self, product, config):
        self.product = product
        with open(config, "r") as f:
            self.options = yaml.safe_load(f)

        if isinstance(self.options["members"], dict):
            members = range(
                self.options["members"]["start"], self.options["members"]["end"] + 1
            )
        else:
            members = range(1, int(self.options["members"]) + 1)
        self.parameters = {
            param: param_config(product, members, cfg)
            for param, cfg in self.options["parameters"].items()
        }


class ParamConfig:
    def __init__(self, members, param_config):
        param_options = param_config.copy()
        self.steps = self._generate_steps(param_options.pop("steps", []))
        self.sources = param_options.pop("sources")
        self.members = members
        self.windows = self._generate_windows(param_options.pop("windows"))
        self.param_operation = self._generate_param_operation(param_options)
        self.target = param_options.pop("target", "fdb:")
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
                filter_configs["comparison"],
                float(filter_configs["threshold"]),
                "input0",
                "input1",
                float(filter_configs.get("replacement", 0)),
            )
        return param_config.pop("input_combine_operation", None)

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
            req = Request(request, no_expand)
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
    def __init__(self, members, param_config):
        super().__init__(members, param_config)

    def vod2uv(self, source: str) -> bool:
        req = self.sources[source]
        if isinstance(req, list):
            req = req[0]
        return req.get("interpolate", {}).get("vod2uv", "0") == "1"

    def forecast_request(self, window: Window, source: str):
        no_expand = ("param") if self.vod2uv(source) else ()
        return super().forecast_request(window, source, no_expand)


class QuantileConfig(ParamConfig):
    def clim_request(
        self, window, accumulated: bool = False, no_expand: tuple[str] = ()
    ):
        clim_reqs = super().clim_request(window, accumulated, no_expand)
        for req in clim_reqs:
            num_quantiles = int(req["quantile"])
            req["quantile"] = ["{}:100".format(i) for i in range(num_quantiles + 1)]
        return clim_reqs
