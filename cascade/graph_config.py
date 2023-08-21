import itertools
from collections import OrderedDict
import numexpr
import yaml
import bisect


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

    def thresholds(self):
        for threshold_options in self.options["thresholds"]:
            yield lambda x: numexpr.evaluate(
                "data "
                + threshold_options["comparison"]
                + str(threshold_options["threshold"]),
                local_dict={"data": x},
            )


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

    def make_dim(self, key, value=None):
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
        param_options = param_config.copy()
        self.steps = self._generate_steps(param_options.pop("steps", []))
        self.base_request = param_options.pop("forecast")
        self.members = members
        self.climatology = param_options.pop("climatology", None)
        self.windows = self._generate_windows(param_options.pop("windows"))
        self.param_operation = self._generate_param_operation(param_options)
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
                lambda x, y, filter_configs=filter_configs: x
                if numexpr.evaluate(
                    "data "
                    + filter_configs["comparison"]
                    + str(filter_configs["threshold"]),
                    local_dict={"data": y},
                )
                else filter_configs["replacement"]
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
            clim_request["step"] = list(
                map(self.climatology.get("steps", {}).get, window_steps, window_steps)
            )
        print(f"Climatology Request {clim_request}")
        return [clim_request]
