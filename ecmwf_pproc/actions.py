import numpy as np
import xarray as xr
import itertools

from earthkit.data import FieldList
from cascade.fluent import Action, Node, Payload
from cascade.fluent import SingleAction as BaseSingleAction
from cascade.fluent import MultiAction as BaseMultiAction

from .io import cluster_write
from .io import write as write_grib
from .graph_config import (
    Window,
)
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

    def cluster(self, config, ncomp_file, indexes, deterministic):
        return self.then(
            Payload(
                functions.cluster,
                (config, "input0", ncomp_file, indexes, deterministic),
            )
        )

    def write(self, target, config_grib_sets: dict):
        if target != "null:":
            grib_sets = config_grib_sets.copy()
            grib_sets.update(self.nodes.attrs)
            for name, values in self.nodes.coords.items():
                if values.data.ndim == 0:
                    grib_sets[name] = values.data
                else:
                    assert values.data.ndim == 1
                    grib_sets[name] = values.data[0]
            payload = Payload(write_grib, (target, "input0", grib_sets))
            self.sinks.append(Node(payload, self.node()))
        return self


class MultiAction(BaseMultiAction):
    def to_single(self, payload: Payload, node: Node = None):
        return SingleAction(payload, self, node)

    def concatenate(self, key: str):
        return self.reduce(Payload(functions.concatenate), key)

    def mean(self, key: str = ""):
        return self.reduce(Payload(functions.mean), key)

    def std(self, key: str = ""):
        return self.reduce(Payload(functions.std), key)

    def maximum(self, key: str = ""):
        return self.reduce(Payload(functions.maximum), key)

    def minimum(self, key: str = ""):
        return self.reduce(Payload(functions.minimum), key)

    def norm(self, key: str = ""):
        return self.reduce(Payload(functions.norm), key)

    def diff(self, key: str = "", extract_keys: tuple = ()):
        return self.reduce(
            Payload(functions.subtract, ("input1", "input0", extract_keys)), key
        )

    def subtract(self, key: str = "", extract_keys: tuple = ()):
        return self.reduce(
            Payload(functions.subtract, ("input0", "input1", extract_keys)), key
        )

    def add(self, key: str = "", extract_keys: tuple = ()):
        return self.reduce(
            Payload(functions.add, ("input0", "input1", extract_keys)), key
        )

    def divide(self, key: str = "", extract_keys: tuple = ()):
        return self.reduce(
            Payload(functions.divide, ("input0", "input1", extract_keys)), key
        )

    def multiply(self, key: str = "", extract_keys: tuple = ()):
        return self.reduce(
            Payload(functions.multiply, ("input0", "input1", extract_keys)), key
        )

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
        return self.reduce(Payload(operation), "param")

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
        if mask is not None:
            raise NotImplementedError()
        return self.reduce(
            Payload(functions.pca, (config, "input0", "input1", mask, target))
        )

    def attribution(self, config, targets):
        def _attribution(action, scenario):
            payload = Payload(
                functions.attribution, (config, scenario, "input0", "input1")
            )
            attr = action.reduce(payload)
            attr._add_dimension("scenario", scenario)
            return attr

        return self.transform(
            _attribution, ["centroids", "representatives"], "scenario"
        ).foreach(
            np.asarray(
                [
                    Payload(
                        cluster_write,
                        (config, "centroids", "input0", targets["centroids"]),
                    ),
                    Payload(
                        cluster_write,
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

    def write(self, target, config_grib_sets: dict):
        if target != "null:":
            coords = list(self.nodes.coords.keys())
            for node_attrs in itertools.product(
                *[self.nodes.coords[key].data for key in coords]
            ):
                node_coords = {
                    key: node_attrs[index] for index, key in enumerate(coords)
                }
                node = self.node(node_coords)

                grib_sets = config_grib_sets.copy()
                grib_sets.update(self.nodes.attrs)
                grib_sets.update(node_coords)
                self.sinks.append(
                    Node(Payload(write_grib, (target, "input0", grib_sets)), [node])
                )
        return self
