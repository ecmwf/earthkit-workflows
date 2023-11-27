import functools
import array_api_compat
import numpy as np

from meteokit import extreme
from meteokit.stats import iter_quantiles
from earthkit.data import FieldList
from earthkit.data.sources.numpy_list import NumpyFieldList
from pproc.common.resources import ResourceMeter
from pproc import clustereps
from pproc.clustereps.utils import normalise_angles
from pproc.clustereps.io import read_steps_grib
from cascade.patch import PatchModule

from .grib import extreme_grib_headers, threshold_grib_headers
from .wrappers.metadata import GribBufferMetaData


def concatenate(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
    """
    Concatenates the list of fields inside each NumpyFieldList into a single
    NumpyFieldList object

    Parameters
    ----------
    arrays: list[NumpyFieldList]
        NumpyFieldList instances to whose fields are to be concatenated

    Return
    ------
    NumpyFieldList
        Contains all fields inside the input field lists
    """
    return sum(arrays[1:], arrays[0])


def standardise_output(data):
    # Also, nest the data to avoid problems with not finding geography attribute
    ret = data
    if len(ret.shape) == 1:
        ret = ret.reshape((1, *data.shape))
    assert len(ret.shape) == 2
    return ret


def multi_arg_function(func: str, *arrays: list[NumpyFieldList]) -> NumpyFieldList:
    if len(arrays) == 1:
        concat = arrays[0].values
    else:
        concat = concatenate(*arrays).values
        # assert len(concat) == len(arrays)

    xp = array_api_compat.array_namespace(concat)
    res = getattr(xp, func)(concat, axis=0)
    return FieldList.from_numpy(standardise_output(res), arrays[0][0].metadata())


def norm(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
    vals = [x.values for x in arrays]
    xp = array_api_compat.array_namespace(*vals)
    if len(vals) == 1:
        # Assume fields to compute norm of are nested in single field list
        vals = vals[0]
    assert len(vals) == 2, f"Expected 2 fields for norm, received {len(vals)}"
    norm = xp.sqrt(vals[0] ** 2 + vals[1] ** 2)
    return FieldList.from_numpy(standardise_output(norm), arrays[0][0].metadata())


def two_arg_function(
    func: str, *arrays: NumpyFieldList, extract_keys: tuple = ()
) -> NumpyFieldList:
    vals = [x.values for x in arrays]
    xp = array_api_compat.array_namespace(*vals)
    if len(vals) == 1:
        # Assume fields to compute norm of are nested in single field list
        vals = vals[0]
        arr2_meta = arrays[0][1].metadata().buffer_to_metadata()
    else:
        arr2_meta = arrays[1][0].metadata().buffer_to_metadata()
    assert (
        len(vals) == 2
    ), f"Expected 2 fields for two_arg_functions@{func}, received {len(vals)}"
    metadata = (
        arrays[0][0]
        .metadata()
        .override({key: arr2_meta.get(key) for key in extract_keys})
    )
    res = getattr(xp, func)(vals[0], vals[1])
    return FieldList.from_numpy(standardise_output(res), metadata)


mean = functools.partial(multi_arg_function, "mean")
std = functools.partial(multi_arg_function, "std")
maximum = functools.partial(multi_arg_function, "max")
minimum = functools.partial(multi_arg_function, "min")
subtract = functools.partial(two_arg_function, "subtract")
add = functools.partial(two_arg_function, "add")
multiply = functools.partial(two_arg_function, "multiply")
divide = functools.partial(two_arg_function, "divide")


def comp_str2func(array_module, comparison: str):
    if comparison == "<=":
        return array_module.less_equal
    if comparison == "<":
        return array_module.less
    if comparison == ">=":
        return array_module.greater_equal
    return array_module.greater


def threshold(
    threshold_config: dict, arr: NumpyFieldList, edition: int = 1
) -> NumpyFieldList:
    xp = array_api_compat.array_namespace(arr.values)
    # Find all locations where np.nan appears as an ensemble value
    is_nan = xp.isnan(arr.values)
    thesh = comp_str2func(xp, threshold_config["comparison"])(
        arr.values, threshold_config["value"]
    )
    res = xp.where(is_nan, xp.nan, thesh)
    threshold_headers = threshold_grib_headers(edition, threshold_config)
    metadata = arr[0].metadata().override(threshold_headers)
    return FieldList.from_numpy(standardise_output(res), metadata)


def efi(
    clim: NumpyFieldList,
    ens: NumpyFieldList,
    eps: float,
    num_steps: int,
    control: bool = False,
) -> NumpyFieldList:
    extreme_headers = extreme_grib_headers(clim, ens, num_steps)
    if control:
        extreme_headers.update({"marsType": "efic", "totalNumber": 1, "number": 0})
        metadata = ens[0].metadata().override(extreme_headers)
    else:
        extreme_headers.update({"marsType": "efi", "efiOrder": 0})
        metadata = ens[0].metadata().override(extreme_headers)

    xp = array_api_compat.array_namespace(ens.values, clim.values)
    with PatchModule(extreme, "numpy", xp):
        res = extreme.efi(clim.values, ens.values, eps)
    return FieldList.from_numpy(standardise_output(res), metadata)


def sot(
    clim: NumpyFieldList, ens: NumpyFieldList, number: int, eps: float, num_steps: int
) -> NumpyFieldList:
    extreme_headers = extreme_grib_headers(clim, ens, num_steps)
    if number == 90:
        efi_order = 99
    elif number == 10:
        efi_order = 1
    else:
        raise Exception(
            "SOT value '{sot}' not supported in template! Only accepting 10 and 90"
        )
    metadata = (
        ens[0]
        .metadata()
        .override({**extreme_headers, "marsType": "sot", "efiOrder": efi_order})
    )

    xp = array_api_compat.array_namespace(ens.values, clim.values)
    with PatchModule(extreme, "numpy", xp):
        res = extreme.sot(clim.values, ens.values, number, eps)
    return FieldList.from_numpy(standardise_output(res), metadata)


def quantiles(ens: NumpyFieldList, quantile: float) -> NumpyFieldList:
    xp = array_api_compat.array_namespace(ens.values)
    with PatchModule(extreme, "numpy", xp):
        res = list(iter_quantiles(ens.values, [quantile], method="numpy"))[0]
    return FieldList.from_numpy(standardise_output(res), ens[0].metadata())


def filter(
    comparison: str,
    threshold: float,
    arr1: NumpyFieldList,
    arr2: NumpyFieldList,
    replacement=0,
) -> NumpyFieldList:
    xp = array_api_compat.array_namespace(arr1.values, arr2.values)
    condition = comp_str2func(xp, comparison)(arr2.values, threshold)
    res = xp.where(condition, replacement, arr1.values)
    return FieldList.from_numpy(standardise_output(res), arr1.metadata())


def pca(
    config, ens: NumpyFieldList, spread: NumpyFieldList, mask: np.ndarray, target: str
):
    xp = array_api_compat.array_namespace(ens.values)
    lat_lon = ens.to_latlon(flatten=True)
    ens_data = xp.reshape(
        ens.to_numpy(flatten=True),
        (config.num_members, len(config.steps), len(lat_lon["lat"])),
    )
    with ResourceMeter(f"PCA: lat {lat_lon['lat'].shape} lon {lat_lon['lon'].shape}"):
        pca_data = clustereps.pca.do_pca(
            config,
            lat_lon["lat"],
            normalise_angles(lat_lon["lon"]),
            ens_data,
            spread[0].values,
            mask,
        )

    ## Save data
    if target is not None:
        np.savez_compressed(target, **pca_data)

    return (pca_data, ens[0].metadata())


def cluster(
    config,
    pca_output: tuple[dict, GribBufferMetaData],
    ncomp_file: str,
    indexes: str,
    deterministic: str,
):
    pca_data, pca_metadata = pca_output

    ## Compute number of PCs based on the variance threshold
    var_cum = pca_data["var_cum"]
    npc = config.npc
    if npc <= 0:
        npc = clustereps.cluster.select_npc(config.var_th, var_cum)
        if ncomp_file is not None:
            with open(ncomp_file, "w") as f:
                print(npc, file=f)

    print(f"Number of PCs used: {npc}, explained variance: {var_cum[npc-1]} %")

    with ResourceMeter("Clustering"):
        (
            ind_cl,
            centroids,
            rep_members,
            centroids_gp,
            rep_members_gp,
            ens_mean,
        ) = clustereps.cluster.do_clustering(
            config, pca_data, npc, verbose=True, dump_indexes=indexes
        )

    ## Find the deterministic forecast
    if deterministic is not None:
        with ResourceMeter("Find deterministic"):
            det = read_steps_grib(config.sources, deterministic, config.steps)
            det_index = clustereps.cluster.find_cluster(
                det,
                ens_mean,
                pca_data["eof"][:npc, ...],
                pca_data["weights"],
                centroids,
            )
    else:
        det_index = 0

    metadata = pca_metadata.override(
        {"rep_members": rep_members, "det_index": det_index, "ind_cl": ind_cl}
    )
    ret = {
        "centroids": [
            np.array(centroids_gp),
            metadata.override({"type": "cm", "scenario": "centroids"}),
        ],
        "representatives": [
            np.array(rep_members_gp),
            metadata.override({"type": "cr", "scenario": "representatives"}),
        ],
    }
    return ret


def attribution(config, scenario: str, pca_output: dict, cluster_output: dict):
    pca_data, _ = pca_output
    scdata, metadata = cluster_output[scenario]

    with ResourceMeter("Read climatology"):
        ## Read climatology fields
        clim = clustereps.attribution.get_climatology_fields(
            config.climMeans, config.seasons, config.stepDate
        )

        ## Read climatological EOFs
        clim_eof, clim_ind = clustereps.attribution.get_climatology_eof(
            config.climClusterCentroidsEOF,
            config.climEOFs,
            config.climPCs,
            config.climSdv,
            config.climClusterIndex,
            config.nClusterClim,
            config.monStartDoS,
            config.monEndDoS,
        )

    weights = pca_data["weights"]

    ## Compute anomalies
    anom = scdata - clim
    anom = np.clip(anom, -config.clip, config.clip)

    with ResourceMeter(f"Attribute {scenario}"):
        cluster_att, min_dist = clustereps.attribution.attribution(
            anom, clim_eof, clim_ind, weights
        )

    return (metadata, scdata, anom, cluster_att, min_dist)
