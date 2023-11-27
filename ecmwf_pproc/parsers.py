from pproc.clustereps.__main__ import get_parser as cluster_parser
from pproc.common.config import default_parser


def basic_parser(description: str, clim: bool = False, det: bool = False):
    parser = default_parser(description)
    parser.add_argument("-e", "--ensemble", required=True, help="Ensemble data (GRIB)")
    if clim:
        parser.add_argument(
            "--climatology", required=True, help="Climatology data (GRIB)"
        )
    if det:
        parser.add_argument(
            "-d", "--deterministic", default=None, help="Deterministic forecast (GRIB)"
        )
    return parser


_parsers = {
    "clustereps": cluster_parser(),
    "prob": basic_parser("Ensemble threshold probabiltities"),
    "anomaly_prob": basic_parser(
        "Ensemble threshold anomaly probabiltities", clim=True
    ),
    "ensms": basic_parser("Ensemble mean and standard deviation"),
    "wind": basic_parser("Ensemble mean and standard deviation", det=True),
    "extreme": basic_parser(
        "Compute EFI and SOT from forecast and climatology", clim=True
    ),
    "quantiles": basic_parser("Compute quantiles of an ensemble"),
}


def get_parser(product: str):
    return _parsers.get(product)
