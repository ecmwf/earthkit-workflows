# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging.config

import fire

from cascade.executor.config import logging_config, logging_config_filehandler
from cascade.gateway.server import serve


def main(url: str, log_base: str | None = None) -> None:
    if log_base:
        log_path = f"{log_base}/gateway.txt"
        logging.config.dictConfig(logging_config_filehandler(log_path))
    else:
        logging.config.dictConfig(logging_config)
    serve(url, log_base)


if __name__ == "__main__":
    fire.Fire(main)
