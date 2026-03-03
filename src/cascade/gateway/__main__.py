# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging.config

import fire

from cascade.deployment.logging import init_from_cliparam
from cascade.gateway.server import roleLoggingStr, serve


def main_cli(
    url: str,
    loggingConfigSer: str | None = None,
    troika_config: str | None = None,
    max_jobs: int | None = None,
) -> None:
    loggingConfig = init_from_cliparam(loggingConfigSer, roleLoggingStr())
    serve(url, loggingConfig, troika_config, max_jobs)

if __name__ == "__main__":
    fire.Fire(main_cli)
