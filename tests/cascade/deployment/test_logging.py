# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from cascade.deployment.logging import LoggingConfig, as_dict_config, process_log_paths


def test_as_dict_config_uses_logs_suffix() -> None:
    cfg = LoggingConfig(path_base="/tmp/cascade.")
    dict_config = as_dict_config(cfg, "worker_w0")
    handler = dict_config["handlers"]["default"]
    assert handler["class"] == "logging.FileHandler"
    assert handler["filename"] == "/tmp/cascade.worker_w0.logs.txt"


def test_process_log_paths_for_worker_role() -> None:
    cfg = LoggingConfig(path_base="/tmp/cascade.")
    paths = process_log_paths(cfg, "worker_w0")
    assert paths is not None
    assert paths.logs == "/tmp/cascade.worker_w0.logs.txt"
    assert paths.stdout == "/tmp/cascade.worker_w0.stdout.txt"
    assert paths.stderr == "/tmp/cascade.worker_w0.stderr.txt"
