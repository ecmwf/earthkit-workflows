# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Logging setup and configuration for any process and deployment

Assumed to propagate in a cascading manner:
 - infrastructure operator configures gateway with logging as cli param,
 - gateway spawns a launcher for each individual job
   - logging config propagated via cli param / other means for troika,slurm,k8s,
 - each launcher spawns procesess for controller and executors
   - logging config propagated via proc kwarg
 - each executor additionally spawns shm server, data server, individual runners
   - only logging dict propagated, via proc kwarg

This cascading manner accumulates context as it goes -- gateway adds jobId every
time it launches a job, executor launcher adds workerId to each of worker processes,
etc. This distinguishes individual log streams

We currently use LoggingConfig pydantic model for all logging configs, and pass
that through cli params via base64 of the json -- not human-practical, but safe

We don't expose much config options -- but any addition to the LoggingConfig class
gets propagated to all places that initiate logging. That is, any logging change
can be implemented fully within this module
"""

import base64
import logging
import logging.config
import os
from typing import Literal

import orjson
from pydantic import BaseModel
from typing_extensions import Self

import cascade.deployment.logging.defaults as defaults


class LoggingConfig(BaseModel):
    formatter: Literal["line", "json"]
    """Line is standard python logging, json uses structured logging"""
    path_base: str|None
    """If None log to stdout, otherwise log into files in path_base directory, with each process having its own files.
    Expected to have been created beforehand"""

    def ser_cliparam(self) -> str:
        return base64.b64encode(self.model_dump_json().encode('utf-8')).decode('utf-8')

    def withContext(self, context_kv: str) -> Self:
        if self.path_base is not None:
            return self.model_copy(update={'path_base': self.path_base + f"{context_kv}."})
        else:
            # TODO handle context in the stdout loggers. Not critical because all
            # logs of zmq messages have context in them anyway
            return self


DefaultLoggingConfig = LoggingConfig(formatter="line", path_base=None)

def as_dict_config(loggingConfig: LoggingConfig, hostAndRole: str) -> dict:
    if loggingConfig.path_base:
        filename = f"{loggingConfig.path_base}{hostAndRole}.txt"
        handler = defaults.handlers['filename'](filename) # ty:ignore[call-non-callable] # sloppy typing on my side
    else:
        handler = defaults.handlers['stdout']
    return {
        **defaults.base,
        **handler,
        **defaults.formatters[loggingConfig.formatter],
    }

def init_from_obj(loggingConfig: LoggingConfig, hostAndRole: str) -> None:
    dictConfig = as_dict_config(loggingConfig, hostAndRole)
    logging.config.dictConfig(dictConfig)

def init_from_cliparam(cliparam: str|None, hostAndRole: str) -> LoggingConfig:
    if cliparam is not None:
        loggingConfig = LoggingConfig(**orjson.loads(base64.b64decode(cliparam.encode('utf-8'))))
    else:
        loggingConfig = DefaultLoggingConfig
        logging.getLogger(__name__).warning("using default config for logging at {hostAndRole}")

    init_from_obj(loggingConfig, hostAndRole)
    return loggingConfig
