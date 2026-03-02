# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Custom json formatter, felt easier than struggling with another 3rd party lib"""

import logging
import logging.config

import orjson


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "process": record.process,
            "message": record.getMessage(),
        }
        
        # NOTE standard_attrs = {
        #    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
        #    'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
        #    'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
        #    'processName', 'process', 'message', 'asctime', 'taskName',
        #}
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return orjson.dumps(log_data).decode("utf-8")
