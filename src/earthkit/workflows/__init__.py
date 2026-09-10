# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

try:
    from ._version import __version__  # noqa: F401
except ImportError:
    # assuming editable install etc
    pass
from . import fluent, mark

__all__ = [
    "mark",
    "fluent",
]
