# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Cascade exception hierarchy.

Three categories:
 - CascadeInternalError: programmer error, unexpected state, internal invariant violation
 - CascadeInfrastructureError: OS, network, shm, or other infra issues; re-run may help
 - CascadeUserError: user code or configuration issue; user must fix their code/config
"""


class CascadeError(Exception):
    """Base class for all Cascade exceptions."""

    def __init__(self, description: str, parent: Exception | None = None) -> None:
        self.description = description
        self.parent = parent
        super().__init__(description)

    def __repr__(self) -> str:
        parent_repr = f", parent={repr(self.parent)}" if self.parent else ""
        return f"{type(self).__name__}({self.description!r}{parent_repr})"

    def __str__(self) -> str:
        if self.parent:
            return f"{self.description} (caused by {repr(self.parent)})"
        return self.description


class CascadeInternalError(CascadeError):
    pass


class CascadeInfrastructureError(CascadeError):
    pass


class CascadeUserError(CascadeError):
    pass
