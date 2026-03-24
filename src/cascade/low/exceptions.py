# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Cascade exception hierarchy.

Three categories:
 - CascadeInternalError: programmer error, unexpected state, internal invariant violation,
   only code update is expected to help
 - CascadeInfrastructureError: OS, network, shm, or other infra issues; re-run may help,
   perhaps with extra resources or less noisy neighbor network traffic
 - CascadeUserError: user code or configuration issue; user must fix their code/config,
   then a re-run may help
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

    def add_context(self, context: str) -> None:
        self.description = f"{context}; {self.description}"


class CascadeInternalError(CascadeError):
    pass


class CascadeInfrastructureError(CascadeError):
    pass


class CascadeUserError(CascadeError):
    pass


def ser(e: Exception, extra_context: str | None = None) -> str:
    """Serialize error to a string representation, converting to CascadeError first if needed."""
    # we go with InternalError, because context-aware conversions should have happened prior
    e = e if isinstance(e, CascadeError) else CascadeInternalError(description=repr(e), parent=e)
    if extra_context:
        e.add_context(extra_context)
    return repr(e)


def des(s: str) -> CascadeError:
    """Deserialize a string back to a CascadeError.

    Attempts to parse the string as '<CascadeErrorClassName>(<detail>)'.
    If successful, constructs the appropriate exception class.
    Otherwise, returns CascadeInternalError with the input string.
    Parent field is always set to None.
    """
    import re

    # Try to match pattern: ClassName('detail') or ClassName("detail")
    # We need to handle both single and double quotes, and escaped quotes
    match = re.match(
        r"^(CascadeError|CascadeInternalError|CascadeInfrastructureError|CascadeUserError)\((['\"])(.+?)\2(?:, parent=.*)?\)$", s, re.DOTALL
    )

    if match:
        class_name = match.group(1)
        detail = match.group(3)

        # Map class name to actual class
        class_map = {
            "CascadeError": CascadeError,
            "CascadeInternalError": CascadeInternalError,
            "CascadeInfrastructureError": CascadeInfrastructureError,
            "CascadeUserError": CascadeUserError,
        }

        exc_class = class_map.get(class_name)
        if exc_class:
            return exc_class(description=detail, parent=None)

    # If parsing fails, return as CascadeInternalError
    return CascadeInternalError(description=s, parent=None)
