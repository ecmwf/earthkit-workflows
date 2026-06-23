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

import orjson

import cascade.low.tracing as tracing


class CascadeError(Exception):
    """Base class for all Cascade exceptions."""

    def __init__(self, description: str, parent: Exception | None = None, _context: dict[str, str] | None = None) -> None:
        self.description = description
        self.context: dict[str, str] = _context if _context is not None else dict(tracing.d)
        self.parent = parent
        super().__init__(description)

    def _context_str(self) -> str:
        if not self.context:
            return ""
        return "; ".join(f"{k}={v}" for k, v in self.context.items()) + "; "

    def __repr__(self) -> str:
        parent_repr = f", parent={repr(self.parent)}" if self.parent else ""
        return f"{type(self).__name__}('{self._context_str()}{self.description}'{parent_repr})"

    def __str__(self) -> str:
        prefix = self._context_str()
        if self.parent:
            return f"{prefix}{self.description} (caused by {repr(self.parent)})"
        return f"{prefix}{self.description}"

    def add_context(self, context: dict[str, str]) -> None:
        self.context.update(context)


class CascadeInternalError(CascadeError):
    pass


class CascadeInfrastructureError(CascadeError):
    pass


class CascadeUserError(CascadeError):
    pass


_CLASS_MAP: dict[str, type[CascadeError]] = {
    "CascadeError": CascadeError,
    "CascadeInternalError": CascadeInternalError,
    "CascadeInfrastructureError": CascadeInfrastructureError,
    "CascadeUserError": CascadeUserError,
}


def ser(e: Exception, extra_context: dict[str, str] | None = None) -> str:
    """Serialize error to a JSON string, converting to CascadeError first if needed.

    The serialized format preserves description and context as separate fields so
    that consumers can distinguish the original error message from propagated context.
    """
    # we go with InternalError, because context-aware conversions should have happened prior
    cascade_e = e if isinstance(e, CascadeError) else CascadeInternalError(description=repr(e), parent=e)
    if extra_context:
        cascade_e.add_context(extra_context)
    data: dict[str, object] = {
        "type": type(cascade_e).__name__,
        "description": cascade_e.description,
        "context": cascade_e.context,
    }
    if cascade_e.parent is not None:
        data["parent"] = repr(cascade_e.parent)
    return orjson.dumps(data).decode()


def des(s: str) -> CascadeError:
    """Deserialize a JSON string back to a CascadeError.

    Expects the format produced by ser(). Falls back to CascadeInternalError wrapping
    the raw string if parsing fails (e.g. for legacy repr-formatted strings).
    Parent field is always set to None on deserialization.
    """
    try:
        data = orjson.loads(s)
        class_name = data.get("type", "")
        description = data.get("description", "")
        context: dict[str, str] = data.get("context", {})
        exc_class = _CLASS_MAP.get(class_name)
        if exc_class and isinstance(description, str):
            return exc_class(description=description, _context=context)
    except Exception:
        pass
    # Fallback: wrap the raw string, no context captured from tracing
    return CascadeInternalError(description=s, _context={})
