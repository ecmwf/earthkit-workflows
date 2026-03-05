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


class CascadeInternalError(CascadeError):
    pass


class CascadeInfrastructureError(CascadeError):
    pass


class CascadeUserError(CascadeError):
    pass

"""
# TODO we need to be able to reliably pass those exceptions through cascade.executor.comms.msg

Define a method in this module `def ser(CascadeError) -> str` which basically returns repr,
and a des(str) -> CascadeError method, which tries to parse the string as '<CascadeError class name>(<detail>)',
if matches constructs the right class and detail and returns it, otherwise returns CascadeInternalError(input).
Don't worry about preserving the parent field, set it to None.

Inspect all places in the cascade.executor module where any of 
TaskFailure, ExecutorFailure, DatasetRetrieveFailure, DatasetTransmitFailure, DatasetPersistFailure
are created, and make sure that the `ser` method is used to build the `detail` of the respective message class
from the Exception that leads to it. But if there is no exception, put there `# TODO fill exception here` comment,
but don't change the code.
There may already be a `# TODO handle proper serde` comment -- remove it if you handle that case properly.

Then in the bridge code, there is a # TODO comment for deserialization -- thats where you need to invoke 
the des method you created.

Write a simple unit test that tests the ser-des method, put it to a tests/cascade/low/test_exceptions.py new file.
Don't change other tests.
After `just val` recipe passes, commit, but dont push.
"""
