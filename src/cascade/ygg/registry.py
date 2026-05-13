# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from cascade.low.exceptions import CascadeInternalError
from cascade.ygg.types import HostEndpoints, HostId, Lane


class HostRegistry:
    def __init__(self) -> None:
        self._hosts: dict[HostId, HostEndpoints] = {}

    def register(self, host: HostId, endpoints: HostEndpoints) -> None:
        self._hosts[host] = endpoints

    def unregister(self, host: HostId) -> None:
        self._hosts.pop(host, None)

    def resolve(self, host: HostId, lane: Lane) -> str:
        endpoints = self._hosts.get(host)
        if endpoints is None:
            raise CascadeInternalError(f"host not registered in ygg registry: {host}")
        if lane == "control":
            return endpoints.control
        if endpoints.bulk is None:
            raise CascadeInternalError(f"host {host} has no bulk endpoint in ygg registry")
        return endpoints.bulk

    def hosts(self) -> tuple[HostId, ...]:
        return tuple(self._hosts.keys())
