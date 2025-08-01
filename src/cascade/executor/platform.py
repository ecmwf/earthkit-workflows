# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Macos-vs-Linux specific code"""

import socket
import sys


def get_bindabble_self():
    """Returns a hostname such that zmq can bind to it"""

    if sys.platform == "darwin":
        # NOTE on macos, getfqdn usually returns like '66246.local', which can't then be bound to
        # This is a stopper for running a cluster of macos devices -- but we don't plan that yet
        return "localhost"
    else:
        # NOTE not sure if fqdn or hostname is better -- all we need is for it to be resolvable within cluster
        return socket.gethostname()  # socket.getfqdn()
