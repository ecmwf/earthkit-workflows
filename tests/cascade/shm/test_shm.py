# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
from multiprocessing import Process

import pytest

import cascade.shm.api as api
import cascade.shm.client as client
import cascade.shm.server as server


def test_shm_simple():
    port = 12345
    api.publish_client_port(port)
    serverP = Process(target=server.entrypoint, args=(port,))
    serverP.start()
    try:
        client.ensure()

        buf = client.allocate("my_data", 7, "some_deser_fun")
        buf.view()[:] = b"a" * 7
        buf.close()

        with pytest.raises(ValueError, match=r"shm already closed"):
            buf.view()

        with pytest.raises(ValueError, match=re.escape("KeyError('missing_data')")):
            client.get("missing_data")

        buf = client.get("my_data")
        assert buf.view() == b"a" * 7
        assert buf.deser_fun == "some_deser_fun"
        with pytest.raises(TypeError, match="cannot modify read-only memory"):
            buf.view()[:] = b"b" * 7
        buf.close()

    finally:
        # test unclean exit:
        serverP.terminate()


def test_shm_disk():
    port = 12346
    capacity = 4
    api.publish_client_port(port)
    serverP = Process(target=server.entrypoint, args=(port, capacity))
    serverP.start()
    try:
        client.ensure()

        buf = client.allocate("m1", 3, "some_deser_fun")
        buf.view()[:] = b"a" * 3
        buf.close()

        assert client.get_free_space() == 1

        with pytest.raises(ValueError, match="capacity exceeded"):
            client.allocate("m2", 5, "some_deser_fun")

        # free space = 1 => allocate 2 first causes persist of m1
        # thus free space will be first 4, then after allocate 2
        buf = client.allocate("m2", 2, "some_deser_fun")
        buf.view()[:] = b"a" * 2
        buf.close()

        assert client.get_free_space() == 2

        buf = client.get("m1")
        assert buf.view() == b"a" * 3
        # m1 was brought back up -> m2 persisted to disk, free space goes 2 -> 4 -> 1
        assert client.get_free_space() == 1

    finally:
        # test clean exit:
        client.shutdown()
        serverP.join()
