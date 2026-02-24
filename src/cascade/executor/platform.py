# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Macos-vs-Linux specific code"""

import fcntl
import multiprocessing as mp
import os
import socket
import sys
import typing


def get_bindabble_self():
    """Returns a hostname such that zmq can bind to it"""

    if sys.platform == "darwin":
        # NOTE on macos, getfqdn usually returns like '66246.local', which can't then be bound to
        # This is a stopper for running a cluster of macos devices -- but we don't plan that yet
        return "localhost"
    else:
        # NOTE not sure if fqdn or hostname is better -- all we need is for it to be resolvable within cluster
        return socket.gethostname()  # socket.getfqdn()


def gpu_init(worker_num: int):
    if sys.platform != "darwin":
        # TODO there is implicit coupling with executor.executor and cascade.main -- make it cleaner!
        gpus = int(os.environ.get("CASCADE_GPU_COUNT", "0"))
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            str(worker_num) if worker_num < gpus else ""
        )
    else:
        pass  # no macos specific gpu init due to unified mem model


MpSituation = typing.Literal[
    "worker", "executor-loc", "executor-aux", "gateway", "other"
]
_MpSituation = typing.get_args(MpSituation)


def get_mp_ctx(situation: MpSituation) -> mp.context.ForkContext|mp.context.SpawnContext|mp.context.ForkServerContext:
    """Generally, forking is safe everywhere as we try to be careful not to
    initialize non-safe objects prior to forking. However, combination of
    mac + mps + anemoi + pickled callables causes xpc_error_connection_invalid,
    thus we stick to spawn on darwin platforms. Setting
    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES helps some but not fully.

    We distinguish in which situation is this method called, as fine graining
    may be (eventually) possible
    """

    if situation not in _MpSituation:
        raise TypeError(f"{situation=} is not in {_MpSituation}")
    if sys.platform == "darwin":
        return mp.get_context("spawn")
    elif situation == "executor-loc":
        # NOTE in the case of executor being launched locally, from eg cascade main
        # after it has constructed a jobInstance, there is a chance of the process
        # being tainted with some imports such as earthkit.workflows.fluent which in
        # turn brings in numpy -- therefore, we cannot allow to fork
        return mp.get_context("forkserver")
    else:
        return mp.get_context("fork")

# NOTE not really perftested, more like for fun
if sys.platform == "darwin":
    def advise_seqread(fd: int) -> None:
        # after https://github.com/python/cpython/issues/113092 merged, use that instead
        F_RDADVISE=44
        # taken from https://github.com/phracker/MacOSX-SDKs/blob/master/MacOSX10.8.sdk/System/Library/Frameworks/Kernel.framework/Versions/A/Headers/sys/fcntl.h#L235
        try:
            fcntl.fcntl(fd, F_RDADVISE, 1)
        except OSError:
            pass
else:
    def advise_seqread(fd: int) -> None:
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
        except OSError:
            pass
