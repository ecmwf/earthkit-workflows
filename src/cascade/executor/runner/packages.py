# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Extending venv with packages required by the executed job

Note that venv itself is left untouched after the run finishes -- we extend sys path
with a temporary directory and install in there

When extending, we check that none of the actually installed packages was not imported
already with a different version
"""

import importlib.metadata
import logging
import os
import re
import site
import subprocess
import sys
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Iterator, Literal, cast

from packaging.specifiers import SpecifierSet
from packaging.version import Version

logger = logging.getLogger(__name__)

class Commands:
    venv_command = lambda name: ["uv", "venv", name]
    install_command = lambda name: [
        "uv",
        "pip",
        "install",
        "--prefix",
        name,
        "--prerelease",
        "explicit",
    ]
    freeze_command = ["uv", "pip", "freeze"]


def run_command(command: list[str]) -> tuple[str, str]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as ex:
        raise ValueError(f"command failure: {ex}")
    if result.returncode != 0:
        msg = f"command failed with {result.returncode}. Stderr: {result.stderr}, Stdout: {result.stdout}, Args: {result.args}"
        logger.error(msg)
        raise ValueError(msg)
    return result.stdout, result.stderr

def new_venv() -> tempfile.TemporaryDirectory:
    """1. Creates a new temporary directory with a venv inside.
    2. Extends sys.path so that packages in that venv can be imported.
    """
    logger.debug("creating a new venv")
    td = tempfile.TemporaryDirectory(prefix="cascade_runner_venv_")
    # NOTE we create a venv instead of just plain directory, because some of the packages create files
    # outside of site-packages. Thus we then install with --prefix, not with --target
    run_command(Commands.venv_command(td.name))

    # NOTE not sure if getsitepackages was intended for this -- if issues, attempt replacing
    # with something like f"{td.name}/lib/python*/site-packages" + globbing
    extra_sp = site.getsitepackages(prefixes=[td.name])
    # NOTE this makes the explicit packages go first, in case of a different version
    logger.debug(f"extending sys.path with {extra_sp}")
    sys.path = extra_sp + sys.path
    logger.debug(f"new sys.path: {sys.path}")

    return td

def _parse_pip_install(pip_output: str) -> dict[str, Version]:
    """Assumed input like: 'Using Python 3.11.8 environment at: <venv>\nResolved 1 package in 5ms\nUninstalled 1 package in 12ms\nInstalled 1 package in 18ms\n - numpy==2.4.2\n + numpy==2.4.1\n'
    Provided output: {'numpy': '2.4.1'}"""
    rv = {}
    for line in pip_output.splitlines():
        clean_line = line.strip()
        if not clean_line.startswith("+"):
            continue
        
        parts = clean_line.lstrip("+ ").split("==")
        if len(parts) != 2:
            logger.warning(f"Suspicious pip output: {clean_line} -- ignoring!")
            continue
        try:
            rv[parts[0]] = Version(parts[1])
        except Exception as e:
            logger.warning(f"failed to parse package {parts[0]} version {parts[1]} due to {repr(e)}-- ignoring!")
    return rv

def _get_dist_modules(dist_name: str) -> list[str]:
    # TODO presumably cacheable
    try:
        handle = importlib.metadata.distribution(dist_name)
        top_level = handle.read_text('top_level.txt')
        if top_level:
            return top_level.split()
        elif handle.files:
            return list({path.parts[0] for path in handle.files if path.suffix == '.py'})
        else:
            logger.warning(f"neither files nor top level for {dist_name} -- ignoring")
            return []
    except Exception as e:
        logging.warning(f"Could not find metadata for installed package: {dist_name} due to {repr(e)} -- ignoring")
        return []

def _maybe_module_version(mod_name: str) -> Version|None:
    # TODO presumably cacheable, unless None
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
        if hasattr(mod, "__version__"):
            try:
                return Version(mod.__version__)
            except Exception as e:
                logger.warning(f"failed to parse module {mod_name} version {mod.__version__} due to {repr(e)}-- ignoring!")
        else:
            logging.warning(f"Module '{mod_name}' is loaded, but has no __version__ attribute -- ignoring")

@dataclass
class InstallIssue:
    dist_name: str
    desired_version: Version
    mod_issues: list[tuple[str, Version]]

class PostinstallException(BaseException):
    issues: list[InstallIssue]

    def __init__(self, issues: list[InstallIssue]) -> None:
        self.issues = issues

    def __str__(self) -> str:
        msg = repr(self.issues)
        return f"failed to install correctly: {msg[:1024]}{'...' if len(msg) > 1024 else ''}"



def _postinstall_verify(pip_output) -> list[InstallIssue]:
    installed_packages = _parse_pip_install(pip_output)
    rv = []

    for dist_name, desired_version in installed_packages.items():
        modules = _get_dist_modules(dist_name)
        versions = [(m, _maybe_module_version(m)) for m in modules]
        # NOTE eg torch is a bit of an issue here:
        # pip reports + torch==2.7.1, but torch module version declares 2.7.1+cu126
        # we have no real means of deciding whether this is correct or not -- but we make the assumption
        # that checkpoints et cetera won't ever depend on this particular build discriminator => .base_version
        mod_issues = [(m, v) for m, v in versions if v and v.base_version != desired_version.base_version] 
        if mod_issues:
            rv.append(InstallIssue(dist_name=dist_name, desired_version=desired_version, mod_issues=mod_issues))

    return rv

def _prefer_installed(packages: list[str]) -> Iterator[str]:
    """If a package is desired to be installed but is not exactly pinned,
    we will inspect pip freeze to see if it is already installed, and inject
    the pin otherwise. This is default `uv` behaviour, but remember we
    override --prefix, thus uv has no way of knowing. We thus have to explicate.
    """
    # NOTE this does not work for transitive dependencies. We may
    # decide to drop the whole --prefix business, and completely switch
    # over to the new venv even before doing any install

    installed_raw, _ = run_command(Commands.freeze_command)
    def maybe_tuple(kv: str) -> None|tuple[str, str]:
        if kv.startswith('-e'):
            return kv.rsplit('/', 1)[1], '--editable'
        elif '@ git' in kv:
            return kv.split('@', 1)[0], '--git'
        elif '==' in kv:
            return cast(tuple[str, str], kv.split('==', 1))
        else:
            logger.warning(f"unable to discern package install {kv}")
            return None
    _installed = (maybe_tuple(kv) for kv in installed_raw.splitlines() if kv)
    installed = dict(tup_or_none for tup_or_none in _installed if tup_or_none)
    for package_spec in packages:
        try:
            parts = re.split(r'([<>=!~].*)', package_spec)
            package = parts[0]
            if package not in installed:
                yield package
            elif len(parts) == 1:
                if installed[package] == '--editable' or installed[package] == '--git':
                    continue
                yield f"{package}=={installed[package]}"
            else:
                specifier = SpecifierSet(parts[1].strip())
                # NOTE in case of mismatch we just warn because we dont know if the module was imported already
                # NOTE we dont check for import because we need to after install *anyway*
                # NOTE for editable + explicit constraint, we leave it to uv -- imo unclear
                if installed[package] == '--editable' or installed[package] == '--git':
                    logger.warning(f"will upgrade a package {package} -- may cause issues in post-verify")
                    yield package_spec
                elif Version(installed[package]) in specifier:
                    yield f"{package}=={installed[package]}"
                else:
                    logger.warning(f"will upgrade a package {package} -- may cause issues in post-verify")
                    yield package_spec
        except Exception as e:
            logger.warning(f"failed to discern preference for package {package} -- continuing")
            yield package


class PackagesEnv(AbstractContextManager):
    def __init__(self) -> None:
        self.td: tempfile.TemporaryDirectory | None = None

    def extend(self, packages: list[str]) -> None:
        if not packages:
            return
        if self.td is None:
            self.td = new_venv()
        packages = list(_prefer_installed(packages))

        logger.debug(f"installing {len(packages)} packages")
        logger.debug(f"installing packages: {','.join(packages)}")
        install_command = Commands.install_command(self.td.name)
        if os.environ.get("VENV_OFFLINE", "") == "YES":
            install_command += ["--offline"]
        if cache_dir := os.environ.get("VENV_CACHE", ""):
            install_command += ["--cache-dir", cache_dir]
        install_command.extend(set(packages))
        logger.debug(f"running install command: {' '.join(install_command)}")
        _, install_output = run_command(install_command)
        logger.debug(f"install result: {install_output}")

        # NOTE this is wrong because we would need to reliably unload modules, but that apparently aint possible in python
        # importlib.invalidate_caches()
        install_issues = _postinstall_verify(install_output)
        if install_issues:
            raise PostinstallException(install_issues)

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        sys.path = [
            p for p in sys.path if self.td is None or not p.startswith(self.td.name)
        ]
        if self.td is not None:
            self.td.cleanup()
        return False
