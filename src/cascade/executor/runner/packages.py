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
from typing import Callable, Iterator, Literal, cast

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from typing_extensions import Self

from cascade.low.exceptions import CascadeInternalError, CascadeUserError

logger = logging.getLogger(__name__)


_python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


class Commands:
    venv_command = lambda name: ["uv", "venv", "--python", _python_version, name]
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


@dataclass
class PostverifyIssue:
    """Coming from post-install check of installed modules"""

    dist_name: str
    desired_version: Version
    mod_issues: list[tuple[str, Version]]


@dataclass
class ResolutionIssue:
    """Coming from pip when receiving conflicting instructions"""

    because: str


class PkgInstallException(BaseException):
    issues: list[PostverifyIssue | ResolutionIssue]
    was_clean: bool

    def __init__(self, issues: list[PostverifyIssue | ResolutionIssue], was_clean: bool) -> None:
        self.issues = issues
        self.was_clean = was_clean

    def __str__(self) -> str:
        return f"failed to install correctly: {repr(self.issues)}, {self.was_clean=}"

    @classmethod
    def from_pip(cls, pip_stderr: str, was_clean: bool) -> Self | None:
        # TODO improve
        intro = "No solution found when resolving dependencies"
        pref = "Because "
        suff = ", we can conclude"
        if intro in pip_stderr and pref in pip_stderr:
            p1 = pip_stderr.split(pref, 1)[1]
            if suff in p1:
                l = ResolutionIssue(p1.split(suff, 1)[0])
                return cls([l], was_clean)
        return None


def run_command(command: list[str], checker: Callable[[subprocess.CompletedProcess], None]) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as ex:
        # either badly deployed or code bug of calling bad command -> InternalError
        raise CascadeInternalError(f"command failure: {ex}", parent=ex) from ex
    checker(result)
    return result


def check_run_result(result: subprocess.CompletedProcess) -> None:
    if result.returncode != 0:
        msg = f"command failed with {result.returncode}. Stderr: {result.stderr}, Stdout: {result.stdout}, Args: {result.args}"
        logger.error(msg)
        raise CascadeInternalError(msg)


def check_install_result(result: subprocess.CompletedProcess, was_clean: bool) -> None:
    if result.returncode != 0:
        msg = f"command failed with {result.returncode}. Stderr: {result.stderr}, Stdout: {result.stdout}, Args: {result.args}"
        logger.error(msg)
        if "was not found in the package registry" in result.stderr:
            # presumably bad job env -> UserError
            raise CascadeUserError(msg)
        elif (ex := PkgInstallException.from_pip(result.stderr, was_clean)) is not None:
            # possibly conflict of user req with installed
            raise ex
        else:
            # no idea -> InternalError
            raise CascadeInternalError(msg)


def new_venv() -> tempfile.TemporaryDirectory:
    """1. Creates a new temporary directory with a venv inside.
    2. Extends sys.path so that packages in that venv can be imported.
    """
    logger.debug("creating a new venv")
    td = tempfile.TemporaryDirectory(prefix="cascade_runner_venv_")
    # NOTE we create a venv instead of just plain directory, because some of the packages create files
    # outside of site-packages. Thus we then install with --prefix, not with --target
    run_command(Commands.venv_command(td.name), check_run_result)

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
    """From package name like 'earthkit-workflows' get the top level importible like 'earthkit', 'cascade'"""
    # TODO presumably cacheable
    try:
        handle = importlib.metadata.distribution(dist_name)
        top_level = handle.read_text("top_level.txt")
        if top_level:
            return top_level.split()
        elif handle.files:
            return list({path.parts[0] for path in handle.files if path.suffix == ".py"})
        else:
            logger.warning(f"neither files nor top level for {dist_name} -- ignoring")
            return []
    except Exception as e:
        logging.warning(f"Could not find metadata for installed package: {dist_name} due to {repr(e)} -- ignoring")
        return []


def _maybe_module_dist(module_name: str) -> tuple[str, Version] | None:
    """From a module name like 'cascade' get the pip installable like 'earthkit-workflows' and version"""
    # TODO presumably cacheable, unless None
    # NOTE we dont rely on __version__, eg:
    # - for torch it adds the +cu310
    # - for mir it drops the fourth version
    # and neither leads to an installable wheel.
    # NOTE that because of the 4-number version of ecmwf libs, we dont use the base_version here
    lookup = importlib.metadata.packages_distributions()
    maybe = lookup.get(module_name, [])
    if len(maybe) >= 1:
        distName = maybe[0]
        try:
            return distName, Version(importlib.metadata.version(distName))
        except importlib.metadata.PackageNotFoundError:
            return None
    else:
        return None


def _maybe_imported_version(mod_name: str) -> Version | None:
    # TODO presumably cacheable, unless None
    if mod_name in ("eccodes", "gribapi"):
        # the eccodes/gribapi __version__ is wrong, reporting that of eccodeslib
        # => we must go to the importlib. This *invalidates* the post install
        # check -- TODO after eccodes wheel fixed, remove this
        return None
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
        if hasattr(mod, "__version__"):
            try:
                return Version(mod.__version__)
            except Exception as e:
                logger.debug(f"failed to parse module {mod_name} version {mod.__version__} due to {repr(e)}-- ignoring!")
        else:
            logging.debug(f"Module '{mod_name}' is loaded, but has no __version__ attribute -- ignoring")


def _postinstall_verify(pip_output) -> list[PostverifyIssue]:
    installed_packages = _parse_pip_install(pip_output)
    rv = []

    for dist_name, desired_version in installed_packages.items():
        modules = _get_dist_modules(dist_name)
        versions = [(m, _maybe_imported_version(m)) for m in modules]
        # NOTE eg torch is a bit of an issue here:
        # pip reports + torch==2.7.1, but torch module version declares 2.7.1+cu126
        # we have no real means of deciding whether this is correct or not -- but we make the assumption
        # that checkpoints et cetera won't ever depend on this particular build discriminator => .base_version
        # For ecmwf libs, this hides the 4th version which is the build counter -- but that is presumably
        # legitimate as well, as it means the api of the underlying library being the same
        mod_issues = [(m, v) for m, v in versions if v and v.base_version != desired_version.base_version]
        if mod_issues:
            rv.append(PostverifyIssue(dist_name=dist_name, desired_version=desired_version, mod_issues=mod_issues))

    return rv


def _prefer_installed(packages: list[str]) -> Iterator[str]:
    """If a package is desired to be installed but is not exactly pinned,
    we will inspect pip freeze to see if it is already installed, and inject
    the pin otherwise. This is default `uv` behaviour, but remember we
    override --prefix, thus uv has no way of knowing. We thus have to explicate.

    Furthermore, we will extend the list with already-imported packages. This
    moves some PostInstall verify issues into the Install phase, and importantly,
    for fresh workers and unavoidable dependencies (like orjson or pydantic),
    forces a pin to prevent unbound resolution infinite loops.
    """
    # NOTE the pip-freeze thing doesnt work for transitive dependencies. We may
    # decide to drop the whole --prefix business, and completely switch
    # over to the new venv even before doing any install

    installed_raw = run_command(Commands.freeze_command, check_run_result).stdout

    def maybe_tuple(kv: str) -> None | tuple[str, str]:
        if kv.startswith("-e"):
            return kv.rsplit("/", 1)[1], "--editable"
        elif "@ git" in kv:
            return kv.split("@", 1)[0], "--git"
        elif "==" in kv:
            return cast(tuple[str, str], kv.split("==", 1))
        else:
            logger.warning(f"unable to discern package install {kv}")
            return None

    _installed = (maybe_tuple(kv) for kv in installed_raw.splitlines() if kv)
    installed = dict(tup_or_none for tup_or_none in _installed if tup_or_none)
    for package_spec in packages:
        try:
            parts = re.split(r"([<>=!~].*)", package_spec)
            package = parts[0]
            if package not in installed:
                yield package_spec
            elif len(parts) == 1:
                if installed[package] == "--editable" or installed[package] == "--git":
                    continue
                yield f"{package}=={installed[package]}"
            else:
                specifier = SpecifierSet(parts[1].strip())
                # NOTE in case of mismatch we just warn because we dont know if the module was imported already
                # NOTE we dont check for import because we need to after install *anyway*
                # NOTE for editable + explicit constraint, we leave it to uv -- imo unclear
                if installed[package] == "--editable" or installed[package] == "--git":
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

    def _is_ignorable(top_level: str):
        is_stdlib = top_level in sys.stdlib_module_names
        is_local = top_level.startswith("_")  # for __main__, __editable, _distutils_hack, etc
        return is_stdlib or is_local

    imported = set(name.split(".")[0] for name in sys.modules)
    for name in imported:
        if not _is_ignorable(name):
            maybe_dist_version = _maybe_module_dist(name)
            if maybe_dist_version is None:
                logger.debug(f"failed to provide install pin for imported: {name=}, {maybe_dist_version=}")
            else:
                dist, version = maybe_dist_version
                # editable/local installs such as that of cascade itself should not be put to
                # the pip command as that wont be resolvable! We need to rely on caching here
                if not version.dev:
                    yield f"{dist}=={version}"


class PackagesEnv(AbstractContextManager):
    def __init__(self) -> None:
        self.td: tempfile.TemporaryDirectory | None = None
        self.clean = True

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
        install_output = run_command(install_command, lambda r: check_install_result(r, self.clean)).stderr
        logger.debug(f"install result: {install_output}")

        install_issues = _postinstall_verify(install_output)
        if install_issues:
            install_issues = cast(list[PostverifyIssue | ResolutionIssue], install_issues)
            raise PkgInstallException(install_issues, self.clean)

        # NOTE do *not* invalidate caches before postinstall verify, that could
        # hide some issues. But afterwards this is important to actually allow
        # importing the modules
        importlib.invalidate_caches()
        self.clean = False

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        sys.path = [p for p in sys.path if self.td is None or not p.startswith(self.td.name)]
        if self.td is not None:
            self.td.cleanup()
        return False
