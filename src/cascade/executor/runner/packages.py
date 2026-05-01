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
import subprocess
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, Literal, cast

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from typing_extensions import Self

from cascade.low.exceptions import CascadeInternalError, CascadeUserError

logger = logging.getLogger(__name__)


_python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


class Commands:
    venv_command = lambda name: ["uv", "venv", "--python", _python_version, name]
    install_command = [
        "uv",
        "pip",
        "install",
        "--prerelease",
        "explicit",
    ]


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


def _maybe_imported_version(mod_name: str) -> Version | None:
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
    return None


def _is_ignorable_module(top_level: str) -> bool:
    # stdlib modules and private/internal modules (eg __main__, _distutils_hack)
    return top_level in sys.stdlib_module_names or top_level.startswith("_")


def _is_ignorable_dist(dist_name: str, dist_version: Version) -> bool:
    # editable/local installs such as that of cascade itself should not be put to
    # the pip command as that wont be resolvable! We need to rely on caching here
    if dist_version.dev is not None or dist_version.local is not None:
        return True
    if dist_version == Version("0.0.0"):
        return True
    try:
        origin = importlib.metadata.distribution(dist_name)
        if hasattr(origin, "url") and isinstance(origin.url, str) and origin.url.startswith("file://"):
            return True
    except Exception:
        pass
    return False


class PackagesEnv(AbstractContextManager):
    """Context manager responsible for runtime pip installs.

    Tracks what we install and caches the module->dist_name mapping for imported
    modules. This avoids repeated expensive packages_distributions() scans and
    eliminates the need for a `pip freeze` call on every extend().

    We only protect already-imported modules from version changes: those are the
    only modules that cannot be safely re-loaded in a running process. Non-imported
    packages can be freely changed by pip.
    """

    def __init__(self) -> None:
        self.clean = True
        # dist_name -> version, accumulated from parsed pip output of each extend() call
        self._installed: dict[str, Version] = {}
        # module_name -> dist_name (or None if no dist found), permanently cached per module.
        # None entries are cleared after each install in case a new dist now provides them.
        self._dist_name_cache: dict[str, str | None] = {}

    def _populate_dist_cache(self, mod_names: set[str]) -> None:
        """Batch-populate the dist_name cache for a set of module names.

        Calls packages_distributions() once for all uncached modules, avoiding
        repeated expensive metadata scans.
        """
        new_mods = {m for m in mod_names if m not in self._dist_name_cache and not _is_ignorable_module(m)}
        if not new_mods:
            return
        pkg_dist = importlib.metadata.packages_distributions()
        for mod in new_mods:
            maybe = pkg_dist.get(mod, [])
            self._dist_name_cache[mod] = maybe[0] if maybe else None

    def _import_pins(self) -> dict[str, Version]:
        """Return {dist_name: version} for all currently imported top-level modules.

        Version is fetched fresh from importlib.metadata (not from __version__),
        which gives the installed dist version rather than the potentially-decorated
        __version__ (eg torch adds +cu126, ecmwf libs add a 4th build counter).
        """
        imported = set(name.split(".")[0] for name in sys.modules)
        self._populate_dist_cache(imported)

        pins: dict[str, Version] = {}
        for name in imported:
            if _is_ignorable_module(name):
                continue
            dist_name = self._dist_name_cache.get(name)
            if dist_name is None:
                logger.debug(f"failed to provide install pin for imported: {name!r}")
                continue
            try:
                version = Version(importlib.metadata.version(dist_name))
            except importlib.metadata.PackageNotFoundError:
                continue
            if not _is_ignorable_dist(dist_name, version):
                pins[dist_name] = version
        return pins

    def _prefer_installed(self, packages: list[str]) -> list[str]:
        """Pin user-requested packages to their currently-imported version where
        compatible, and append pins for all other imported distributions.

        This replaces the old `uv pip freeze` approach. We only protect imported
        modules because those are the only ones that cannot be safely changed in a
        running process.
        """
        import_pins = self._import_pins()
        result: list[str] = []

        for package_spec in packages:
            try:
                parts = re.split(r"([<>=!~].*)", package_spec)
                package = parts[0]
                if package in import_pins:
                    pin_version = import_pins[package]
                    if len(parts) == 1:
                        # bare name with no constraint -> pin to imported version
                        result.append(f"{package}=={pin_version}")
                    else:
                        specifier = SpecifierSet(parts[1].strip())
                        if pin_version in specifier:
                            # constraint is compatible -> pin to imported version
                            result.append(f"{package}=={pin_version}")
                        else:
                            # constraint conflicts with imported version -> pass through and let
                            # _check_conflicts or post-install verify catch it
                            logger.warning(f"will upgrade a package {package} -- may cause issues in post-verify")
                            result.append(package_spec)
                else:
                    result.append(package_spec)
            except Exception:
                logger.warning(f"failed to discern preference for package {package_spec} -- continuing")
                result.append(package_spec)

        # Append pins for all imported distributions not already covered above.
        # This prevents pip from silently downgrading or upgrading loaded modules
        # when resolving transitive dependencies.
        for dist, version in import_pins.items():
            result.append(f"{dist}=={version}")

        return result

    def _check_conflicts(self, packages: list[str]) -> list[ResolutionIssue]:
        """Pre-pip conflict detection.

        Groups package specs by name and checks whether any exact == pin is
        incompatible with another spec for the same package. Emits at most one
        ResolutionIssue per conflicting package name.
        """
        specs: dict[str, list[SpecifierSet]] = {}
        for pkg in packages:
            try:
                parts = re.split(r"([<>=!~].*)", pkg)
                name = parts[0]
                spec = SpecifierSet(parts[1].strip()) if len(parts) > 1 else SpecifierSet("")
                specs.setdefault(name, []).append(spec)
            except Exception:
                continue

        issues: list[ResolutionIssue] = []
        for name, specsets in specs.items():
            if len(specsets) <= 1:
                continue
            conflict: str | None = None
            for i, si in enumerate(specsets):
                for spec_item in si:
                    if spec_item.operator == "==":
                        try:
                            pin_ver = Version(spec_item.version)
                            for j, sj in enumerate(specsets):
                                if j != i and str(sj) and pin_ver not in sj:
                                    conflict = f"{name} pinned to {pin_ver} but {sj} also required"
                                    break
                        except Exception:
                            pass
                    if conflict:
                        break
                if conflict:
                    break
            if conflict:
                issues.append(ResolutionIssue(conflict))

        return issues

    def _postinstall_verify(self, pip_output: str) -> list[PostverifyIssue]:
        """Check that no already-imported module has a version mismatch with what pip just installed.

        Uses the dist_name cache (populated by _prefer_installed -> _import_pins) to
        map module names to distributions without an extra _get_dist_modules() call.

        We compare base_version (stripping local/build parts like +cu126) because pip
        reports the base wheel version while __version__ may include build discriminators.
        """
        installed_packages = _parse_pip_install(pip_output)
        if not installed_packages:
            return []

        imported = set(name.split(".")[0] for name in sys.modules)
        self._populate_dist_cache(imported)

        dist_to_issues: dict[str, list[tuple[str, Version]]] = {}
        for mod_name in imported:
            if _is_ignorable_module(mod_name):
                continue
            dist_name = self._dist_name_cache.get(mod_name)
            if dist_name is None or dist_name not in installed_packages:
                continue
            desired_version = installed_packages[dist_name]
            mod_ver = _maybe_imported_version(mod_name)
            if mod_ver is None:
                continue
            if mod_ver.base_version != desired_version.base_version:
                dist_to_issues.setdefault(dist_name, []).append((mod_name, mod_ver))

        return [
            PostverifyIssue(dist_name=dist_name, desired_version=installed_packages[dist_name], mod_issues=mod_issues)
            for dist_name, mod_issues in dist_to_issues.items()
        ]

    def extend(self, packages: list[str]) -> None:
        if not packages:
            return

        packages = list(self._prefer_installed(packages))

        conflicts = self._check_conflicts(packages)
        if conflicts:
            raise PkgInstallException(cast(list[PostverifyIssue | ResolutionIssue], conflicts), self.clean)

        logger.debug(f"installing {len(packages)} packages")
        logger.debug(f"installing packages: {','.join(packages)}")
        install_command = list(Commands.install_command)
        if os.environ.get("VENV_OFFLINE", "") == "YES":
            install_command += ["--offline"]
        if cache_dir := os.environ.get("VENV_CACHE", ""):
            install_command += ["--cache-dir", cache_dir]
        install_command.extend(set(packages))
        logger.debug(f"running install command: {' '.join(install_command)}")
        install_output = run_command(install_command, lambda r: check_install_result(r, self.clean)).stderr
        logger.debug(f"install result: {install_output}")

        newly_installed = _parse_pip_install(install_output)
        self._installed.update(newly_installed)

        # Clear None cache entries for modules that might now be provided by newly
        # installed packages. Non-None entries are permanent (dist names don't change).
        if newly_installed:
            self._dist_name_cache = {k: v for k, v in self._dist_name_cache.items() if v is not None}

        install_issues = self._postinstall_verify(install_output)
        if install_issues:
            raise PkgInstallException(cast(list[PostverifyIssue | ResolutionIssue], install_issues), self.clean)

        # NOTE do *not* invalidate caches before postinstall verify, that could
        # hide some issues. But afterwards this is important to actually allow
        # importing the modules
        importlib.invalidate_caches()
        self.clean = False

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        return False
