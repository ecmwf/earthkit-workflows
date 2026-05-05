# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Runtime package management for worker processes.

Each worker runs in its own dedicated venv, created initially with earthkit-workflows
installed. PackagesEnv manages additional runtime pip installs into that venv as
requested by executed jobs.

The central challenge is that if a module is already imported in the process, a pip
install of a different version of that same module cannot take effect (the old version
stays loaded). PackagesEnv handles this in two ways:

1. Prevention: before calling pip, it pins already-imported modules to their current
   versions, and checks for spec conflicts up-front.
2. Detection: after pip completes, it verifies that no imported module ended up at a
   mismatched version.
"""

import importlib.metadata
import json
import logging
import os
import re
import subprocess
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, cast

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


def _earthkit_install_spec() -> str:
    """Returns the install spec for earthkit-workflows suitable for uv pip install.

    For editable/source installs (dev mode), uses the local source path directly.
    Otherwise, pins to the currently installed version.
    """
    ek_version = importlib.metadata.version("earthkit-workflows")
    try:
        dist = importlib.metadata.distribution("earthkit-workflows")
        direct_url_text = dist.read_text("direct_url.json")
        if direct_url_text:
            info = json.loads(direct_url_text)
            url = info.get("url", "")
            if url.startswith("file://") and info.get("dir_info", {}).get("editable", False):
                path = url[len("file://") :]
                logger.debug(f"earthkit-workflows is an editable install at {path}")
                return path
    except Exception as e:
        logger.debug(f"could not read direct_url.json for earthkit-workflows: {repr(e)}")
    return f"earthkit-workflows=={ek_version}"


def initial_venv_packages() -> list[str]:
    """Returns the list of packages to install into a fresh worker venv.

    Used by setup.py to create the initial venv, and by PackagesEnv to
    pre-populate its installed-packages state.
    """
    return [_earthkit_install_spec()]


def _parse_editable_paths_from_pip_output(pip_output: str) -> list[str]:
    """Extract local source paths for editable installs from pip/uv install output.

    We do this post-install rather than pre-install (e.g. scanning the -e specs) because
    uv may satisfy transitive dependencies via local file:// paths too -- for example when
    a workspace package declares another workspace package as a dependency, uv resolves it
    as an editable install without an explicit -e flag in our request.  Parsing the output
    captures all such recursive editable installs that we would otherwise miss.

    Matches lines like:
      + fiab-plugin-test==0.1.0 (from file:///home/.../fiab-plugin-test)
    """
    paths: list[str] = []
    for line in pip_output.splitlines():
        clean_line = line.strip()
        if not clean_line.startswith("+"):
            continue
        match = re.search(r"\(from file://([^)]+)\)", clean_line)
        if match:
            paths.append(match.group(1))
    return paths


def _extend_sys_path_for_editables(editable_paths: list[str]) -> None:
    """Extend sys.path with the src/ subdirectory of each editable install path.

    NOTE: This assumes the src-layout convention (PEP 517) where package code lives
    in <editable_path>/src/. The flat layout (code directly in <editable_path>/) is
    not handled. To support it, one could inspect the editable package's pyproject.toml
    for the [tool.setuptools.package-dir] / [tool.hatch.build.targets.wheel] source
    root setting, or fall back to <editable_path>/ itself when no src/ subdirectory
    is present.
    """
    for path in editable_paths:
        src_path = os.path.join(path, "src")
        if src_path not in sys.path:
            logger.debug(f"extending sys.path with editable install src path: {src_path}")
            sys.path.insert(0, src_path)
        else:
            logger.debug(f"editable install src path already in sys.path: {src_path}")


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

    Tracks installed packages and caches the module->dist_name mapping for imported
    modules. importlib.metadata.packages_distributions() is not cached by Python
    itself (it rescans all package metadata on every call), so we maintain our own
    permanent per-module cache to avoid repeated scans.

    We only protect already-imported modules from version changes: those are the
    only modules that cannot be safely re-loaded in a running process. Non-imported
    packages can be freely changed by pip.
    """

    def __init__(self) -> None:
        self.clean = True
        # dist_name -> version, tracks what has been installed into this venv
        # (pre-populated with the initial earthkit-workflows install from venv creation)
        self._installed: dict[str, Version] = {}
        for spec in initial_venv_packages():
            # parse "name==version" style specs; skip paths (editable installs have no "==")
            if "==" in spec:
                name, _, ver_str = spec.partition("==")
                try:
                    self._installed[name.strip()] = Version(ver_str.strip())
                except Exception:
                    pass
        # module_name -> dist_name (or None if no dist found).
        # Populated lazily via _populate_dist_cache(). None entries are kept permanently:
        # modules without a dist are typically oddities and gaining a dist post-install
        # is not a realistic scenario we need to handle.
        self._dist_name_cache: dict[str, str | None] = {}
        # Pre-populate cache for the initially installed packages so that packages_distributions()
        # is not needed for those modules when they are later imported.
        self._cache_dist_modules(self._installed)

    def _populate_dist_cache(self, mod_names: set[str]) -> None:
        """Batch-populate the dist_name cache for a set of module names.

        Falls back to packages_distributions() for modules not already cached by
        _cache_dist_modules(). packages_distributions() rescans all installed package
        metadata (~130ms on a typical venv) so we only call it when there are truly
        uncached modules that need lookup.
        """
        new_mods = {m for m in mod_names if m not in self._dist_name_cache and not _is_ignorable_module(m)}
        if not new_mods:
            return
        pkg_dist = importlib.metadata.packages_distributions()
        for mod in new_mods:
            maybe = pkg_dist.get(mod, [])
            self._dist_name_cache[mod] = maybe[0] if maybe else None

    def _cache_dist_modules(self, dist_names: Iterable[str]) -> None:
        """Pre-populate the dist_name cache from the dist->modules direction.

        Cheaper than packages_distributions() when only a few dists are known:
        each individual dist lookup takes ~0.5ms vs ~130ms for a full scan.
        Called after each install for the newly installed dists, and at __init__
        for the initially installed packages. This way, when those modules are later
        imported and _populate_dist_cache() is called, they are already cached and
        packages_distributions() is not invoked for them.
        """
        for dist_name in dist_names:
            try:
                handle = importlib.metadata.distribution(dist_name)
                top_level = handle.read_text("top_level.txt")
                if top_level:
                    modules = top_level.split()
                elif handle.files:
                    modules = list({path.parts[0] for path in handle.files if path.suffix == ".py"})
                else:
                    modules = []
                for mod in modules:
                    if not _is_ignorable_module(mod):
                        self._dist_name_cache[mod] = dist_name
            except Exception:
                pass

    def _import_pins(self) -> dict[str, Version]:
        """Return {dist_name: version} for all currently imported top-level modules.

        Version is fetched from importlib.metadata rather than module.__version__,
        because importlib.metadata gives the pip-installable dist version:
        - for torch, __version__ adds a local +cu126 tag that is not in the wheel name
        - for ecmwf libs, __version__ omits the 4th build counter that IS in the wheel name
        Using importlib.metadata gives us the correct installable version in both cases.
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

        We only protect imported modules because those are the only ones that cannot
        be safely changed in a running process.
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

        Uses the dist_name cache to map module names to distributions, iterating
        in the import->dist direction (rather than dist->modules) for efficiency.

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
        self._cache_dist_modules(newly_installed)

        install_issues = self._postinstall_verify(install_output)
        if install_issues:
            raise PkgInstallException(cast(list[PostverifyIssue | ResolutionIssue], install_issues), self.clean)

        # NOTE do *not* invalidate caches before postinstall verify, that could
        # hide some issues. But afterwards this is important to actually allow
        # importing the modules
        importlib.invalidate_caches()
        editable_paths = _parse_editable_paths_from_pip_output(install_output)
        if editable_paths:
            _extend_sys_path_for_editables(editable_paths)
        self.clean = False

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        return False
