"""Fresh source imports for explicitly bounded experiment source packages.

Only trusted, import-side-effect-free source is supported. This is not a Python
sandbox or a transaction over arbitrary module globals. The app must close its
experiment tabs and exclude operations before calling ``load``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import keyword
import sys
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import get_ident
from types import CodeType, ModuleType

from zcu_tools.gui.app.measure.catalog import CatalogReloadError, PreparedCatalogReload
from zcu_tools.gui.app.measure.registry import Registry


def _within(name: str, root: str) -> bool:
    return name == root or name.startswith(root + ".")


def _within_any(name: str, roots: tuple[str, ...]) -> bool:
    return any(_within(name, root) for root in roots)


def _valid_module_name(name: str) -> bool:
    return all(
        part.isidentifier() and not keyword.iskeyword(part) for part in name.split(".")
    )


@dataclass(frozen=True)
class SourcePackage:
    """One source package whose files the catalog loader monitors.

    ``namespace`` is a nonempty dotted Python package name, such as ``my_lab``.
    ``package_root`` is the existing directory containing that package's source,
    not its parent directory. The loader resolves relative paths at construction.
    Source namespaces must not overlap; the loader rejects invalid declarations.
    """

    namespace: str
    package_root: Path


@dataclass(frozen=True)
class _Source:
    path: Path
    content: bytes
    package: bool


@dataclass(frozen=True)
class _Revision:
    token: PreparedCatalogReload
    sources: dict[str, _Source]
    code: dict[str, CodeType]


class _SourceLoader(importlib.abc.Loader):
    def __init__(
        self,
        source: _Source,
        code: CodeType,
        retained: dict[str, ModuleType],
    ) -> None:
        self._source = source
        self._code = code
        self._retained = retained

    def exec_module(self, module: ModuleType) -> None:
        module.__file__ = str(self._source.path)
        # Cached children are not re-executed by Python. Reattach them before
        # package code runs, including ``from package import retained_child``.
        for name, child in self._retained.items():
            parent, _, leaf = name.rpartition(".")
            if parent == module.__name__:
                setattr(module, leaf, child)
        exec(self._code, module.__dict__)


class _SourceFinder(importlib.abc.MetaPathFinder):
    def __init__(
        self,
        namespaces: tuple[str, ...],
        owns: Callable[[str], bool],
        revision: _Revision,
        fixed_names: frozenset[str],
        retained: dict[str, ModuleType],
    ) -> None:
        self._namespaces = namespaces
        self._owns = owns
        self._revision = revision
        self._fixed_names = fixed_names
        self._retained = retained

    def find_spec(self, fullname: str, _path=None, _target=None):
        if not _within_any(fullname, self._namespaces):
            return None
        if not self._owns(fullname):
            if fullname not in self._fixed_names:
                raise CatalogReloadError(
                    f"New fixed dependency {fullname!r}; restart the app",
                    restart_required=True,
                )
            return None
        source = self._revision.sources.get(fullname)
        if source is None:
            raise ModuleNotFoundError(f"No source for reload module {fullname!r}")
        loader = _SourceLoader(source, self._revision.code[fullname], self._retained)
        return importlib.util.spec_from_file_location(
            fullname,
            source.path,
            loader=loader,
            submodule_search_locations=[str(source.path.parent)]
            if source.package
            else None,
        )


class SourceExperimentCatalogLoader:
    """Load a source catalog exporting ``register_all(registry)``.

    ``sources`` declares all monitored framework and user source packages.
    ``reload_modules`` contains nonempty module prefixes allowed to be reloaded.
    ``preserved_modules`` contains fixed prefixes, taking precedence over reload
    prefixes even when nested within them. All other source files are also fixed.
    ``catalog_module`` must be inside the effective reload scope. Invalid names,
    overlapping source namespaces, missing directories and foreign scopes raise
    ``ValueError`` at construction; unreadable source raises ``OSError``.

    A new unused fixed file is harmless; importing a new fixed module requires
    restart. Each plan belongs to this loader and is single-use. Construct and
    invoke on the app owner thread. The loader does not discover package roots,
    publish registries, close tabs or restore arbitrary import side effects.
    """

    def __init__(
        self,
        *,
        sources: tuple[SourcePackage, ...],
        reload_modules: tuple[str, ...],
        preserved_modules: tuple[str, ...],
        catalog_module: str,
    ) -> None:
        if not sources or not reload_modules:
            raise ValueError("Source packages and nonempty reload scope are required")
        self._sources = tuple(
            SourcePackage(source.namespace, source.package_root.resolve())
            for source in sources
        )
        self._namespaces = tuple(source.namespace for source in self._sources)
        for index, source in enumerate(self._sources):
            if not _valid_module_name(source.namespace):
                raise ValueError(
                    "Source namespaces must be dotted Python package names"
                )
            if not source.package_root.is_dir():
                raise ValueError("Each source package must have an existing directory")
            if any(
                _within(source.namespace, other) or _within(other, source.namespace)
                for other in self._namespaces[:index]
            ):
                raise ValueError("Source namespaces must not overlap")
        self._reload_modules = reload_modules
        self._preserved_modules = preserved_modules
        self._catalog_module = catalog_module
        if any(
            not _valid_module_name(name) or not _within_any(name, self._namespaces)
            for name in (*reload_modules, *preserved_modules, catalog_module)
        ):
            raise ValueError("All scopes must belong to a declared source namespace")
        if not self._owns(catalog_module):
            raise ValueError("The catalog must belong to the reload scope")
        self._owner = get_ident()
        self._loading = False
        self._prepared: _Revision | None = None
        self._fixed = {
            name: sha256(source.content).digest()
            for name, source in self._inventory().items()
            if not self._owns(name)
        }

    def _owns(self, name: str) -> bool:
        return _within_any(name, self._reload_modules) and not _within_any(
            name, self._preserved_modules
        )

    def _assert_owner(self) -> None:
        if get_ident() != self._owner:
            raise CatalogReloadError(
                "Reload must run on its owner thread", restart_required=True
            )
        if self._loading:
            raise CatalogReloadError("A catalog load is already in progress")

    def _inventory(self) -> dict[str, _Source]:
        inventory: dict[str, _Source] = {}
        for source in self._sources:
            for path in source.package_root.rglob("*.py"):
                parts = path.relative_to(source.package_root).with_suffix("").parts
                package = parts[-1] == "__init__"
                if package:
                    parts = parts[:-1]
                name = ".".join((source.namespace, *parts))
                if name in inventory:
                    raise CatalogReloadError(f"Ambiguous source module {name!r}")
                inventory[name] = _Source(path, path.read_bytes(), package)
        return inventory

    def _check_fixed(self, sources: dict[str, _Source]) -> None:
        for name, digest in self._fixed.items():
            source = sources.get(name)
            if source is None or sha256(source.content).digest() != digest:
                raise CatalogReloadError(
                    f"Fixed source {name!r} changed; restart the app",
                    restart_required=True,
                )
        for name in tuple(sys.modules):
            if (
                _within_any(name, self._namespaces)
                and not self._owns(name)
                and name not in self._fixed
            ):
                raise CatalogReloadError(
                    f"New fixed dependency {name!r} is already loaded; restart the app",
                    restart_required=True,
                )

    def prepare(self) -> PreparedCatalogReload:
        """Preflight source and return a new opaque, single-use reload plan.

        Supersedes any previous plan without changing imported modules. Raises
        ``CatalogReloadError`` for wrong-thread/reentrant use, invalid source or
        fixed-source changes; its ``restart_required`` flag distinguishes faults
        that cannot be repaired by retrying source reload.
        """
        self._assert_owner()
        self._prepared = None
        try:
            sources = self._inventory()
            self._check_fixed(sources)
            owned = {
                name: source for name, source in sources.items() if self._owns(name)
            }
            if self._catalog_module not in owned:
                raise CatalogReloadError("The experiment catalog source is missing")
            code = {
                name: compile(
                    source.content, str(source.path), "exec", dont_inherit=True
                )
                for name, source in owned.items()
            }
        except (OSError, SyntaxError) as exc:
            raise CatalogReloadError(f"Catalog preflight failed: {exc}") from exc
        token = PreparedCatalogReload()
        self._prepared = _Revision(token, owned, code)
        return token

    def _check_revision(self, revision: _Revision) -> None:
        sources = self._inventory()
        self._check_fixed(sources)
        owned = {name: source for name, source in sources.items() if self._owns(name)}
        if owned != revision.sources:
            raise CatalogReloadError(
                "Experiment source changed after preflight; prepare again"
            )

    def _evict(self) -> None:
        names = sorted(
            (name for name in sys.modules if self._owns(name)), key=len, reverse=True
        )
        for name in names:
            module = sys.modules.pop(name)
            parent_name, _, leaf = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, leaf, None) is module:
                delattr(parent, leaf)

    def _import_catalog(
        self, revision: _Revision, retained: dict[str, ModuleType]
    ) -> Registry:
        """Import and validate one revision while the bounded finder is installed."""
        finder = _SourceFinder(
            self._namespaces, self._owns, revision, frozenset(self._fixed), retained
        )
        sys.meta_path.insert(0, finder)
        try:
            catalog = importlib.import_module(self._catalog_module)
            register = getattr(catalog, "register_all", None)
            if not callable(register):
                raise TypeError("Catalog must export register_all(registry)")
            candidate = Registry()
            register(candidate)
            candidate.validate()
            self._check_revision(revision)
            if any(
                sys.modules.get(name) is not module for name, module in retained.items()
            ):
                raise CatalogReloadError(
                    "Import replaced a fixed module; restart the app",
                    restart_required=True,
                )
            return candidate
        finally:
            try:
                sys.meta_path.remove(finder)
            except ValueError as exc:
                raise CatalogReloadError(
                    "Import changed the loader chain; restart the app",
                    restart_required=True,
                ) from exc

    def load(self, plan: PreparedCatalogReload) -> Registry:
        """Consume this loader's latest plan and return a validated new Registry.

        The caller must first close experiment tabs and exclude operations.
        Wrong-thread/reentrant use, stale/foreign/consumed plans and import or
        integrity failures raise ``CatalogReloadError``. Failed imports discard
        partial reload modules, not arbitrary side effects. A restart-required
        error forbids retry without restarting the app.
        """
        self._assert_owner()
        revision = self._prepared
        if revision is None or revision.token is not plan:
            raise CatalogReloadError("Unknown, superseded or consumed reload plan")
        self._prepared = None
        # Reject stale plans before evicting anything; disk can change after prepare.
        try:
            self._check_revision(revision)
        except OSError as exc:
            raise CatalogReloadError(f"Cannot read experiment source: {exc}") from exc
        # Import failure may leave None entries; only actual modules have identity.
        modules: dict[str, object] = dict(sys.modules)
        retained = {
            name: module
            for name, module in modules.items()
            if isinstance(module, ModuleType)
            and _within_any(name, self._namespaces)
            and not self._owns(name)
        }
        self._loading = True
        try:
            # Deferred imports outlive the finder. Remove old owned bytecode too.
            for source in revision.sources.values():
                Path(importlib.util.cache_from_source(str(source.path))).unlink(
                    missing_ok=True
                )
            self._evict()
            importlib.invalidate_caches()
            return self._import_catalog(revision, retained)
        except BaseException as exc:
            # No rollback claim: discard owned modules and retain the RAM snapshot.
            try:
                self._evict()
            except Exception as cleanup_error:
                raise CatalogReloadError(
                    f"Partial module cleanup failed: {cleanup_error}; restart the app",
                    restart_required=True,
                ) from cleanup_error
            if any(
                sys.modules.get(name) is not module for name, module in retained.items()
            ):
                raise CatalogReloadError(
                    "Import altered fixed modules before failing; restart the app",
                    restart_required=True,
                ) from exc
            try:
                self._check_fixed(self._inventory())
            except (CatalogReloadError, OSError) as integrity_error:
                raise CatalogReloadError(
                    f"Cannot confirm fixed-source integrity: {integrity_error}",
                    restart_required=True,
                ) from exc
            if isinstance(exc, CatalogReloadError):
                raise
            if isinstance(exc, Exception):
                raise CatalogReloadError(
                    f"Experiment catalog load failed: {exc}"
                ) from exc
            raise
        finally:
            self._loading = False
