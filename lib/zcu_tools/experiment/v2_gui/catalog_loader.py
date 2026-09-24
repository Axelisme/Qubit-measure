"""Fresh source imports for a bounded experiment namespace.

Only trusted, import-side-effect-free source is supported. This is not a Python
sandbox or a transaction over arbitrary module globals. The app must close its
experiment tabs and exclude operations before calling ``load``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import get_ident
from types import CodeType, ModuleType
from typing import cast

from zcu_tools.gui.app.main.catalog import CatalogReloadError, PreparedCatalogReload
from zcu_tools.gui.app.main.registry import Registry


def _within(name: str, root: str) -> bool:
    return name == root or name.startswith(root + ".")


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

    def create_module(self, spec):
        return None

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
        namespace: str,
        owns: Callable[[str], bool],
        revision: _Revision,
        fixed_names: frozenset[str],
        retained: dict[str, ModuleType],
    ) -> None:
        self._namespace = namespace
        self._owns = owns
        self._revision = revision
        self._fixed_names = fixed_names
        self._retained = retained

    def find_spec(self, fullname: str, path=None, target=None):
        if not _within(fullname, self._namespace):
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

    ``package_root`` contains ``namespace``'s source, not its parent directory.
    All source outside ``reload_modules`` (and all ``preserved_modules``) is
    fixed at construction. A new unused fixed file is harmless; importing a new
    fixed module requires restart. Each plan belongs to this loader and is
    single-use. Construct and invoke on the app owner thread.
    """

    def __init__(
        self,
        *,
        package_root: Path,
        namespace: str,
        reload_modules: Sequence[str],
        preserved_modules: Sequence[str],
        catalog_module: str,
    ) -> None:
        self._root = package_root.resolve()
        self._namespace = namespace
        self._reload_modules = tuple(reload_modules)
        self._preserved_modules = tuple(preserved_modules)
        self._catalog_module = catalog_module
        if not self._root.is_dir() or not self._reload_modules:
            raise ValueError("A source package and nonempty reload scope are required")
        if any(
            not _within(name, namespace)
            for name in (*reload_modules, *preserved_modules)
        ):
            raise ValueError(
                "Reload and preserved scopes must belong to the source namespace"
            )
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
        return any(_within(name, root) for root in self._reload_modules) and not any(
            _within(name, root) for root in self._preserved_modules
        )

    def _assert_owner(self) -> None:
        if get_ident() != self._owner:
            raise CatalogReloadError(
                "Reload must run on its owner thread", restart_required=True
            )
        if self._loading:
            raise CatalogReloadError("A catalog load is already in progress")

    def _inventory(self) -> dict[str, _Source]:
        sources: dict[str, _Source] = {}
        for path in self._root.rglob("*.py"):
            parts = path.relative_to(self._root).with_suffix("").parts
            package = parts[-1] == "__init__"
            if package:
                parts = parts[:-1]
            name = ".".join((self._namespace, *parts))
            if name in sources:
                raise CatalogReloadError(f"Ambiguous source module {name!r}")
            sources[name] = _Source(path, path.read_bytes(), package)
        return sources

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
                _within(name, self._namespace)
                and not self._owns(name)
                and name not in self._fixed
            ):
                raise CatalogReloadError(
                    f"New fixed dependency {name!r} is already loaded; restart the app",
                    restart_required=True,
                )

    def prepare(self) -> PreparedCatalogReload:
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

    def load(self, plan: PreparedCatalogReload) -> Registry:
        self._assert_owner()
        revision = self._prepared
        if revision is None or revision.token is not plan:
            raise CatalogReloadError("Unknown, superseded or consumed reload plan")
        self._prepared = None
        # A stale token must not evict anything. The app also prepares before
        # closing tabs, but disk may change at any point in a source workflow.
        try:
            self._check_revision(revision)
        except OSError as exc:
            raise CatalogReloadError(f"Cannot read experiment source: {exc}") from exc
        retained = {
            name: module
            for name, module in sys.modules.copy().items()
            if isinstance(module, ModuleType)
            and _within(name, self._namespace)
            and not self._owns(name)
        }
        finder = _SourceFinder(
            self._namespace, self._owns, revision, frozenset(self._fixed), retained
        )
        self._loading = True
        try:
            # A function-local import may happen after our finder is retired.
            # Remove this interpreter's old bytecode for every owned source,
            # including modules not eagerly imported by the catalog.
            for source in revision.sources.values():
                Path(importlib.util.cache_from_source(str(source.path))).unlink(
                    missing_ok=True
                )
            self._evict()
            importlib.invalidate_caches()
            sys.meta_path.insert(0, finder)
            try:
                catalog = importlib.import_module(self._catalog_module)
                register = getattr(catalog, "register_all", None)
                if not callable(register):
                    raise TypeError("Catalog must export register_all(registry)")
                candidate = Registry()
                cast(Callable[[Registry], None], register)(candidate)
                candidate.validate()
                self._check_revision(revision)
                if any(
                    sys.modules.get(name) is not module
                    for name, module in retained.items()
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
        except BaseException as exc:
            # No rollback claim: discard partial owned modules. The orchestrator
            # retains its RAM snapshot and keeps the live registry unavailable.
            try:
                self._evict()
            except Exception as cleanup_error:
                raise CatalogReloadError(
                    f"Partial module cleanup failed: {cleanup_error}; restart the app",
                    restart_required=True,
                ) from exc
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


def make_catalog_loader() -> SourceExperimentCatalogLoader:
    """Composition-root factory, constructed after initial catalog imports."""
    return SourceExperimentCatalogLoader(
        package_root=Path(__file__).resolve().parents[2],
        namespace="zcu_tools",
        reload_modules=(
            "zcu_tools.experiment.v2",
            "zcu_tools.experiment.v2_gui.adapters",
            "zcu_tools.experiment.v2_gui.registry",
        ),
        preserved_modules=(
            "zcu_tools.experiment.v2.runner",
            "zcu_tools.experiment.v2.utils",
            "zcu_tools.experiment.v2_gui.adapters.base",
            "zcu_tools.experiment.v2_gui.adapters._support",
        ),
        catalog_module="zcu_tools.experiment.v2_gui.registry",
    )
