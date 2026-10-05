"""Public loader contract using isolated source packages in the current process."""

from __future__ import annotations

import importlib
import os
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import zcu_tools
from zcu_tools.gui.app.measure.adapter import ExpAdapterProtocol
from zcu_tools.gui.app.measure.catalog import CatalogReloadError
from zcu_tools.gui.app.measure.catalog_loader import (
    SourceExperimentCatalogLoader,
    SourcePackage,
)
from zcu_tools.gui.app.measure.registry import Registry


@dataclass
class CatalogFixture:
    root: Path
    loader: SourceExperimentCatalogLoader
    original: ModuleType

    def write(self, relative: str, content: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")

    def reload(self) -> Registry:
        return self.loader.load(self.loader.prepare())


@pytest.fixture
def source_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[CatalogFixture]:
    root = tmp_path / "reload_fixture"
    root.mkdir()
    files = {
        "__init__.py": "",
        "core.py": """
            from zcu_tools.gui.app.measure.adapter import ExpAdapterProtocol, AdapterCapabilities
            class Base(ExpAdapterProtocol):
                capabilities = AdapterCapabilities()
        """,
        "domain/__init__.py": """
            from .experiment import Experiment
            from . import fixed
        """,
        "domain/fixed.py": "TOKEN = object()\n",
        "domain/experiment.py": """
            from . import fixed
            class Experiment:
                def label(self):
                    return "v1"
                def token(self):
                    return fixed.TOKEN
        """,
        "adapters/__init__.py": "from .example import Adapter\n",
        "adapters/example.py": """
            from reload_fixture.core import Base
            from reload_fixture.domain import Experiment
            from zcu_tools.gui.app.measure.adapter import AdapterGuide
            class Adapter(Base):
                @classmethod
                def guide(cls):
                    return AdapterGuide(Experiment().label(), "", "", "", "")
        """,
        "catalog.py": """
            from .adapters import Adapter
            def register_all(registry):
                registry.register("demo", Adapter)
        """,
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    original = importlib.import_module("reload_fixture.domain")
    importlib.import_module("reload_fixture.catalog")
    loader = SourceExperimentCatalogLoader(
        sources=(SourcePackage("reload_fixture", root),),
        reload_modules=(
            "reload_fixture.domain",
            "reload_fixture.adapters",
            "reload_fixture.catalog",
        ),
        preserved_modules=("reload_fixture.domain.fixed",),
        catalog_module="reload_fixture.catalog",
    )
    try:
        yield CatalogFixture(root, loader, original)
    finally:
        for name in tuple(sys.modules):
            if name == "reload_fixture" or name.startswith("reload_fixture."):
                del sys.modules[name]
        importlib.invalidate_caches()


def test_modified_experiment_reexports_and_fixed_child_identity(
    source_catalog: CatalogFixture,
) -> None:
    fixed = source_catalog.original.fixed
    core = importlib.import_module("reload_fixture.core")
    source_catalog.write(
        "domain/experiment.py",
        """
        from . import fixed
        class Experiment:
            def label(self):
                return "v2"
            def token(self):
                return fixed.TOKEN
    """,
    )
    registry = source_catalog.reload()
    assert registry.create("demo").guide().behavior == "v2"
    domain = importlib.import_module("reload_fixture.domain")
    assert domain is not source_catalog.original
    assert domain.fixed is fixed
    assert domain.Experiment().token() is fixed.TOKEN
    assert isinstance(registry.create("demo"), core.Base)


def test_same_timestamp_same_size_edits_bypass_old_bytecode(
    source_catalog: CatalogFixture,
) -> None:
    path = source_catalog.root / "domain/experiment.py"
    stat = path.stat()
    path.write_text(path.read_text().replace('"v1"', '"v2"'), encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert source_catalog.reload().create("demo").guide().behavior == "v2"


def test_deferred_import_cannot_use_previous_bytecode(
    source_catalog: CatalogFixture,
) -> None:
    source_catalog.write("domain/__init__.py", "")
    source_catalog.write(
        "adapters/example.py",
        """
        from reload_fixture.core import Base
        from zcu_tools.gui.app.measure.adapter import AdapterGuide
        class Adapter(Base):
            @classmethod
            def guide(cls):
                from reload_fixture.domain.experiment import Experiment
                return AdapterGuide(Experiment().label(), "", "", "", "")
    """,
    )
    path = source_catalog.root / "domain/experiment.py"
    stat = path.stat()
    path.write_text(path.read_text().replace('"v1"', '"v2"'), encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert source_catalog.reload().create("demo").guide().behavior == "v2"


def test_fixed_module_corruption_during_failed_import_requires_restart(
    source_catalog: CatalogFixture,
) -> None:
    source_catalog.write(
        "catalog.py",
        """
        import sys
        del sys.modules["reload_fixture.core"]
        raise ValueError("failed after altering shared state")
    """,
    )
    with pytest.raises(CatalogReloadError) as failure:
        source_catalog.reload()
    assert failure.value.restart_required


def test_new_adapter_and_shared_helper(source_catalog: CatalogFixture) -> None:
    source_catalog.write("domain/seed.py", 'LABEL = "new"\n')
    source_catalog.write(
        "adapters/new.py",
        """
        from .example import Adapter
        from reload_fixture.domain.seed import LABEL
        from zcu_tools.gui.app.measure.adapter import AdapterGuide
        class NewAdapter(Adapter):
            @classmethod
            def guide(cls):
                return AdapterGuide(LABEL, "", "", "", "")
    """,
    )
    source_catalog.write(
        "catalog.py",
        """
        from .adapters import Adapter
        from .adapters.new import NewAdapter
        def register_all(registry):
            registry.register("demo", Adapter)
            registry.register("new", NewAdapter)
    """,
    )
    registry = source_catalog.reload()
    assert registry.list_names() == ["demo", "new"]
    assert registry.create("new").guide().behavior == "new"
    source_catalog.write("domain/seed.py", 'LABEL = "changed"\n')
    assert source_catalog.reload().create("new").guide().behavior == "changed"


def test_removed_export_cannot_silently_use_stale_class(
    source_catalog: CatalogFixture,
) -> None:
    source_catalog.write("domain/experiment.py", "RENAMED = True\n")
    with pytest.raises(CatalogReloadError, match="Experiment"):
        source_catalog.reload()
    source_catalog.write(
        "domain/experiment.py",
        """
        class Experiment:
            def label(self):
                return "fixed"
    """,
    )
    assert source_catalog.reload().create("demo").guide().behavior == "fixed"


def test_removed_source_cannot_fall_back_to_cached_module(
    source_catalog: CatalogFixture,
) -> None:
    (source_catalog.root / "domain/experiment.py").unlink()
    with pytest.raises(CatalogReloadError, match="experiment"):
        source_catalog.reload()


def test_failed_import_is_retryable_and_finder_is_retired(
    source_catalog: CatalogFixture,
) -> None:
    source_catalog.write("catalog.py", 'raise ValueError("bad catalog")\n')
    finders = list(sys.meta_path)
    with pytest.raises(CatalogReloadError, match="bad catalog") as failure:
        source_catalog.reload()
    assert not failure.value.restart_required
    assert sys.meta_path == finders
    source_catalog.write(
        "catalog.py",
        """
        from .adapters import Adapter
        def register_all(registry):
            registry.register("repaired", Adapter)
    """,
    )
    assert source_catalog.reload().list_names() == ["repaired"]


def test_invalid_adapter_fails_before_publication(
    source_catalog: CatalogFixture,
) -> None:
    source_catalog.write(
        "catalog.py",
        """
        def register_all(registry):
            registry.register("invalid", object)
    """,
    )
    with pytest.raises(CatalogReloadError, match="ExpAdapterProtocol"):
        source_catalog.reload()


def test_syntax_preflight_leaves_live_modules_untouched(
    source_catalog: CatalogFixture,
) -> None:
    source_catalog.write("adapters/example.py", "def broken(\n")
    with pytest.raises(CatalogReloadError, match="preflight"):
        source_catalog.loader.prepare()
    assert importlib.import_module("reload_fixture.domain") is source_catalog.original


@pytest.mark.parametrize("remove", [False, True])
def test_modified_or_removed_fixed_source_requires_restart(
    source_catalog: CatalogFixture, remove: bool
) -> None:
    path = source_catalog.root / "domain/fixed.py"
    if remove:
        path.unlink()
    else:
        path.write_text("TOKEN = 'changed'\n", encoding="utf-8")
    with pytest.raises(CatalogReloadError, match="Fixed source") as failure:
        source_catalog.loader.prepare()
    assert failure.value.restart_required
    assert importlib.import_module("reload_fixture.domain") is source_catalog.original


def test_unrelated_new_fixed_file_is_allowed_until_imported(
    source_catalog: CatalogFixture,
) -> None:
    source_catalog.write("new_fixed.py", "VALUE = 1\n")
    assert source_catalog.reload().list_names() == ["demo"]
    source_catalog.write(
        "catalog.py",
        """
        from . import new_fixed
        def register_all(registry):
            pass
    """,
    )
    with pytest.raises(CatalogReloadError, match="New fixed dependency") as failure:
        source_catalog.reload()
    assert failure.value.restart_required


def test_stale_and_consumed_plans_are_rejected(source_catalog: CatalogFixture) -> None:
    token = source_catalog.loader.prepare()
    source_catalog.write("domain/seed.py", "VALUE = 1\n")
    with pytest.raises(CatalogReloadError, match="changed after preflight"):
        source_catalog.loader.load(token)
    assert importlib.import_module("reload_fixture.domain") is source_catalog.original
    with pytest.raises(CatalogReloadError, match="consumed"):
        source_catalog.loader.load(token)
    token = source_catalog.loader.prepare()
    source_catalog.loader.load(token)
    with pytest.raises(CatalogReloadError, match="consumed"):
        source_catalog.loader.load(token)


def test_superseded_plan_does_not_consume_current_plan(
    source_catalog: CatalogFixture,
) -> None:
    old = source_catalog.loader.prepare()
    current = source_catalog.loader.prepare()
    with pytest.raises(CatalogReloadError, match="superseded"):
        source_catalog.loader.load(old)
    assert source_catalog.loader.load(current).list_names() == ["demo"]


def test_source_mutation_during_load_is_not_reported_as_success(
    source_catalog: CatalogFixture,
) -> None:
    source_catalog.write(
        "catalog.py",
        """
        from pathlib import Path
        from .adapters import Adapter
        def register_all(registry):
            registry.register("demo", Adapter)
            path = Path(__file__).parent / "domain" / "experiment.py"
            path.write_text("CHANGED = True\\n")
    """,
    )
    with pytest.raises(CatalogReloadError, match="changed after preflight"):
        source_catalog.reload()


@dataclass
class DualCatalogFixture:
    """User catalog plus the fixed framework it consumes.

    user owns the reloadable package and loader; framework is its fixed source
    directory; fixed_module is the original imported framework core identity.
    """

    user: CatalogFixture
    framework: Path
    fixed_module: ModuleType


@pytest.fixture
def dual_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[DualCatalogFixture]:
    framework = tmp_path / "loader_framework"
    framework.mkdir()
    (framework / "__init__.py").write_text("", encoding="utf-8")
    (framework / "core.py").write_text(
        textwrap.dedent("""
            from zcu_tools.gui.app.measure.adapter import ExpAdapterProtocol, AdapterCapabilities
            class Base(ExpAdapterProtocol):
                capabilities = AdapterCapabilities()
        """),
        encoding="utf-8",
    )
    root = tmp_path / "loader_user"
    root.mkdir()
    files = {
        "__init__.py": "",
        "domain/__init__.py": "from . import fixed\nfrom .experiment import Experiment\n",
        "domain/fixed.py": "TOKEN = object()\n",
        "domain/experiment.py": """
            from . import fixed
            class Experiment:
                def label(self):
                    return "v1"
                def token(self):
                    return fixed.TOKEN
        """,
        "catalog.py": """
            from loader_framework.core import Base
            from loader_user.domain import Experiment
            from zcu_tools.gui.app.measure.adapter import AdapterGuide
            class Adapter(Base):
                @classmethod
                def guide(cls):
                    return AdapterGuide(Experiment().label(), "", "", "", "")
            def register_all(registry):
                registry.register("demo", Adapter)
        """,
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        original = importlib.import_module("loader_user.domain")
        fixed_module = importlib.import_module("loader_framework.core")
        importlib.import_module("loader_user.catalog")
        loader = SourceExperimentCatalogLoader(
            sources=(
                SourcePackage("loader_framework", framework),
                SourcePackage("loader_user", root),
            ),
            reload_modules=("loader_user.domain", "loader_user.catalog"),
            preserved_modules=("loader_user.domain.fixed",),
            catalog_module="loader_user.catalog",
        )
        yield DualCatalogFixture(
            CatalogFixture(root, loader, original), framework, fixed_module
        )
    finally:
        for name in tuple(sys.modules):
            if name in ("loader_framework", "loader_user") or name.startswith(
                ("loader_framework.", "loader_user.")
            ):
                del sys.modules[name]
        importlib.invalidate_caches()


def test_user_catalog_reloads_without_replacing_framework_identity(
    dual_catalog: DualCatalogFixture,
) -> None:
    user = dual_catalog.user
    fixed = user.original.fixed
    user.write(
        "domain/experiment.py",
        """
        from . import fixed
        class Experiment:
            def label(self):
                return "v2"
            def token(self):
                return fixed.TOKEN
        """,
    )
    registry = user.reload()
    assert registry.create("demo").guide().behavior == "v2"
    assert isinstance(registry.create("demo"), dual_catalog.fixed_module.Base)
    assert importlib.import_module("loader_framework.core") is dual_catalog.fixed_module
    domain = importlib.import_module("loader_user.domain")
    assert domain is not user.original
    assert domain.fixed is fixed
    assert domain.Experiment().token() is fixed.TOKEN


@pytest.fixture
def leaf_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[CatalogFixture]:
    """Use a core/gui leaf, fixed shared types and startup-only roles."""
    root = tmp_path / "loader_leaf"
    files = {
        "__init__.py": "",
        "v2/__init__.py": "from .family.demo.core import Experiment\n",
        "v2/family/__init__.py": "",
        "v2/family/demo/__init__.py": "",
        "v2/_support/__init__.py": "",
        "v2/_support/shared.py": "class Result:\n    pass\n",
        "roles.py": "from .v2._support.shared import Result\nROLE_RESULT = Result()\n",
        "v2/family/demo/core.py": """
            from loader_leaf.v2._support.shared import Result
            class Experiment:
                def label(self):
                    return "v1"
        """,
        "v2/family/demo/gui.py": """
            from zcu_tools.gui.app.measure.adapter import (
                ExpAdapterProtocol, AdapterCapabilities, AdapterGuide,
            )
            from .core import Experiment
            class Adapter(ExpAdapterProtocol):
                capabilities = AdapterCapabilities()
                @classmethod
                def guide(cls):
                    return AdapterGuide(Experiment().label(), "", "", "", "")
        """,
        "definitions.py": """
            from .v2.family.demo.gui import Adapter
            def register_all(registry):
                registry.register("demo", Adapter)
        """,
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        original = importlib.import_module("loader_leaf.v2.family.demo.core")
        importlib.import_module("loader_leaf.roles")
        importlib.import_module("loader_leaf.definitions")
        loader = SourceExperimentCatalogLoader(
            sources=(
                SourcePackage("zcu_tools", Path(zcu_tools.__file__).parent),
                SourcePackage("loader_leaf", root),
            ),
            reload_modules=("loader_leaf.v2", "loader_leaf.definitions"),
            preserved_modules=("loader_leaf.v2._support", "loader_leaf.roles"),
            catalog_module="loader_leaf.definitions",
        )
        yield CatalogFixture(root, loader, original)
    finally:
        for name in tuple(sys.modules):
            if name == "loader_leaf" or name.startswith("loader_leaf."):
                del sys.modules[name]
        importlib.invalidate_caches()


def test_leaf_catalog_reloads_core_gui_and_exports_without_replacing_fixed_types(
    leaf_catalog: CatalogFixture,
) -> None:
    shared = importlib.import_module("loader_leaf.v2._support.shared")
    roles = importlib.import_module("loader_leaf.roles")
    old_adapter = importlib.import_module("loader_leaf.v2.family.demo.gui").Adapter
    leaf_catalog.write(
        "v2/family/demo/core.py",
        """
        from loader_leaf.v2._support.shared import Result
        class Experiment:
            def label(self):
                return "v2"
        """,
    )
    registry = leaf_catalog.reload()
    adapter = registry.create("demo")
    core = importlib.import_module("loader_leaf.v2.family.demo.core")
    assert adapter.guide().behavior == "v2"
    assert isinstance(adapter, ExpAdapterProtocol)
    assert type(adapter) is not old_adapter
    assert core.Experiment is not leaf_catalog.original.Experiment
    assert importlib.import_module("loader_leaf.v2").Experiment is core.Experiment
    assert core.Result is shared.Result
    assert importlib.import_module("loader_leaf.roles") is roles
    assert isinstance(roles.ROLE_RESULT, core.Result)


@pytest.mark.parametrize("relative", ["v2/_support/shared.py", "roles.py"])
def test_leaf_fixed_shared_or_startup_role_change_requires_restart(
    leaf_catalog: CatalogFixture, relative: str
) -> None:
    leaf_catalog.write(relative, "CHANGED = True\n")
    with pytest.raises(CatalogReloadError, match="Fixed source") as failure:
        leaf_catalog.loader.prepare()
    assert failure.value.restart_required
    assert (
        importlib.import_module("loader_leaf.v2.family.demo.core")
        is leaf_catalog.original
    )


@pytest.mark.parametrize("package", ["framework", "user"])
@pytest.mark.parametrize("remove", [False, True])
def test_fixed_source_in_either_package_requires_restart(
    dual_catalog: DualCatalogFixture, package: str, remove: bool
) -> None:
    path = (
        dual_catalog.framework / "core.py"
        if package == "framework"
        else dual_catalog.user.root / "domain" / "fixed.py"
    )
    if remove:
        path.unlink()
    else:
        path.write_text("CHANGED = True\n", encoding="utf-8")
    with pytest.raises(CatalogReloadError, match="Fixed source") as failure:
        dual_catalog.user.loader.prepare()
    assert failure.value.restart_required
    assert importlib.import_module("loader_framework.core") is dual_catalog.fixed_module
    assert importlib.import_module("loader_user.domain") is dual_catalog.user.original


@pytest.mark.parametrize("package", ["framework", "user"])
@pytest.mark.parametrize("already_imported", [False, True])
def test_new_fixed_dependency_in_either_package_requires_restart(
    dual_catalog: DualCatalogFixture, package: str, already_imported: bool
) -> None:
    root = dual_catalog.framework if package == "framework" else dual_catalog.user.root
    namespace = "loader_framework" if package == "framework" else "loader_user"
    (root / "new_fixed.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert dual_catalog.user.reload().list_names() == ["demo"]
    if already_imported:
        importlib.import_module(namespace + ".new_fixed")
    else:
        dual_catalog.user.write(
            "catalog.py",
            f"""
            from {namespace} import new_fixed
            def register_all(registry):
                pass
            """,
        )
    with pytest.raises(CatalogReloadError, match="New fixed dependency") as failure:
        dual_catalog.user.reload()
    assert failure.value.restart_required


@pytest.mark.parametrize(
    "namespace", ["loader_framework.core", "loader_user.domain.fixed"]
)
def test_fixed_module_corruption_in_either_package_requires_restart(
    dual_catalog: DualCatalogFixture, namespace: str
) -> None:
    dual_catalog.user.write(
        "catalog.py",
        f"""
        import sys
        sys.modules[{namespace!r}] = None
        def register_all(registry):
            pass
        """,
    )
    with pytest.raises(CatalogReloadError, match="fixed module") as failure:
        dual_catalog.user.reload()
    assert failure.value.restart_required


@pytest.mark.parametrize(
    ("namespaces", "reload_modules", "preserved_modules", "catalog_module", "message"),
    [
        ((), ("demo.catalog",), (), "demo.catalog", "Source packages"),
        (("demo",), (), (), "demo.catalog", "nonempty reload"),
        (("demo", "demo"), ("demo.catalog",), (), "demo.catalog", "overlap"),
        (("demo", "demo.child"), ("demo.catalog",), (), "demo.catalog", "overlap"),
        (("demo.child", "demo"), ("demo.catalog",), (), "demo.catalog", "overlap"),
        (("",), ("demo.catalog",), (), "demo.catalog", "dotted"),
        (("demo..child",), ("demo.catalog",), (), "demo.catalog", "dotted"),
        (("class",), ("class.catalog",), (), "class.catalog", "dotted"),
        (("demo",), ("foreign.catalog",), (), "demo.catalog", "declared source"),
        (("demo",), ("demo_extra.catalog",), (), "demo.catalog", "declared source"),
        (("demo",), ("demo..catalog",), (), "demo.catalog", "declared source"),
        (
            ("demo",),
            ("demo.catalog",),
            ("foreign.fixed",),
            "demo.catalog",
            "declared source",
        ),
        (("demo",), ("demo.domain",), (), "demo.catalog", "reload scope"),
        (("demo",), ("demo",), ("demo.catalog",), "demo.catalog", "reload scope"),
        (("demo",), ("demo",), (), "foreign.catalog", "declared source"),
    ],
)
def test_invalid_source_and_scope_declarations_fail_at_construction(
    tmp_path: Path,
    namespaces: tuple[str, ...],
    reload_modules: tuple[str, ...],
    preserved_modules: tuple[str, ...],
    catalog_module: str,
    message: str,
) -> None:
    sources = []
    for index, namespace in enumerate(namespaces):
        root = tmp_path / str(index)
        root.mkdir()
        sources.append(SourcePackage(namespace, root))
    with pytest.raises(ValueError, match=message):
        SourceExperimentCatalogLoader(
            sources=tuple(sources),
            reload_modules=reload_modules,
            preserved_modules=preserved_modules,
            catalog_module=catalog_module,
        )


def test_missing_source_root_fails_at_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing directory"):
        SourceExperimentCatalogLoader(
            sources=(SourcePackage("demo", tmp_path / "missing"),),
            reload_modules=("demo.catalog",),
            preserved_modules=(),
            catalog_module="demo.catalog",
        )
