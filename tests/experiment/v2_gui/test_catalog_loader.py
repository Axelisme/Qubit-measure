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
from zcu_tools.experiment.v2_gui.catalog_loader import SourceExperimentCatalogLoader
from zcu_tools.gui.app.main.catalog import CatalogReloadError
from zcu_tools.gui.app.main.registry import Registry


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
            from zcu_tools.gui.app.main.adapter import ExpAdapterProtocol, AdapterCapabilities
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
            from zcu_tools.gui.app.main.adapter import AdapterGuide
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
        package_root=root,
        namespace="reload_fixture",
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
        from zcu_tools.gui.app.main.adapter import AdapterGuide
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
        from zcu_tools.gui.app.main.adapter import AdapterGuide
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
