"""Registry bootstrap and pollution guards at the entry seam."""

from collections.abc import Generator
from pathlib import Path

import pytest
from zcu_tools.resources.entry import ResultEntry

from .fakes import register_fakes, registry_state, restore_registry


@pytest.fixture
def entry_roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "results", tmp_path / "Database"


@pytest.fixture
def entry(entry_roots: tuple[Path, Path]) -> ResultEntry:
    results, database = entry_roots
    return ResultEntry.create("entry", result_root=results, database_root=database)


@pytest.fixture(scope="module", autouse=True)
def entry_registry_models() -> Generator[None]:
    before = registry_state()
    try:
        register_fakes()
        yield
    finally:
        restore_registry(before)


@pytest.fixture(scope="module", autouse=True)
def registry_module_guard(entry_registry_models: None) -> Generator[None]:
    before = registry_state()
    yield
    assert registry_state() == before, "entry tests polluted shared registries"


@pytest.fixture(autouse=True)
def registry_state_guard(
    request: pytest.FixtureRequest, registry_module_guard: None
) -> Generator[None]:
    before = registry_state()
    yield
    assert registry_state() == before, f"registry polluter: {request.node.nodeid}"
