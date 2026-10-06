"""Shared entry builders and model registration custody for view contracts."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from zcu_tools.resources.entry import ComponentSchema, ResultEntry, component_registry
from zcu_tools.resources.entry.views import FieldView


@contextmanager
def registered_model(kind: str, model: type[ComponentSchema]) -> Generator[str]:
    """Register a fake kind/model for this context and unregister it on exit.

    Yield the registered kind. Duplicate or invalid registrations propagate the
    public registry error without entering the context.
    """
    component_registry.register(kind, model)
    try:
        yield kind
    finally:
        component_registry.unregister(kind)


def create_entry(tmp_path: Path) -> tuple[ResultEntry, Path, Path]:
    """Create 'entry' under an unused test directory and return it and both roots.

    ResultEntry creation errors propagate; callers supply their pytest tmp_path.
    """
    results, database = tmp_path / "results", tmp_path / "Database"
    entry = ResultEntry.create("entry", result_root=results, database_root=database)
    return entry, results, database


def field_view(value: object) -> FieldView:
    """Require an editable container from a public component or child read."""
    assert isinstance(value, FieldView)
    return value
