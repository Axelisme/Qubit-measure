"""Component identifiers and the public schema-to-view binding contract."""

from pathlib import Path

import pytest
from ruamel.yaml import YAML
from zcu_tools.resources.document_store import DocumentStore
from zcu_tools.resources.entry import ResultEntry, snapshot, source_path
from zcu_tools.resources.entry.points import PointView
from zcu_tools.resources.entry.schema import PointDocument, SetupDocument
from zcu_tools.resources.entry.views import EditView, SetupView


@pytest.mark.parametrize(
    "name", ["wiring", "ext", "kind", "components", "entry_id", "created_at"]
)
@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("operation", ["add", "load"])
def test_component_field_names_support_public_reads_and_writes(
    entry: ResultEntry, name: str, owner: str, operation: str
) -> None:
    view = entry.setup if owner == "setup" else entry.new_point("a")
    if operation == "add":
        failure = None
        try:
            view.add_component(name, kind="fake/sensor", rate=10.0)
        except ValueError as error:
            failure = error
        assert failure is None, f"legal component {name!r} was rejected: {failure}"
    else:
        # Publish a document through the public schema/store seam, then bind it.
        document = snapshot(view).model_dump()
        document["components"][name] = {"kind": "fake/sensor", "rate": 10.0}
        schema = SetupDocument if owner == "setup" else PointDocument
        try:
            candidate = schema.model_validate(document)
        except ValueError as error:
            raise AssertionError(f"schema rejected legal component {name!r}") from error
        store = DocumentStore(
            source_path(view), schema, format="zcu.parameter-container"
        )
        with store.edit() as draft:
            draft.components = candidate.components
        view.refresh()

    component = getattr(view, name)
    assert component.rate == 10.0
    component.rate = 11.0
    with view.edit() as draft:
        draft.set(f"{name}.rate", 12.0)
        assert getattr(draft, name).rate == 12.0
        assert component.rate == 11.0
    assert component.rate == 12.0
    source = view.meta(f"{name}.rate")
    assert source is not None and source.source == "manual"
    detached = snapshot(view)
    detached.components.clear()
    assert component.rate == 12.0
    view.refresh()
    assert getattr(view, name).rate == 12.0
    before = source_path(view).read_bytes()
    cached = snapshot(view)
    for invalid in ("class", "_hidden", "not-valid"):
        with pytest.raises(ValueError, match="public identifier"):
            view.add_component(invalid, kind="fake/sensor")
        assert snapshot(view) == cached
        assert source_path(view).read_bytes() == before


@pytest.mark.parametrize("owner", ["setup", "point", "edit"])
def test_schema_accepts_identifiers_but_view_binding_rejects_shadowing(
    entry: ResultEntry, entry_roots: tuple[Path, Path], owner: str
) -> None:
    view = entry.setup if owner == "setup" else entry.new_point("a")
    document = snapshot(view).model_dump()
    document["components"]["edit"] = {"kind": "fake/sensor"}
    schema = SetupDocument if owner == "setup" else PointDocument
    try:
        candidate = schema.model_validate(document)
    except ValueError as error:
        raise AssertionError("schema must not enforce view membership") from error
    source = source_path(view)
    before = source.read_bytes()
    ledger = entry.result_path / "records/ledger.jsonl"
    if owner == "edit":
        with pytest.raises(ValueError, match="edit.*shadow.*view"):
            EditView(candidate, ledger=ledger, entry_id=entry.entry_id)
    else:
        store = DocumentStore(source, schema, format="zcu.parameter-container")
        with store.edit() as draft:
            draft.components = candidate.components
        before = source.read_bytes()
        if owner == "setup":
            setup_store = DocumentStore(
                source, SetupDocument, format="zcu.parameter-container"
            )
            with pytest.raises(ValueError, match="edit.*shadow.*view"):
                SetupView(setup_store, source, ledger=ledger, entry_id=entry.entry_id)
        else:
            point_store = DocumentStore(
                source, PointDocument, format="zcu.parameter-container"
            )
            with pytest.raises(ValueError, match="edit.*shadow.*view"):
                PointView(
                    point_store,
                    source,
                    ledger=ledger,
                    entry_id=entry.entry_id,
                    result_root=entry_roots[0],
                    database_root=entry_roots[1],
                )
    assert source.read_bytes() == before
    assert "edit" not in snapshot(view).components


@pytest.mark.parametrize(
    "name",
    [
        "general",
        "description",
        "edit",
        "meta",
        "refresh",
        "add_component",
        "resolve",
        "set",
    ],
)
@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("operation", ["add", "bind"])
def test_public_view_member_conflicts_are_named_without_publication(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    name: str,
    owner: str,
    operation: str,
) -> None:
    view = entry.setup if owner == "setup" else entry.new_point("a")
    view.add_component("R1", kind="fake/sensor", rate=10.0)
    path = source_path(view)
    cached = snapshot(view)
    if operation == "bind":
        document = YAML(typ="safe").load(path)
        document["components"][name] = {"kind": "fake/sensor"}
        with path.open("w", encoding="utf-8") as stream:
            YAML(typ="rt").dump(document, stream)
    before = path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            view.add_component(name, kind="fake/sensor")
        elif owner == "setup":
            ResultEntry.open(
                "entry", result_root=entry_roots[0], database_root=entry_roots[1]
            )
        else:
            entry.use_point("a")

    with pytest.raises(ValueError, match="shadow.*view") as failure:
        perform_operation()
    assert name in str(failure.value)
    assert "shadow" in str(failure.value) and "view" in str(failure.value)
    assert path.read_bytes() == before
    assert snapshot(view) == cached
