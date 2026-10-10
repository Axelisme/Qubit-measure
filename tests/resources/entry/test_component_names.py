"""Component identifiers and the public schema-to-view binding contract."""

from pathlib import Path

import pytest
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


@pytest.mark.parametrize("member_owner", [SetupView, PointView, EditView])
@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("operation", ["add", "bind"])
def test_new_public_view_member_is_rejected_across_view_types_without_publication(
    entry: ResultEntry,
    monkeypatch: pytest.MonkeyPatch,
    member_owner: type[SetupView] | type[PointView] | type[EditView],
    owner: str,
    operation: str,
) -> None:
    view = entry.setup if owner == "setup" else entry.new_point("a")
    view.add_component("R1", kind="fake/sensor", rate=10.0)
    if operation == "bind":
        view.add_component("calibrate", kind="fake/sensor", rate=20.0)
    monkeypatch.setattr(
        member_owner, "calibrate", property(lambda self: "calibration"), raising=False
    )
    before = source_path(view).read_bytes()
    cached = snapshot(view)
    failure = None
    try:
        if operation == "add":
            view.add_component("calibrate", kind="fake/sensor", rate=20.0)
        elif owner == "setup":
            _ = entry.setup
        else:
            entry.use_point("a")
    except ValueError as error:
        failure = error
    assert failure is not None, "new public view member must reserve its component name"
    assert (
        "calibrate" in str(failure)
        and "shadow" in str(failure)
        and "view" in str(failure)
    )
    assert source_path(view).read_bytes() == before
    assert snapshot(view) == cached
