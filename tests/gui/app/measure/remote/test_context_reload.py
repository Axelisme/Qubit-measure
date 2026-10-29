"""Point-file reload and optimistic RPC writes through registered dispatch."""

from hashlib import sha256
from pathlib import Path
from typing import Literal

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec, TemplateDef
from zcu_tools.gui.remote.errors import ErrorCode
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import FileRevision

from tests.gui.app.measure._record_support import require_target

from ._point_application import PointApplication

Document = Literal["parameters", "library", "aliases"]


@pytest.fixture
def application_templates() -> dict[str, TemplateDef]:
    return {
        "synthetic.const": TemplateDef(
            ConstWaveformCfg, "Synthetic constant", {"length": 1.0}
        )
    }


@pytest.fixture
def point_files(application: PointApplication) -> dict[Document, FileRevision]:
    application.open_entry()
    control = application.controller.context_control
    control.add_component("Q1", "fake/drive/a", {"rate": 10.0}, setup=True)
    application.result("point.new", {"label": "base"})
    target = require_target(control.read_context())
    control.set_waveform(target, "Q1.waveforms.seed", ConstWaveformCfg(length=10.0))
    control.set_module(
        target,
        "Q1.pulses.seed",
        PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.1, waveform="Q1.waveforms.seed"),
    )
    return {item.document: item for item in control.get_file_revisions()}


def error_reply(
    application: PointApplication, method: str, params: dict[str, object]
) -> dict[str, object]:
    reply = application.rpc(method, params)
    assert reply["ok"] is False, reply
    error = reply["error"]
    assert isinstance(error, dict), reply
    return error


def external_edit(file: FileRevision) -> bytes:
    if file.document == "aliases":
        candidate = (
            b'format: zcu.aliases\nformat_version: "1.0"\naliases:\n  driver: Q1\n'
        )
    else:
        original = file.path.read_bytes()
        assert b"10.0" in original
        candidate = original.replace(b"10.0", b"20.0")
    file.path.write_bytes(candidate)
    return candidate


@pytest.mark.parametrize("document", ["parameters", "library", "aliases"])
def test_reload_installs_edited_disk_using_installed_base_revision(
    application: PointApplication,
    point_files: dict[Document, FileRevision],
    document: Document,
) -> None:
    file = point_files[document]
    candidate = external_edit(file)
    revision = sha256(candidate).hexdigest()
    reply = application.rpc(
        "context.reload", {"path": str(file.path), "expected_revision": file.revision}
    )
    assert reply["ok"] is True, reply
    assert reply["result"] == {
        "document": document,
        "path": str(file.path),
        "revision": revision,
    }
    control = application.controller.context_control
    installed = {item.document: item for item in control.get_file_revisions()}
    assert installed[document].revision == revision
    assert file.path.read_bytes() == candidate
    if document == "parameters":
        assert control.read_context().lookup_point("Q1.rate").value == 20.0
    elif document == "library":
        assert control.read_context().library.get_waveform(
            "Q1.waveforms.seed"
        ) == ConstWaveformCfg(length=20.0)
    else:
        assert control.get_alias_rows() == (AliasRow("driver", "Q1"),)


@pytest.mark.parametrize("external_exists", [False, True])
def test_library_writer_null_checks_absence_before_first_creation(
    application: PointApplication, external_exists: bool
) -> None:
    info = application.open_entry()
    application.result("point.new", {"label": "portable"})
    path = info.result_path / "points/portable/library.yaml"
    external = path.read_bytes() + b"# external creation\n"
    path.unlink()
    application.result("point.use", {"label": "portable"})
    files = application.result("entry.info", {})["files"]
    assert isinstance(files, list)
    assert files[1] == {"document": "library", "path": str(path), "revision": None}
    assert not path.exists()
    if external_exists:
        path.write_bytes(external)
    params: dict[str, object] = {
        "role_id": "synthetic.const",
        "name": "Q1.waveforms.new",
        "edits": [{"path": ["length"], "value": 0.2}],
        "expected_revision": None,
    }
    if external_exists:
        error = error_reply(application, "context.ml_create_from_role", params)
        assert (error["code"], error.get("reason")) == (
            ErrorCode.PRECONDITION_FAILED,
            "revision_mismatch",
        )
        assert path.read_bytes() == external
        assert application.result("entry.info", {})["files"] == files
    else:
        reply = application.rpc("context.ml_create_from_role", params)
        assert reply["ok"] is True, reply
        assert (
            application.controller.context_control.read_context().library.waveform_keys()
            == ("Q1.waveforms.new",)
        )
        assert path.is_file()


def test_absent_library_reports_null_and_reloads_external_file(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    application.result("point.new", {"label": "portable"})
    path = info.result_path / "points/portable/library.yaml"
    body = path.read_bytes()
    path.unlink()
    application.result("point.use", {"label": "portable"})
    files = application.result("entry.info", {})["files"]
    assert isinstance(files, list)
    assert files[1] == {"document": "library", "path": str(path), "revision": None}
    assert not path.exists()
    path.write_bytes(b"invalid: true\n")
    error = error_reply(
        application, "context.reload", {"path": str(path), "expected_revision": None}
    )
    assert error["code"] == ErrorCode.INVALID_PARAMS
    assert application.result("entry.info", {})["files"] == files
    path.write_bytes(body)
    assert application.result(
        "context.reload", {"path": str(path), "expected_revision": None}
    ) == {
        "document": "library",
        "path": str(path),
        "revision": sha256(body).hexdigest(),
    }


def test_reload_rejects_disk_hash_instead_of_installed_base(
    application: PointApplication, point_files: dict[Document, FileRevision]
) -> None:
    file = point_files["parameters"]
    candidate = external_edit(file)
    error = error_reply(
        application,
        "context.reload",
        {"path": str(file.path), "expected_revision": sha256(candidate).hexdigest()},
    )
    assert (error["code"], error.get("reason")) == (
        ErrorCode.PRECONDITION_FAILED,
        "revision_mismatch",
    )
    assert (
        application.controller.context_control.read_context()
        .lookup_point("Q1.rate")
        .value
        == 10.0
    )
    assert file.path.read_bytes() == candidate


@pytest.mark.parametrize("document", ["parameters", "library", "aliases"])
def test_reload_validation_failure_names_file_and_preserves_installation(
    application: PointApplication,
    point_files: dict[Document, FileRevision],
    document: Document,
) -> None:
    control = application.controller.context_control
    before = control.read_context()
    file = point_files[document]
    candidate = b"not: [valid\n"
    file.path.write_bytes(candidate)
    error = error_reply(
        application,
        "context.reload",
        {"path": str(file.path), "expected_revision": file.revision},
    )
    assert error["code"] == ErrorCode.INVALID_PARAMS, error
    assert str(file.path) in str(error["message"])
    assert len(str(error["message"])) > len(str(file.path))
    assert control.read_context().document() == before.document()
    assert control.read_context().library.document() == before.library.document()
    assert control.get_alias_rows() == ()
    assert control.get_file_revisions() == tuple(point_files.values())
    assert file.path.read_bytes() == candidate


@pytest.mark.parametrize("path_kind", ["relative", "outside", "unselected"])
def test_reload_rejects_paths_outside_selected_point_owners(
    application: PointApplication,
    point_files: dict[Document, FileRevision],
    tmp_path: Path,
    path_kind: str,
) -> None:
    file = point_files["parameters"]
    path = {
        "relative": Path(file.path.name),
        "outside": tmp_path / "outside.yaml",
        "unselected": file.path.parents[1] / "other/parameters.yaml",
    }[path_kind]
    before = file.path.read_bytes()
    error = error_reply(
        application,
        "context.reload",
        {"path": str(path), "expected_revision": file.revision},
    )
    assert error["code"] == ErrorCode.INVALID_PARAMS, error
    assert str(path) in str(error["message"])
    assert file.path.read_bytes() == before


@pytest.mark.parametrize("global_active", [False, True])
def test_reload_entry_while_point_active_keeps_active_files_and_version(
    application: PointApplication, global_active: bool
) -> None:
    info = application.open_entry()
    control = application.controller.context_control
    control.add_component("Q1", "fake/drive/a", {"rate": 10.0}, setup=True)
    if not global_active:
        application.result("point.new", {"label": "base"})
    active_files = application.result("entry.info", {})["files"]
    path = info.result_path / "parameters.yaml"
    file = next(
        item for item in control.get_installed_file_revisions() if item.path == path
    )
    version = application.state.version.get("context")
    candidate = path.read_bytes().replace(b"10.0", b"20.0")
    path.write_bytes(candidate)
    application.result("control.release", {})
    rejected = error_reply(
        application,
        "context.reload",
        {
            "path": str(path),
            "expected_revision": file.revision,
        },
    )
    assert rejected.get("reason") == "not_holder"
    assert application.state.version.get("context") == version
    assert application.result("entry.info", {})["files"] == active_files
    application.result("control.acquire", {})
    assert application.result(
        "context.reload",
        {
            "path": str(path),
            "expected_revision": file.revision,
        },
    ) == {
        "document": "parameters",
        "path": str(path),
        "revision": sha256(candidate).hexdigest(),
    }
    assert control.read_context(setup=True).value("Q1.rate") == 20.0
    if global_active:
        assert application.state.version.get("context") == version + 1
        assert control.read_context().value("Q1.rate") == 20.0
    else:
        assert application.state.version.get("context") == version
        assert application.result("entry.info", {})["files"] == active_files
        assert control.read_context().value("Q1.rate") == 10.0


def test_reload_requires_holder_before_installing_external_edit(
    application: PointApplication, point_files: dict[Document, FileRevision]
) -> None:
    file = point_files["parameters"]
    candidate = external_edit(file)
    application.result("control.release", {})
    error = error_reply(
        application,
        "context.reload",
        {"path": str(file.path), "expected_revision": file.revision},
    )
    assert error.get("reason") == "not_holder", error
    assert (
        application.controller.context_control.read_context()
        .lookup_point("Q1.rate")
        .value
        == 10.0
    )
    assert file.path.read_bytes() == candidate


@pytest.mark.parametrize(
    "expected", [{}, {"expected_revision": 1}, {"expected_revision": None}]
)
def test_reload_requires_explicit_string_base_for_existing_parameter_file(
    application: PointApplication,
    point_files: dict[Document, FileRevision],
    expected: dict[str, object],
) -> None:
    file = point_files["parameters"]
    error = error_reply(
        application, "context.reload", {"path": str(file.path), **expected}
    )
    assert error["code"] == ErrorCode.INVALID_PARAMS, error


WRITERS: tuple[tuple[str, Document, dict[str, object]], ...] = (
    ("context.set_parameter", "parameters", {"path": "Q1.rate", "value": 30.0}),
    ("context.aliases_set", "aliases", {"rows": [{"alias": "driver", "value": "Q1"}]}),
    (
        "context.ml_edit",
        "library",
        {
            "kind": "waveform",
            "name": "Q1.waveforms.seed",
            "edits": [{"path": ["length"], "value": 30.0}],
        },
    ),
    (
        "context.ml_create_from_role",
        "library",
        {
            "role_id": "synthetic.const",
            "name": "Q1.waveforms.new",
            "edits": [{"path": ["length"], "value": 30.0}],
        },
    ),
)


@pytest.mark.parametrize(("method", "document", "params"), WRITERS)
def test_optional_writer_revision_rejects_stale_disk_without_side_effects(
    application: PointApplication,
    point_files: dict[Document, FileRevision],
    method: str,
    document: Document,
    params: dict[str, object],
) -> None:
    control = application.controller.context_control
    file = point_files[document]
    before = control.read_context()
    candidate = external_edit(file)
    error = error_reply(
        application, method, {**params, "expected_revision": file.revision}
    )
    assert (error["code"], error.get("reason")) == (
        ErrorCode.PRECONDITION_FAILED,
        "revision_mismatch",
    ), error
    assert file.path.read_bytes() == candidate
    assert control.read_context().document() == before.document()
    assert control.read_context().library.document() == before.library.document()
    assert control.get_alias_rows() == ()
    assert control.get_file_revisions() == tuple(point_files.values())


@pytest.mark.parametrize(
    ("method", "document", "params"),
    [writer for writer in WRITERS if writer[1] != "aliases"],
)
def test_optional_writer_revision_accepts_matching_disk_and_checks_null(
    application: PointApplication,
    point_files: dict[Document, FileRevision],
    method: str,
    document: Document,
    params: dict[str, object],
) -> None:
    file = point_files[document]
    error = error_reply(application, method, {**params, "expected_revision": None})
    if document == "parameters":
        assert error["code"] == ErrorCode.INVALID_PARAMS, error
    else:
        assert (error["code"], error.get("reason")) == (
            ErrorCode.PRECONDITION_FAILED,
            "revision_mismatch",
        ), error
    reply = application.rpc(method, {**params, "expected_revision": file.revision})
    assert reply["ok"] is True, reply
    installed = {
        item.document: item
        for item in application.controller.context_control.get_file_revisions()
    }
    assert installed[document].revision == sha256(file.path.read_bytes()).hexdigest()
    assert installed[document].revision != file.revision


def test_alias_writer_explicit_null_does_not_skip_missing_file_guard(
    application: PointApplication, point_files: dict[Document, FileRevision]
) -> None:
    file = point_files["aliases"]
    candidate = external_edit(file)
    error = error_reply(
        application, "context.aliases_set", {"rows": [], "expected_revision": None}
    )
    assert (error["code"], error.get("reason")) == (
        ErrorCode.PRECONDITION_FAILED,
        "revision_mismatch",
    )
    assert file.path.read_bytes() == candidate
    assert application.controller.context_control.get_alias_rows() == ()
