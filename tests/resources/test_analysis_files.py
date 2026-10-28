"""Dispersive fit file owner contracts, independent of analysis completeness."""

import json
from pathlib import Path

import pytest
from zcu_tools.resources.dispersive_file import (
    DispersiveFile,
)
from zcu_tools.resources.dispersive_file import (
    load_dispersive_file as load,
)
from zcu_tools.resources.dispersive_file import (
    save_dispersive_file as save,
)


def document() -> dict[str, object]:
    """Return incomplete but representable analysis data with unknown provenance."""
    return {
        "format": "zcu.dispersive",
        "version": 1,
        "params": {
            "g": None,
            "bare_rf": 5.1,
            "plot_transitions": {"r_f": None, "0.1": [1, 2], "future": {"x": True}},
        },
        "source": {"run_id": None, "created_at": None},
    }


@pytest.mark.parametrize("source", [None, [], {}, {"created_at": "naive"}])
def test_reader_ignores_unused_content_and_reports_unknown_source(tmp_path, source):
    raw = document()
    raw.pop("source")
    if source is not None:
        raw["source"] = source
    path = tmp_path / "reader.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    before = path.read_bytes()

    loaded = load(path)

    assert loaded.params.bare_rf == 5.1
    assert loaded.source is None
    assert path.read_bytes() == before


def test_reader_ignores_extra_parameter_fields(tmp_path):
    raw = document()
    raw["params"] = {"bare_rf": 5.1, "future": {"unrelated": []}}
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load(path).params.bare_rf == 5.1


@pytest.mark.parametrize("g", [0.02, None, "broken", True, float("inf"), float("nan")])
def test_reader_defers_coupling_validation_until_requested(tmp_path, g):
    raw = document()
    raw["params"] = {"g": g, "bare_rf": 5.1}
    path = tmp_path / "coupling.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    params = load(path).params
    assert params.bare_rf == 5.1
    if g == 0.02:
        assert params.require_g() == 0.02
    else:
        with pytest.raises(ValueError, match="g"):
            params.require_g()


def test_owner_roundtrip_retains_unknowns_without_creating_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.json"
    source.write_text(json.dumps(document()), encoding="utf-8")
    loaded = load(source)
    target = tmp_path / "output.json"
    save(target, loaded)
    assert load(target) == loaded
    assert loaded.source is not None
    assert loaded.source.run_id is None
    assert loaded.source.created_at is None
    assert loaded.params.g is None
    assert loaded.params.plot_transitions.r_f is None
    assert loaded.params.plot_transitions.model_extra == {
        "0.1": [1, 2],
        "future": {"x": True},
    }


@pytest.mark.parametrize(
    ("path", "bad"),
    [
        (("version",), True),
        (("version",), 1.0),
        (("version",), 2),
        (("format",), "other"),
        (("source",), {"run_id": None}),
        (("source", "run_id"), ""),
        (("source", "created_at"), "2020-07-07T16:30:00"),
        (("source", "extra"), 1),
        (("extra",), 1),
    ],
)
def test_owner_validates_header_and_projects_optional_source(
    tmp_path: Path, path: tuple[str, ...], bad: object
) -> None:
    raw = document()
    if len(path) == 1:
        raw[path[0]] = bad
    else:
        source = dict(run_id=None, created_at=None)
        raw["source"] = {**source, path[1]: bad}
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(raw), encoding="utf-8")
    if path == ("source", "extra"):
        assert load(file).source is not None
    elif path[0] == "source":
        assert load(file).source is None
    elif path[0] == "extra":
        assert load(file).params.bare_rf == 5.1
    else:
        with pytest.raises(ValueError, match=path[0]):
            load(file)


@pytest.mark.parametrize(
    "params",
    [
        {"bare_rf": "5.0"},
        {"plot_transitions": {"r_f": float("inf")}},
    ],
)
def test_owner_rejects_bad_content(tmp_path: Path, params: dict[str, object]) -> None:
    raw = document()
    raw["params"] = params
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="params"):
        load(file)


@pytest.mark.parametrize("fault", ["version", "source", "g"])
def test_save_revalidates_and_retains_existing_file_on_invalid_input(
    tmp_path: Path,
    fault: str,
) -> None:
    file = tmp_path / "fit.json"
    file.write_text(json.dumps(document()), encoding="utf-8")
    loaded = load(file)
    if fault == "g":
        invalid = loaded.model_copy(
            update={"params": loaded.params.model_copy(update={"g": "bad"})}
        )
    else:
        invalid = loaded.model_copy(
            update={fault: True if fault == "version" else None}
        )
    before = file.read_bytes()
    with pytest.raises(ValueError, match=fault):
        save(file, invalid)
    assert file.read_bytes() == before


@pytest.mark.parametrize(
    "created_at",
    [
        "2026-10-20T08:00Z",
        "2026-10-20T08:00:00Z",
        "2026-10-20T08:00:00.1Z",
        "2026-10-20T08:00:00.123456Z",
        "2026-10-20T16:00+08:00",
        "2026-10-20T16:00:00+08:00",
        "2026-10-20T01:30:00.123456-06:30",
        "2026-10-20T08:00:00+00:00",
    ],
)
def test_owner_preserves_proven_source_text(tmp_path: Path, created_at: str) -> None:
    raw = document()
    raw["source"] = {
        "run_id": "historical-run",
        "created_at": created_at,
    }
    file = tmp_path / "fit.json"
    file.write_text(json.dumps(raw), encoding="utf-8")
    loaded = load(file)
    save(file, loaded)
    assert loaded.source is not None
    assert loaded.source.created_at == created_at
    assert json.loads(file.read_text(encoding="utf-8"))["source"] == raw["source"]


@pytest.mark.parametrize("operation", ["load", "save"])
@pytest.mark.parametrize(
    "created_at",
    [
        "20261020T08:00:00Z",
        "2026-W43-2T08:00:00Z",
        "2026W432T08:00:00Z",
        "2026-293T08:00:00Z",
        "2026-10-20t08:00:00Z",
        "2026-10-20X08:00:00Z",
        "2026-10-20 08:00:00Z",
        "2026-10-20T08:00:00",
        "2026-10-20T08:00:00z",
        "2026-02-30T08:00:00Z",
        "2026-10-20T24:00:00Z",
        "2026-10-20T08:60:00Z",
        "2026-10-20T08:00:60Z",
        "2026-10-20T080000Z",
        "2026-10-20T08Z",
        "2026-10-20T08:00.1Z",
        "2026-10-20T08:00:00,1Z",
        "2026-10-20T08:00:00.1234567Z",
        "2026-10-20T08:00:00+0800",
        "2026-10-20T08:00:00+08",
        "2026-10-20T08:00:00+08:00:00",
        "2026-10-20T08:00:00+08:60",
        "2026-10-20T08:00:00+24:00",
        "2026-10-20T08:00:00Z\n",
    ],
)
def test_unproven_created_at_is_unknown_on_read_but_rejected_on_save(
    tmp_path: Path, operation: str, created_at: str
) -> None:
    file = tmp_path / "fit.json"
    raw = document()
    file.write_text(json.dumps(raw), encoding="utf-8")
    valid = DispersiveFile.model_validate(raw)
    before = file.read_bytes()
    if operation == "load":
        raw["source"] = {"run_id": None, "created_at": created_at}
        file.write_text(json.dumps(raw), encoding="utf-8")
        assert load(file).source is None
    else:
        invalid = valid.model_copy(
            update={
                "source": valid.source.model_copy(update={"created_at": created_at})
            }
        )
        with pytest.raises(ValueError, match="created_at"):
            save(file, invalid)
        assert file.read_bytes() == before
