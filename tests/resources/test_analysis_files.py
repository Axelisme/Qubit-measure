"""Dispersive fit file owner contracts, independent of analysis completeness."""

import json
from pathlib import Path

import pytest
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


def test_owner_roundtrip_retains_unknowns_without_creating_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.json"
    source.write_text(json.dumps(document()), encoding="utf-8")
    loaded = load(source)
    target = tmp_path / "output.json"
    save(target, loaded)
    assert load(target) == loaded
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
def test_owner_rejects_invalid_envelope(
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
    with pytest.raises(ValueError, match=path[0]):
        load(file)


@pytest.mark.parametrize(
    "params",
    [
        {"g": True},
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


def test_save_revalidates_and_retains_existing_file_on_invalid_input(
    tmp_path: Path,
) -> None:
    file = tmp_path / "fit.json"
    file.write_text(json.dumps(document()), encoding="utf-8")
    loaded = load(file)
    invalid = loaded.model_copy(update={"version": True})
    before = file.read_bytes()
    with pytest.raises(ValueError, match="version"):
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
def test_owner_rejects_unproven_created_at_on_load_and_save(
    tmp_path: Path, operation: str, created_at: str
) -> None:
    file = tmp_path / "fit.json"
    raw = document()
    file.write_text(json.dumps(raw), encoding="utf-8")
    valid = load(file)
    before = file.read_bytes()
    if operation == "load":
        raw["source"] = {"run_id": None, "created_at": created_at}
        file.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="created_at"):
            load(file)
    else:
        invalid = valid.model_copy(
            update={
                "source": valid.source.model_copy(update={"created_at": created_at})
            }
        )
        with pytest.raises(ValueError, match="created_at"):
            save(file, invalid)
        assert file.read_bytes() == before
