"""Dispersive and T1 fit file owner contracts, independent of analysis completeness."""

import importlib
import json
from pathlib import Path

import pytest


def document(kind: str) -> dict[str, object]:
    """Return incomplete but representable analysis data with unknown provenance."""
    params: dict[str, object]
    if kind == "dispersive":
        params = {
            "g": None,
            "bare_rf": 5.1,
            "plot_transitions": {"r_f": None, "0.1": [1, 2], "future": {"x": True}},
        }
    else:
        params = {
            "params": {"Q_cap": None, "Temp": 0.04},
            "stderr": {"Q_cap": None},
            "fixed": ["Temp"],
            "free": ["Q_cap"],
            "bounds": {"Q_cap": [1, None]},
            "init": {"Q_cap": 1000, "Temp": None},
            "cost": None,
            "success": False,
        }
    return {
        "format": f"zcu.{kind.replace('_', '-')}",
        "version": 1,
        "params": params,
        "source": {"run_id": None, "created_at": None},
    }


@pytest.mark.parametrize("kind", ["dispersive", "t1_curve"])
def test_owner_roundtrip_retains_unknowns_without_creating_provenance(
    tmp_path: Path, kind: str
) -> None:
    owner = importlib.import_module(f"zcu_tools.resources.{kind}_file")
    source = tmp_path / "input.json"
    source.write_text(json.dumps(document(kind)), encoding="utf-8")
    load = getattr(owner, f"load_{kind}_file")
    save = getattr(owner, f"save_{kind}_file")
    loaded = load(source)
    target = tmp_path / "output.json"
    save(target, loaded)
    assert load(target) == loaded
    assert loaded.source.run_id is None
    assert loaded.source.created_at is None
    if kind == "dispersive":
        assert loaded.params.g is None
        assert loaded.params.plot_transitions.r_f is None
        assert loaded.params.plot_transitions.model_extra == {
            "0.1": [1, 2],
            "future": {"x": True},
        }
    else:
        assert loaded.params.params.Q_cap is None
        assert loaded.params.params.model_fields_set == {"Q_cap", "Temp"}
        assert loaded.params.bounds["Q_cap"] == (1.0, None)
        assert loaded.params.success is False


@pytest.mark.parametrize("kind", ["dispersive", "t1_curve"])
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
    tmp_path: Path, kind: str, path: tuple[str, ...], bad: object
) -> None:
    owner = importlib.import_module(f"zcu_tools.resources.{kind}_file")
    raw = document(kind)
    if len(path) == 1:
        raw[path[0]] = bad
    else:
        source = dict(run_id=None, created_at=None)
        raw["source"] = {**source, path[1]: bad}
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=path[0]):
        getattr(owner, f"load_{kind}_file")(file)


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("dispersive", {"g": True}),
        ("dispersive", {"bare_rf": "5.0"}),
        ("dispersive", {"plot_transitions": {"r_f": float("inf")}}),
        ("t1_curve", {"params": {"Temp": True}}),
        ("t1_curve", {"params": {"Q_cap": float("nan")}}),
        ("t1_curve", {"params": {"Temp": None}, "free": ["Q_cap"]}),
        ("t1_curve", {"params": {"Temp": None}, "bounds": {"Q_cap": [1, 2]}}),
        ("t1_curve", {"params": {"Temp": None}, "stderr": {"Q_cap": 1}}),
        ("t1_curve", {"params": {"Temp": None}, "init": {"Q_cap": 1}}),
    ],
)
def test_owner_rejects_bad_content(
    tmp_path: Path, kind: str, params: dict[str, object]
) -> None:
    owner = importlib.import_module(f"zcu_tools.resources.{kind}_file")
    raw = document(kind)
    raw["params"] = params
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="params"):
        getattr(owner, f"load_{kind}_file")(file)


@pytest.mark.parametrize("kind", ["dispersive", "t1_curve"])
def test_save_revalidates_and_retains_existing_file_on_invalid_input(
    tmp_path: Path, kind: str
) -> None:
    owner = importlib.import_module(f"zcu_tools.resources.{kind}_file")
    file = tmp_path / "fit.json"
    file.write_text(json.dumps(document(kind)), encoding="utf-8")
    loaded = getattr(owner, f"load_{kind}_file")(file)
    invalid = loaded.model_copy(update={"version": True})
    before = file.read_bytes()
    with pytest.raises(ValueError, match="version"):
        getattr(owner, f"save_{kind}_file")(file, invalid)
    assert file.read_bytes() == before


@pytest.mark.parametrize("kind", ["dispersive", "t1_curve"])
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
def test_owner_preserves_proven_source_text(
    tmp_path: Path, kind: str, created_at: str
) -> None:
    owner = importlib.import_module(f"zcu_tools.resources.{kind}_file")
    raw = document(kind)
    raw["source"] = {
        "run_id": "historical-run",
        "created_at": created_at,
    }
    file = tmp_path / "fit.json"
    file.write_text(json.dumps(raw), encoding="utf-8")
    loaded = getattr(owner, f"load_{kind}_file")(file)
    getattr(owner, f"save_{kind}_file")(file, loaded)
    assert loaded.source.created_at == created_at
    assert json.loads(file.read_text(encoding="utf-8"))["source"] == raw["source"]


@pytest.mark.parametrize("kind", ["dispersive", "t1_curve"])
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
    tmp_path: Path, kind: str, operation: str, created_at: str
) -> None:
    owner = importlib.import_module(f"zcu_tools.resources.{kind}_file")
    file = tmp_path / "fit.json"
    raw = document(kind)
    file.write_text(json.dumps(raw), encoding="utf-8")
    load = getattr(owner, f"load_{kind}_file")
    save = getattr(owner, f"save_{kind}_file")
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
