"""Public zcu.predictor v2 read/write contracts; no numerical model construction."""

import json
from pathlib import Path

import pytest
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorFlux,
    PredictorSource,
    load_predictor_file,
    save_predictor_file,
    validate_predictor_file,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.parameters import ModelName, params_class
from zcu_tools.simulate.transmon import TransmonParams


@pytest.fixture
def document() -> dict[str, object]:
    return {
        "format": "zcu.predictor",
        "version": 2,
        "model": "fluxonium",
        "params": {
            "EJ": 5.0,
            "EC": 1.0,
            "EL": 0.5,
        },
        "flux": {"offset": 0.25, "period": {"bias": -0.4}, "unit": {"bias": "mA"}},
        "source": {"run_id": "run-α", "created_at": "2026-10-20T12:34:56+08:00"},
    }


def write_document(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def save_or_validate(operation: str, path: Path, predictor: PredictorFile) -> None:
    """Exercise the requested owner write or pure validation boundary."""
    if operation == "save":
        save_predictor_file(path, predictor)
    elif operation == "validate":
        validate_predictor_file(predictor)
    else:
        raise ValueError(f"Unknown operation: {operation}")


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (
            "fluxonium",
            {
                "EJ": 5.0,
                "EC": 1,
                "EL": 0.5,
            },
        ),
        (
            "transmon",
            {
                "EJmax": 20,
                "EC": 0.3,
                "d": 0.1,
            },
        ),
        ("fluxonium", {"EJ": None}),
        ("transmon", {}),
        ("fluxonium", {"EJ": -1, "EC": 0}),
        ("transmon", {"EC": -1, "d": 2}),
    ],
    ids=[
        "fluxonium",
        "transmon",
        "null-unknown",
        "omitted-unknown",
        "fluxonium-physical-check-deferred",
        "transmon-physical-check-deferred",
    ],
)
def test_load_selects_numeric_type_without_completing_or_building_model(
    tmp_path: Path,
    document: dict[str, object],
    model: ModelName,
    values: dict[str, object],
) -> None:
    document.update(model=model, params=values)
    path = write_document(tmp_path / "predictor.json", document)
    before = path.read_bytes()
    expected = PredictorFile(
        model=model,
        params=params_class(model).from_mapping(values),
        flux=PredictorFlux(0.25, {"bias": -0.4}, {"bias": "mA"}),
        source=PredictorSource("run-α", "2026-10-20T12:34:56+08:00"),
    )

    assert load_predictor_file(path) == expected
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "timestamp", [None, "2026-10-20T04:34:56Z", "2026-10-20T12:34:56.123+08:00"]
)
def test_load_preserves_unknown_metadata_and_timezone_text(
    tmp_path: Path,
    document: dict[str, object],
    timestamp: str | None,
) -> None:
    document.update(
        flux={"offset": 0.0, "period": {}, "unit": {}},
        params={},
        source={"run_id": None, "created_at": timestamp},
    )
    path = write_document(tmp_path / "predictor.json", document)

    assert load_predictor_file(path) == PredictorFile(
        model="fluxonium",
        params=FluxoniumParams(),
        flux=PredictorFlux(0.0, {}, {}),
        source=PredictorSource(None, timestamp),
    )


@pytest.mark.parametrize("operation", ["load", "save", "validate"])
@pytest.mark.parametrize(
    "timestamp",
    [
        pytest.param("2026-10-20X12:34:56Z", id="x-separator"),
        pytest.param("2026-10-20 12:34:56Z", id="space-separator"),
        pytest.param("2026-10-20\n12:34:56Z", id="newline-separator"),
        pytest.param("20261020T12:34:56Z", id="basic-date"),
        pytest.param("2026-W43-2T12:34:56Z", id="week-date"),
        pytest.param("2026-10-20T123456Z", id="basic-time"),
        pytest.param("2026-10-20T12:34:56+0800", id="basic-offset"),
        pytest.param("2026-10-20T12:34:56+08", id="hour-offset"),
        pytest.param("2026-10-20T12:34:56+08:00:30", id="second-offset"),
        pytest.param("2026-10-20T12:34:56+08:00:30.5", id="fractional-offset"),
        pytest.param("2026-10-20T12:34:56,123Z", id="comma-fraction"),
        pytest.param("2026-10-20T12:34:56.1234567Z", id="excessive-fraction"),
        pytest.param("2026-10-20T12Z", id="hour-only"),
        pytest.param("2026-10-20T12:34.5Z", id="fractional-minute"),
    ],
)
def test_rejects_timestamp_outside_extended_shape(
    tmp_path: Path,
    document: dict[str, object],
    operation: str,
    timestamp: str,
) -> None:
    document["source"] = {"run_id": None, "created_at": timestamp}
    path = write_document(tmp_path / "predictor.json", document)
    before = path.read_bytes()

    if operation == "load":
        with pytest.raises(ValueError, match="."):
            load_predictor_file(path)
    else:
        predictor = PredictorFile(
            "fluxonium",
            FluxoniumParams(),
            PredictorFlux(0.0, {}, {}),
            PredictorSource(None, timestamp),
        )
        with pytest.raises(ValueError, match="."):
            save_or_validate(operation, path, predictor)

    assert path.read_bytes() == before


@pytest.mark.parametrize("operation", ["load", "save", "validate"])
@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-293T12:34:56Z",
        "2026-02-30T12:34:56Z",
        "2026-10-20T24:34:56Z",
        "2026-10-20T12:34:56+24:00",
    ],
    ids=["ordinal-date", "invalid-calendar-date", "invalid-time", "invalid-offset"],
)
def test_rejects_invalid_timestamp_values(
    tmp_path: Path,
    document: dict[str, object],
    operation: str,
    timestamp: str,
) -> None:
    document["source"] = {"run_id": None, "created_at": timestamp}
    path = write_document(tmp_path / "predictor.json", document)
    if operation == "load":
        with pytest.raises(ValueError, match="."):
            load_predictor_file(path)
    else:
        predictor = PredictorFile(
            "fluxonium",
            FluxoniumParams(),
            PredictorFlux(0.0, {}, {}),
            PredictorSource(None, timestamp),
        )
        with pytest.raises(ValueError, match="."):
            save_or_validate(operation, path, predictor)


@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        "2026-10-20T12:34Z",
        "2026-10-20T12:34+08:00",
        "2026-10-20T12:34-03:30",
        "2026-10-20T12:34:56Z",
        "2026-10-20T12:34:56+00:00",
        "2026-10-20T12:34:56.1Z",
        "2026-10-20T12:34:56.123456-03:30",
    ],
)
def test_timestamp_extended_shape_round_trips_verbatim(
    tmp_path: Path, timestamp: str | None
) -> None:
    path = tmp_path / "predictor.json"
    predictor = PredictorFile(
        "fluxonium",
        FluxoniumParams(),
        PredictorFlux(0.0, {}, {}),
        PredictorSource(None, timestamp),
    )
    save_predictor_file(path, predictor)

    assert load_predictor_file(path) == predictor
    assert (
        json.loads(path.read_text(encoding="utf-8"))["source"]["created_at"]
        == timestamp
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"format": "other"},
        {"version": 3},
        {"version": True},
        {"version": 1.0},
        {"version": "1"},
        {"model": "unknown"},
        {"model": None},
        {"params": []},
        {"params": None},
        {"params": {"typo": 1}},
        {"params": {"EJmax": 1}},
        {"params": {"EJ": True}},
        {"params": {"EJ": "5"}},
        {"params": {"EC": float("nan")}},
        {"params": {"EL": float("inf")}},
        {"params": {"flux_period": -float("inf")}},
        {"model": "transmon", "params": {"EJ": 1}},
        {"model": "transmon", "params": {"d": False}},
        {"model": "transmon", "params": {"EJmax": float("nan")}},
        {"flux": {"offset": 0, "period": {"bias": 1}, "unit": {"bias": ""}}},
        {"flux": {"offset": 0, "period": {"bias": 1}, "unit": {"bias": "  "}}},
        {"flux": {"offset": 0, "period": {"bias": 1}, "unit": {"bias": 1}}},
        {"source": None},
        {"source": []},
        {"source": {"run_id": "", "created_at": None}},
        {"source": {"run_id": False, "created_at": None}},
        {"source": {"run_id": None, "created_at": ""}},
        {"source": {"run_id": None, "created_at": 42}},
        {"source": {"run_id": None, "created_at": "yesterday"}},
        {"source": {"run_id": None, "created_at": "2026-10-20"}},
        {"source": {"run_id": None, "created_at": "2026-10-20T12:00:00"}},
        {"source": {"run_id": None, "created_at": "2026-02-30T12:00:00Z"}},
        {"source": {"run_id": None, "created_at": None, "extra": 1}},
        {"extra": 1},
    ],
    ids=[
        "wrong-format",
        "future-version",
        "bool-version",
        "float-version",
        "text-version",
        "unknown-model",
        "null-model",
        "array-params",
        "null-params",
        "unknown-parameter",
        "wrong-model-parameter",
        "bool-parameter",
        "text-parameter",
        "nan-parameter",
        "inf-parameter",
        "negative-inf-parameter",
        "transmon-wrong-parameter",
        "transmon-bool",
        "transmon-nan",
        "empty-unit",
        "blank-unit",
        "numeric-unit",
        "null-source",
        "array-source",
        "empty-run",
        "bool-run",
        "empty-time",
        "numeric-time",
        "invalid-time",
        "date-only",
        "naive-time",
        "invalid-date",
        "extra-source",
        "extra-envelope",
    ],
)
def test_load_rejects_invalid_envelope_metadata_and_parameters(
    tmp_path: Path,
    document: dict[str, object],
    changes: dict[str, object],
) -> None:
    document.update(changes)
    path = write_document(tmp_path / "invalid.json", document)

    with pytest.raises(ValueError, match="."):
        load_predictor_file(path)


@pytest.mark.parametrize(
    "field", ["format", "version", "model", "params", "flux", "source"]
)
def test_load_requires_every_envelope_field(
    tmp_path: Path,
    document: dict[str, object],
    field: str,
) -> None:
    del document[field]
    path = write_document(tmp_path / "missing.json", document)

    with pytest.raises(ValueError, match=field):
        load_predictor_file(path)


@pytest.mark.parametrize("field", ["run_id", "created_at"])
def test_load_requires_nullable_source_fields(
    tmp_path: Path,
    document: dict[str, object],
    field: str,
) -> None:
    source: dict[str, object] = {"run_id": None, "created_at": None}
    del source[field]
    document["source"] = source
    path = write_document(tmp_path / "missing.json", document)

    with pytest.raises(ValueError, match=field):
        load_predictor_file(path)


@pytest.mark.parametrize("content", ["[]", "null", "42", "{broken"])
def test_load_rejects_non_object_or_malformed_json(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="."):
        load_predictor_file(path)


def test_load_energy_only_v2_with_multiple_flux_sources(tmp_path: Path) -> None:
    path = write_document(
        tmp_path / "v2.json",
        {
            "format": "zcu.predictor",
            "version": 2,
            "model": "fluxonium",
            "params": {"EJ": 5, "EC": 1, "EL": 0.5},
            "flux": {
                "offset": 0.25,
                "period": {"bias": -2.0, "coil": 4.0},
                "unit": {"bias": "mA", "coil": "A"},
            },
            "source": {"run_id": None, "created_at": None},
        },
    )
    loaded = load_predictor_file(path)
    assert loaded.params.to_dict() == {"EJ": 5, "EC": 1, "EL": 0.5}
    assert loaded.flux.offset == 0.25
    assert loaded.flux.period == {"bias": -2.0, "coil": 4.0}
    save_predictor_file(path, loaded)
    assert load_predictor_file(path) == loaded


@pytest.mark.parametrize(
    "model, energies",
    [
        ("fluxonium", {"EJ": 5, "EC": 1, "EL": 0.5}),
        ("transmon", {"EJmax": 20, "EC": 0.3, "d": 0.1}),
    ],
)
@pytest.mark.parametrize("unit", ["mA", None])
def test_upgrade_v1_preserves_energy_and_unwrapped_alignment(
    model: ModelName, energies: dict[str, float], unit: str | None
) -> None:
    from copy import deepcopy

    from zcu_tools.resources import predictor_file

    document = {
        "format": "zcu.predictor",
        "version": 1,
        "model": model,
        "params": {**energies, "flux_half": 0.3, "flux_period": -0.4, "flux_bias": 0.1},
        "flux_unit": unit,
        "source": {"run_id": None, "created_at": None},
    }
    before = deepcopy(document)
    upgraded = predictor_file.upgrade_predictor_document(document)
    assert upgraded.params.to_dict() == energies
    assert upgraded.flux == PredictorFlux(
        1.0, {"unnamed_device": -0.4}, {"unnamed_device": unit or "native"}
    )
    assert document == before


def test_v1_reader_requires_explicit_upgrade(tmp_path: Path) -> None:
    path = write_document(
        tmp_path / "old.json",
        {
            "format": "zcu.predictor",
            "version": 1,
            "model": "fluxonium",
            "params": {},
            "flux_unit": None,
            "source": {"run_id": None, "created_at": None},
        },
    )
    with pytest.raises(ValueError, match="scripts/upgrade_parameters.py.*<entry>"):
        load_predictor_file(path)


@pytest.mark.parametrize(
    "flux",
    [
        None,
        {},
        {"offset": None, "period": {}, "unit": {}},
        {"offset": True, "period": {}, "unit": {}},
        {"offset": float("inf"), "period": {}, "unit": {}},
        {"offset": 0, "period": {"bias": 0}, "unit": {"bias": "A"}},
        {"offset": 0, "period": {"bias": True}, "unit": {"bias": "A"}},
        {"offset": 0, "period": {"bias": None}, "unit": {"bias": "A"}},
        {"offset": 0, "period": {"bias": float("nan")}, "unit": {"bias": "A"}},
        {"offset": 0, "period": {"bias": 1}, "unit": {}},
        {"offset": 0, "period": {}, "unit": {"extra": "A"}},
        {"offset": 0, "period": {"bad.name": 1}, "unit": {"bad.name": "A"}},
        {"offset": 0, "period": {}, "unit": {}, "extra": 1},
    ],
)
def test_load_rejects_invalid_flux_frame(
    tmp_path: Path, document: dict[str, object], flux: object
) -> None:
    document["flux"] = flux
    with pytest.raises(ValueError, match="flux|offset|period|[Dd]evice"):
        load_predictor_file(write_document(tmp_path / "bad.json", document))


def test_predictor_flux_snapshots_inputs_and_reuses_analysis_frame() -> None:
    from zcu_tools.analysis.flux import flux_from_device_values

    periods, units = {"bias": -2.0}, {"bias": "A"}
    flux = PredictorFlux(1.25, periods, units)
    periods["bias"] = 5.0
    units["bias"] = "V"
    assert flux_from_device_values(flux.to_frame(), {"bias": 0.5}) == 1.0
    assert flux.unit == {"bias": "A"}


@pytest.mark.parametrize(
    "affine",
    [
        {"flux_half": None, "flux_period": 1, "flux_bias": 0},
        {"flux_half": 0, "flux_period": 0, "flux_bias": 0},
        {"flux_half": 0, "flux_period": 1, "flux_bias": float("inf")},
    ],
)
def test_upgrade_rejects_unusable_affine_without_mutating_input(
    affine: dict[str, object],
) -> None:
    from copy import deepcopy

    from zcu_tools.resources.predictor_file import upgrade_predictor_document

    document = {
        "format": "zcu.predictor",
        "version": 1,
        "model": "fluxonium",
        "params": {"EJ": None, **affine},
        "source": {"run_id": None, "created_at": None},
    }
    before = deepcopy(document)
    with pytest.raises(ValueError, match="flux"):
        upgrade_predictor_document(document)
    assert document == before


def test_load_missing_file_propagates_io_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_predictor_file(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "predictor",
    [
        PredictorFile(
            "fluxonium",
            FluxoniumParams(EJ=5),
            PredictorFlux(0.25, {"bias": -0.4}, {"bias": "mA"}),
            PredictorSource("run-α", "2026-10-20T12:34:56+08:00"),
        ),
        PredictorFile(
            "transmon",
            TransmonParams(EJmax=20, d=0),
            PredictorFlux(0.0, {}, {}),
            PredictorSource(None, None),
        ),
    ],
    ids=["fluxonium", "transmon"],
)
def test_save_writes_versioned_json_and_round_trips(
    tmp_path: Path,
    predictor: PredictorFile,
) -> None:
    destination = tmp_path / "predictor.json"

    save_predictor_file(destination, predictor)

    assert destination.is_file()
    encoded = json.loads(destination.read_text(encoding="utf-8"))
    assert encoded == {
        "format": "zcu.predictor",
        "version": 2,
        "model": predictor.model,
        "params": predictor.params.to_dict(),
        "flux": {
            "offset": predictor.flux.offset,
            "period": dict(predictor.flux.period),
            "unit": dict(predictor.flux.unit),
        },
        "source": {
            "run_id": predictor.source.run_id,
            "created_at": predictor.source.created_at,
        },
    }
    assert load_predictor_file(destination) == predictor


@pytest.mark.parametrize("operation", ["save", "validate"])
def test_rejects_mismatched_model_before_overwriting(
    tmp_path: Path, operation: str
) -> None:
    destination = tmp_path / "predictor.json"
    destination.write_text("keep me", encoding="utf-8")
    predictor = PredictorFile(
        "transmon",
        FluxoniumParams(),
        PredictorFlux(0.0, {}, {}),
        PredictorSource(None, None),
    )

    with pytest.raises(ValueError, match="model"):
        save_or_validate(operation, destination, predictor)
    assert destination.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("operation", ["save", "validate"])
def test_rejects_invalid_numeric_values_before_writing(
    tmp_path: Path, operation: str
) -> None:
    destination = tmp_path / "predictor.json"
    predictor = PredictorFile(
        "fluxonium",
        FluxoniumParams(EJ=float("nan")),
        PredictorFlux(0.0, {}, {}),
        PredictorSource(None, None),
    )

    with pytest.raises(ValueError, match="EJ"):
        save_or_validate(operation, destination, predictor)
    assert not destination.exists()


@pytest.mark.parametrize(
    "predictor",
    [
        PredictorFile(
            "fluxonium",
            FluxoniumParams(EC=-1),
            PredictorFlux(0.0, {}, {}),
            PredictorSource(None, None),
        ),
        PredictorFile(
            "transmon",
            TransmonParams(d=2),
            PredictorFlux(0.0, {}, {}),
            PredictorSource("historical-run", "2026-10-20T12:34-03:30"),
        ),
    ],
)
def test_validate_accepts_partial_unphysical_inputs_without_mutation_or_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, predictor: PredictorFile
) -> None:
    monkeypatch.chdir(tmp_path)
    before = predictor.params.to_dict()
    source = predictor.source

    validate_predictor_file(predictor)

    assert predictor.params.to_dict() == before
    assert predictor.source == source
    assert list(tmp_path.iterdir()) == []


def test_save_requires_existing_parent(tmp_path: Path) -> None:
    destination = tmp_path / "missing" / "predictor.json"
    predictor = PredictorFile(
        "fluxonium",
        FluxoniumParams(),
        PredictorFlux(0.0, {}, {}),
        PredictorSource(None, None),
    )

    with pytest.raises(FileNotFoundError):
        save_predictor_file(destination, predictor)
    assert not destination.parent.exists()
