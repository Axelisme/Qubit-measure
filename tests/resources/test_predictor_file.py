"""Public zcu.predictor v1 read/write contracts; no numerical model construction."""

import json
from pathlib import Path

import pytest
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorSource,
    load_predictor_file,
    save_predictor_file,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.parameters import ModelName, params_class
from zcu_tools.simulate.transmon import TransmonParams


@pytest.fixture
def document() -> dict[str, object]:
    return {
        "format": "zcu.predictor",
        "version": 1,
        "model": "fluxonium",
        "params": {
            "EJ": 5.0,
            "EC": 1.0,
            "EL": 0.5,
            "flux_half": 0.1,
            "flux_period": -0.4,
            "flux_bias": 0.0,
        },
        "flux_unit": "mA",
        "source": {"run_id": "run-α", "created_at": "2026-10-20T12:34:56+08:00"},
    }


def write_document(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (
            "fluxonium",
            {
                "EJ": 5.0,
                "EC": 1,
                "EL": 0.5,
                "flux_half": 0.1,
                "flux_period": -0.4,
                "flux_bias": 0,
            },
        ),
        (
            "transmon",
            {
                "EJmax": 20,
                "EC": 0.3,
                "d": 0.1,
                "flux_half": 0,
                "flux_period": 2,
                "flux_bias": 0.2,
            },
        ),
        ("fluxonium", {"EJ": None}),
        ("transmon", {}),
        ("fluxonium", {"EJ": -1, "flux_period": 0}),
        ("transmon", {"EC": -1, "d": 2, "flux_period": 0}),
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
        flux_unit="mA",
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
        flux_unit=None, params={}, source={"run_id": None, "created_at": timestamp}
    )
    path = write_document(tmp_path / "predictor.json", document)

    assert load_predictor_file(path) == PredictorFile(
        model="fluxonium",
        params=FluxoniumParams(),
        flux_unit=None,
        source=PredictorSource(None, timestamp),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"format": "other"},
        {"version": 2},
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
        {"flux_unit": ""},
        {"flux_unit": "  "},
        {"flux_unit": 1},
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
    "field", ["format", "version", "model", "params", "flux_unit", "source"]
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


def test_load_missing_file_propagates_io_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_predictor_file(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "predictor",
    [
        PredictorFile(
            "fluxonium",
            FluxoniumParams(EJ=5, flux_period=-2),
            "mA",
            PredictorSource("run-α", "2026-10-20T12:34:56+08:00"),
        ),
        PredictorFile(
            "transmon", TransmonParams(EJmax=20, d=0), None, PredictorSource(None, None)
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
        "version": 1,
        "model": predictor.model,
        "params": predictor.params.to_dict(),
        "flux_unit": predictor.flux_unit,
        "source": {
            "run_id": predictor.source.run_id,
            "created_at": predictor.source.created_at,
        },
    }
    assert load_predictor_file(destination) == predictor


def test_save_rejects_mismatched_model_before_overwriting(tmp_path: Path) -> None:
    destination = tmp_path / "predictor.json"
    destination.write_text("keep me", encoding="utf-8")
    predictor = PredictorFile(
        "transmon", FluxoniumParams(), None, PredictorSource(None, None)
    )

    with pytest.raises(ValueError, match="model"):
        save_predictor_file(destination, predictor)
    assert destination.read_text(encoding="utf-8") == "keep me"


def test_save_rejects_invalid_numeric_values_before_writing(tmp_path: Path) -> None:
    destination = tmp_path / "predictor.json"
    predictor = PredictorFile(
        "fluxonium", FluxoniumParams(EJ=float("nan")), None, PredictorSource(None, None)
    )

    with pytest.raises(ValueError, match="EJ"):
        save_predictor_file(destination, predictor)
    assert not destination.exists()


def test_save_requires_existing_parent(tmp_path: Path) -> None:
    destination = tmp_path / "missing" / "predictor.json"
    predictor = PredictorFile(
        "fluxonium", FluxoniumParams(), None, PredictorSource(None, None)
    )

    with pytest.raises(FileNotFoundError):
        save_predictor_file(destination, predictor)
    assert not destination.parent.exists()
