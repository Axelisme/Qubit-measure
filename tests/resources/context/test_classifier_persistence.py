"""Calibrated and uncalibrated readouts retain their library YAML contracts."""

import json
from pathlib import Path

import pytest
from ruamel.yaml import YAML
from zcu_tools.cfg_model import (
    CircleClassifier,
    LinearClassifier,
    MuxReadoutSpec,
    PulseReadoutSpec,
    PulseSpec,
)
from zcu_tools.resources.context import ModuleLibrary


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    """Create empty library YAML and return its path plus an unused ledger path."""
    path = tmp_path / "module_cfg.yaml"
    path.write_text(
        "format: zcu.module-library\nformat_version: '2.0'\n", encoding="utf-8"
    )
    records = tmp_path / "records"
    records.mkdir()
    return path, records / "ledger.jsonl"


def open_library(files: tuple[Path, Path]) -> ModuleLibrary:
    """Open the fixture library with a fixed entry identity and no asset provider."""
    return ModuleLibrary(
        files[0], ledger=files[1], entry_id="78fa0941-f9ce-49d4-9665-4eb9302b8162"
    )


@pytest.mark.parametrize("family", ["pulse", "mux"])
@pytest.mark.parametrize("mode", ["circle", "linear"])
def test_readout_classifier_survives_library_edit_and_reopen(
    files: tuple[Path, Path], family: str, mode: str
) -> None:
    classifier = (
        CircleClassifier(
            g_center=1.0 - 2.0j,
            e_center=-3.0 + 4.0j,
            radius=0.5,
            confusion_matrix=[[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.0]],
        )
        if mode == "circle"
        else LinearClassifier(
            angle=-0.75, threshold=0.25, confusion_matrix=[[0.9, 0.1], [0.2, 0.8]]
        )
    )
    raw = {
        "type": f"readout/{family}",
        "gen_ch": 0,
        "nqz": 1,
        "ro_length": 1.5,
    }
    spec = (
        PulseReadoutSpec.model_validate(
            {
                **raw,
                "freq": 700.0,
                "gain": 0.2,
                "waveform": "Q1.waveforms.base",
                "ro_ch": 1,
                "classifier": classifier,
            }
        )
        if family == "pulse"
        else MuxReadoutSpec.model_validate(
            {
                **raw,
                "mixer_freq": 6000.0,
                "length": 1.0,
                "tones": [
                    {"freq": 6020.0, "gain": 0.1, "ro_ch": 3, "classifier": classifier},
                    {"freq": 5980.0, "gain": 0.2, "ro_ch": 1},
                ],
            }
        )
    )
    library = open_library(files)
    with library.edit() as draft:
        draft.set_module("Q1.readout.calibrated", spec)

    reopened = open_library(files).snapshot()
    assert reopened.module_spec("Q1.readout.calibrated") == spec
    assert (
        reopened.entry_value("Q1.readout.calibrated", kind="module")
        == spec.model_dump()
    )
    saved = YAML(typ="safe").load(files[0].read_text())
    saved_readout = saved["components"]["Q1"]["readout"]["calibrated"]
    saved_classifier = (
        saved_readout["classifier"]
        if family == "pulse"
        else saved_readout["tones"][0]["classifier"]
    )
    assert saved_classifier == classifier.to_dict()


def test_uncalibrated_library_preserves_old_readout_fields_on_unrelated_edit(
    files: tuple[Path, Path],
) -> None:
    old_readouts = {
        "pulse": {
            "type": "readout/pulse",
            "gen_ch": 0,
            "nqz": 1,
            "freq": 700.0,
            "gain": 0.2,
            "waveform": "Q1.waveforms.base",
            "ro_ch": 1,
            "ro_length": 1.5,
        },
        "mux": {
            "type": "readout/mux",
            "gen_ch": 2,
            "nqz": 2,
            "mixer_freq": 6000.0,
            "length": 1.0,
            "ro_length": 1.5,
            "tones": [{"freq": 6020.0, "gain": 0.1, "ro_ch": 3}],
        },
    }
    files[0].write_text(
        json.dumps(
            {
                "format": "zcu.module-library",
                "format_version": "2.0",
                "components": {"Q1": {"readout": old_readouts}},
            }
        ),
        encoding="utf-8",
    )
    original = files[0].read_bytes()
    library = open_library(files)
    assert files[0].read_bytes() == original
    pulse_spec = library.snapshot().module_spec("Q1.readout.pulse")
    mux_spec = library.snapshot().module_spec("Q1.readout.mux")
    assert isinstance(pulse_spec, PulseReadoutSpec)
    assert isinstance(mux_spec, MuxReadoutSpec)
    assert pulse_spec.classifier is None
    assert mux_spec.tones[0].classifier is None

    with library.edit() as draft:
        draft.set_module(
            "Q1.pulses.pi",
            PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.25, waveform="Q1.waveforms.base"),
        )

    saved = YAML(typ="safe").load(files[0].read_text())
    assert saved["components"]["Q1"]["readout"] == old_readouts
