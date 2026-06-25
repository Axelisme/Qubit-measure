from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.v2.onetone.flux_dep import FluxDepExp as OneToneFluxDepExp
from zcu_tools.experiment.v2.onetone.freq import FreqExp as OneToneFreqExp
from zcu_tools.experiment.v2.twotone.fluxdep import FreqFluxExp
from zcu_tools.utils.datasaver import load_labber_data, save_labber_data

from script.migrate_experiment_data import migrate_experiment_data


def _write_legacy_onetone_freq(path: Path) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.array([5000.0, 5001.0, 5002.0], dtype=np.float64)
    signals = np.array([1.0 + 0.1j, 0.8 + 0.2j, 1.1 + 0.3j], dtype=np.complex128)
    save_labber_data(
        str(path),
        z=("Signal", "ADC unit", signals),
        axes=[("Frequency", "MHz", freqs)],
        comment="legacy comment",
        tags=["onetone/freq"],
    )
    return freqs, signals


def _write_legacy_flux_dep(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.array([4300.0, 4350.0, 4400.0], dtype=np.float64)
    values = np.array([-0.25, 0.45], dtype=np.float64)
    signals = np.array(
        [[1.0 + 0.1j, 1.1 + 0.2j, 1.2 + 0.3j], [2.0 + 0.4j, 2.1 + 0.5j, 2.2 + 0.6j]],
        dtype=np.complex128,
    )
    save_labber_data(
        str(path),
        z=("Signal", "ADC unit", signals),
        axes=[
            ("Frequency", "Hz", freqs * 1e6),
            ("Yoko", "A", values),
        ],
        comment="legacy flux comment",
        tags=["TwoTone"],
    )
    return freqs, values, signals


def test_migrate_onetone_freq_mhz_legacy_file_to_canonical(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy_onetone_freq.hdf5"
    freqs, signals = _write_legacy_onetone_freq(legacy_path)
    legacy_bytes = legacy_path.read_bytes()

    migrated = migrate_experiment_data(
        experiment="onetone/freq",
        input_path=legacy_path,
        output_path=tmp_path / "canonical.hdf5",
    )

    assert legacy_path.read_bytes() == legacy_bytes
    loaded = OneToneFreqExp().load(str(migrated))
    np.testing.assert_allclose(loaded.freqs, freqs, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.signals, signals, rtol=0, atol=0)

    raw = load_labber_data(str(migrated))
    assert raw.axes[0].name == "Frequency"
    assert raw.axes[0].unit == "Hz"
    np.testing.assert_allclose(raw.axes[0].values, freqs * 1e6, rtol=0, atol=1e-3)
    assert raw.data.unit == "a.u."
    assert raw.comment == "legacy comment"


def test_migrate_twotone_flux_dep_legacy_file_to_canonical(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy_twotone_flux.hdf5"
    freqs, values, signals = _write_legacy_flux_dep(legacy_path)

    with pytest.raises(ValueError, match="canonical axis 1 label"):
        FreqFluxExp().load(str(legacy_path))

    migrated = migrate_experiment_data(
        experiment="twotone/flux_dep",
        input_path=legacy_path,
        output_path=tmp_path / "canonical_twotone_flux.hdf5",
    )

    loaded = FreqFluxExp().load(str(migrated))
    np.testing.assert_allclose(loaded.freqs, freqs, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.values, values, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.signals, signals, rtol=0, atol=0)

    raw = load_labber_data(str(migrated))
    assert [(axis.name, axis.unit) for axis in raw.axes] == [
        ("Frequency", "Hz"),
        ("Flux device value", "a.u."),
    ]
    assert raw.data.unit == "a.u."


def test_migrate_onetone_flux_dep_uses_result_native_axis_order(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy_onetone_flux.hdf5"
    freqs, values, signals = _write_legacy_flux_dep(legacy_path)

    migrated = migrate_experiment_data(
        experiment="onetone/flux_dep",
        input_path=legacy_path,
        output_path=tmp_path / "canonical_onetone_flux.hdf5",
    )

    loaded = OneToneFluxDepExp().load(str(migrated))
    np.testing.assert_allclose(loaded.freqs, freqs, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.values, values, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.signals, signals, rtol=0, atol=0)


def test_migration_rejects_unsupported_legacy_axis_unit(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy_bad_unit.hdf5"
    freqs = np.array([1.0, 2.0], dtype=np.float64)
    save_labber_data(
        str(legacy_path),
        z=("Signal", "ADC unit", np.ones(2, dtype=np.complex128)),
        axes=[("Frequency", "furlong", freqs)],
    )

    with pytest.raises(ValueError, match="axis 0 unit"):
        migrate_experiment_data(
            experiment="onetone/freq",
            input_path=legacy_path,
            output_path=tmp_path / "canonical_bad_unit.hdf5",
        )
