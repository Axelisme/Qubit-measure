from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.legacy_migration import CONVERTERS as LIB_CONVERTERS
from zcu_tools.experiment.v2.onetone.flux_dep import FluxDepExp as OneToneFluxDepExp
from zcu_tools.experiment.v2.onetone.freq import FreqExp as OneToneFreqExp
from zcu_tools.experiment.v2.twotone.fluxdep import FreqFluxExp
from zcu_tools.utils.datasaver import load_labber_data, save_labber_data

import script.migrate_experiment_data as migrate_mod
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


def test_migration_rejects_same_input_and_output_path_with_overwrite(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy_onetone_freq.hdf5"
    _write_legacy_onetone_freq(legacy_path)
    legacy_bytes = legacy_path.read_bytes()

    with pytest.raises(ValueError, match="same file"):
        migrate_experiment_data(
            experiment="onetone/freq",
            input_path=legacy_path,
            output_path=legacy_path,
            overwrite=True,
        )
    assert legacy_path.read_bytes() == legacy_bytes


def test_migration_rejects_pathname_alias_of_input_with_overwrite(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy_onetone_freq.hdf5"
    _write_legacy_onetone_freq(legacy_path)
    alias = tmp_path / "sub" / ".." / "legacy_onetone_freq.hdf5"
    legacy_bytes = legacy_path.read_bytes()

    with pytest.raises(ValueError, match="same file"):
        migrate_experiment_data(
            experiment="onetone/freq",
            input_path=legacy_path,
            output_path=alias,
            overwrite=True,
        )
    assert legacy_path.read_bytes() == legacy_bytes


def test_migration_rejects_symlink_alias_of_input_with_overwrite(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy_onetone_freq.hdf5"
    _write_legacy_onetone_freq(legacy_path)
    alias = tmp_path / "alias.hdf5"
    alias.symlink_to(legacy_path)
    legacy_bytes = legacy_path.read_bytes()

    with pytest.raises(ValueError, match="same file"):
        migrate_experiment_data(
            experiment="onetone/freq",
            input_path=legacy_path,
            output_path=alias,
            overwrite=True,
        )
    assert legacy_path.read_bytes() == legacy_bytes


def test_migration_rejects_hardlink_alias_of_input_with_overwrite(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy_onetone_freq.hdf5"
    _write_legacy_onetone_freq(legacy_path)
    alias = tmp_path / "hardlink.hdf5"
    alias.hardlink_to(legacy_path)
    legacy_bytes = legacy_path.read_bytes()

    with pytest.raises(ValueError, match="same file"):
        migrate_experiment_data(
            experiment="onetone/freq",
            input_path=legacy_path,
            output_path=alias,
            overwrite=True,
        )
    assert legacy_path.read_bytes() == legacy_bytes


def test_migration_still_overwrites_separate_existing_output(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy_onetone_freq.hdf5"
    _write_legacy_onetone_freq(legacy_path)
    output = tmp_path / "canonical.hdf5"
    output.write_bytes(b"existing separate file")

    migrated = migrate_experiment_data(
        experiment="onetone/freq",
        input_path=legacy_path,
        output_path=output,
        overwrite=True,
    )
    assert migrated == output
    assert output.read_bytes() != b"existing separate file"


_SIDECAR_INPUT_SUFFIXES: dict[str, tuple[str, ...]] = {
    "singleshot/ac_stark": ("_g_pop", "_e_pop"),
    "singleshot/mist/power_freq": ("_g_population", "_e_population"),
    "singleshot/t1/t1": ("_initg", "_inite"),
    "singleshot/t1/t1_with_tone": ("_initg", "_inite"),
    "singleshot/t1/t1_with_tone_sweep": (
        "_gg_pop",
        "_ge_pop",
        "_eg_pop",
        "_ee_pop",
    ),
    "twotone/ckp": ("_ground", "_excited"),
    "twotone/ro_optimize/auto_optimize": ("_params", "_signals"),
    "jpa/jpa_auto_optimize": ("_params", "_phases", "_signals"),
}

_SINGLE_FILE_EXPERIMENTS = {
    "grouped/v1",
    "onetone/freq",
    "onetone/flux_dep",
    "twotone/freq",
    "twotone/flux_dep",
    "twotone/flux_dep/freq",
    "twotone/fluxdep",
    "singleshot/ge",
    "singleshot/len_rabi",
    "singleshot/mist/freq",
    "singleshot/mist/power",
    "singleshot/mist/pre_freq",
    "twotone/reset/bath/length",
    "twotone/cpmg",
    "jpa/jpa_auto_optimize/legacy_a",
}


def test_registered_converters_partition_into_single_file_and_sidecar() -> None:
    """Every registered converter declares its actual input resolution; the
    two declared classes (single-file vs sidecar-base) cover the whole
    registry, so no converter is left with an unverified input declaration.
    """
    assert set(migrate_mod.CONVERTERS) == _SINGLE_FILE_EXPERIMENTS | set(
        _SIDECAR_INPUT_SUFFIXES
    )


def test_registered_single_file_converters_resolve_to_the_input_itself(
    tmp_path: Path,
) -> None:
    for experiment in sorted(_SINGLE_FILE_EXPERIMENTS):
        base = tmp_path / f"legacy_{experiment.replace('/', '_')}"
        resolved = migrate_mod.CONVERTERS[experiment].input_paths(base)
        assert resolved == (base,)


def test_registered_sidecar_converters_resolve_their_exact_input_files(
    tmp_path: Path,
) -> None:
    """Sidecar-base converters resolve exactly the sidecar files their loader
    reads (same resolution helpers), so the generic guard protects every
    actual input without guessing filename patterns itself.
    """
    for experiment, suffixes in _SIDECAR_INPUT_SUFFIXES.items():
        base = tmp_path / f"legacy_{experiment.replace('/', '_')}"
        resolved = migrate_mod.CONVERTERS[experiment].input_paths(base)
        assert tuple(path.name for path in resolved) == tuple(
            base.name + suffix + ".hdf5" for suffix in suffixes
        )


def test_lib_single_file_registry_resolves_to_the_input_itself(
    tmp_path: Path,
) -> None:
    """The lib registry (used by the GUI adapter fallback) stays single-file
    and resolves each converter's input to the file itself.
    """
    for name, spec in LIB_CONVERTERS.items():
        base = tmp_path / f"legacy_{name.replace('/', '_')}"
        assert spec.input_paths(base) == (base,)
