from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from zcu_tools.experiment import AxesSpec
from zcu_tools.experiment.legacy_migration import (
    CONVERTERS as SINGLE_FILE_CONVERTERS,
)
from zcu_tools.experiment.legacy_migration import (
    ConverterSpec,
)
from zcu_tools.experiment.legacy_migration import (
    migrate_experiment_data as _migrate_experiment_data,
)
from zcu_tools.experiment.v2.jpa.jpa_auto_optimize import (
    JPAOptimizeResult,
    load_jpa_auto_grouped_result,
    save_jpa_auto_grouped_result,
)
from zcu_tools.experiment.v2.singleshot.ac_stark import AcStarkExp, AcStarkResult
from zcu_tools.experiment.v2.singleshot.ge import GE_Exp, GE_Result
from zcu_tools.experiment.v2.singleshot.len_rabi import LenRabiExp, LenRabiResult
from zcu_tools.experiment.v2.singleshot.mist.freq import FreqDepExp, FreqResult
from zcu_tools.experiment.v2.singleshot.mist.power import PowerExp, PowerResult
from zcu_tools.experiment.v2.singleshot.mist.power_freq import (
    FreqPowerExp,
    FreqPowerResult,
)
from zcu_tools.experiment.v2.singleshot.mist.pre_freq import (
    PreFreqExp,
    PreFreqResult,
)
from zcu_tools.experiment.v2.singleshot.t1.t1 import T1Exp, T1Result
from zcu_tools.experiment.v2.singleshot.t1.t1_with_tone import (
    T1WithToneExp,
    T1WithToneResult,
)
from zcu_tools.experiment.v2.singleshot.t1.t1_with_tone_sweep import (
    T1WithToneSweepExp,
    T1WithToneSweepResult,
)
from zcu_tools.experiment.v2.twotone.ckp import CKP_Exp, CKP_Result
from zcu_tools.experiment.v2.twotone.reset.bath.length import LengthExp, LengthResult
from zcu_tools.experiment.v2.twotone.ro_optimize.auto_optimize import (
    AutoOptResult,
    load_auto_opt_grouped_result,
    save_auto_opt_grouped_result,
)
from zcu_tools.experiment.v2.twotone.time_domain.cpmg import (
    CPMG_Result,
    load_cpmg_grouped_result,
    save_cpmg_grouped_result,
)
from zcu_tools.utils.datasaver import (
    LabberData,
    format_ext,
    load_labber_data,
    save_labber_data,
)


def migrate_experiment_data(
    *,
    experiment: str,
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    return _migrate_experiment_data(
        experiment=experiment,
        input_path=input_path,
        output_path=output_path,
        overwrite=overwrite,
        converters=CONVERTERS,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy experiment data into current HDF5 format."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=sorted(CONVERTERS),
        help="Experiment converter to run.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Legacy input file, or legacy base path for sidecar converters.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output HDF5 file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    args = parser.parse_args(argv)

    try:
        migrated = migrate_experiment_data(
            experiment=args.experiment,
            input_path=args.input,
            output_path=args.output,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(migrated)
    return 0


def _save_axes_spec_result_exact(
    output_path: Path,
    axes_spec: AxesSpec[Any, Any],
    result: object,
    *,
    comment: str = "",
    tag: str | None = None,
) -> None:
    axes = [
        (
            axis.label,
            axis.unit,
            np.asarray(getattr(result, axis.field_name)) * axis.scale,
        )
        for axis in axes_spec.axes
    ]
    z = (
        axes_spec.z.label,
        axes_spec.z.unit,
        np.asarray(getattr(result, axes_spec.z.field_name)),
    )

    requested_path = Path(format_ext(str(output_path)))
    written_path = Path(
        save_labber_data(
            str(output_path),
            z=z,
            axes=axes,
            comment=comment,
            tags=tag or axes_spec.tag,
        )
    )
    if written_path != requested_path:
        raise RuntimeError(
            f"converter wrote {written_path}, expected exact path {requested_path}"
        )


def _convert_cpmg_npz(input_path: Path, output_path: Path) -> None:
    with np.load(input_path) as data:
        missing = {"times", "lengths", "signals2D"} - set(data.files)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"legacy CPMG npz is missing required arrays: {missing_text}"
            )

        result = CPMG_Result(
            ns=np.asarray(data["times"], dtype=np.int64),
            delays=np.asarray(data["lengths"], dtype=np.float64),
            signals=np.asarray(data["signals2D"], dtype=np.complex128),
        )
        comment = _npz_comment(data)

    save_cpmg_grouped_result(str(output_path), result, comment=comment)


def _convert_ckp_sidecars(input_path: Path, output_path: Path) -> None:
    result, comment = _load_legacy_ckp_sidecars(input_path)
    axes_spec = CKP_Exp.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError("CKP_Exp has no AXES_SPEC")

    _save_axes_spec_result_exact(output_path, axes_spec, result, comment=comment)


def _convert_ro_auto_sidecars(input_path: Path, output_path: Path) -> None:
    result, comment = _load_legacy_ro_auto_sidecars(input_path)
    save_auto_opt_grouped_result(str(output_path), result, comment=comment)


def _convert_jpa_auto_sidecars(input_path: Path, output_path: Path) -> None:
    result, comment = _load_legacy_jpa_auto_sidecars(input_path)
    save_jpa_auto_grouped_result(str(output_path), result, comment=comment)


def _convert_ge_labber(input_path: Path, output_path: Path) -> None:
    data = load_labber_data(str(input_path))
    shot_values, prepared_values = _legacy_two_axes(
        data,
        input_path,
        expected=(("shot", "point"), ("ge", "")),
    )
    _require_axis_values(
        "prepared state",
        prepared_values,
        np.array([0, 1], dtype=np.int64),
        path=input_path,
    )
    signals = _legacy_z(
        data,
        input_path,
        expected_shape=(len(prepared_values), len(shot_values)),
    )

    result = GE_Result(
        signals=signals,
        shot_indices=shot_values.astype(np.int64),
        prepared_states=prepared_values.astype(np.int64),
    )
    axes_spec = GE_Exp.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError("GE_Exp has no AXES_SPEC")

    _save_axes_spec_result_exact(output_path, axes_spec, result, comment=data.comment)


def _convert_bath_length_labber(input_path: Path, output_path: Path) -> None:
    data = load_labber_data(str(input_path))
    length_seconds, phases = _legacy_two_axes(
        data,
        input_path,
        expected=(("Length", "s"), ("Pi2 Phase", "deg")),
    )
    _require_axis_values(
        "phase",
        phases,
        np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64),
        path=input_path,
    )
    legacy_signals = _legacy_z(
        data,
        input_path,
        expected_shape=(len(phases), len(length_seconds)),
    )

    result = LengthResult(
        lengths=(length_seconds * 1e6).astype(np.float64),
        phases=phases.astype(np.float64),
        signals=legacy_signals.T.astype(np.complex128),
    )
    axes_spec = LengthExp.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError("LengthExp has no AXES_SPEC")

    _save_axes_spec_result_exact(output_path, axes_spec, result, comment=data.comment)


def _convert_len_rabi_labber(input_path: Path, output_path: Path) -> None:
    data = load_labber_data(str(input_path))
    length_seconds, population_values = _legacy_two_axes(
        data,
        input_path,
        expected=(("Length", "s"), ("GE population", "a.u.")),
    )
    _require_population_axis(population_values, input_path)
    legacy_populations = _legacy_population_z(
        data,
        input_path,
        expected_shape=(len(population_values), len(length_seconds)),
    )

    result = LenRabiResult(
        lengths=(length_seconds * 1e6).astype(np.float64),
        signals=legacy_populations.T.astype(np.float64),
        population_states=population_values.astype(np.int64),
    )
    axes_spec = LenRabiExp.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError("LenRabiExp has no AXES_SPEC")

    _save_axes_spec_result_exact(output_path, axes_spec, result, comment=data.comment)


def _convert_ac_stark_sidecars(input_path: Path, output_path: Path) -> None:
    ground_path, excited_path = _legacy_sidecar_paths_by_name(
        input_path, ("_g_pop", "_e_pop")
    )
    ground = load_labber_data(str(ground_path))
    excited = load_labber_data(str(excited_path))
    _require_same_metadata(ground, (excited,), "legacy AC Stark sidecars")

    gains, freq_hz = _legacy_axes(
        ground,
        ground_path,
        expected=(("Stark Pulse Gain", "a.u."), ("Frequency", "Hz")),
    )
    excited_gains, excited_freq_hz = _legacy_axes(
        excited,
        excited_path,
        expected=(("Stark Pulse Gain", "a.u."), ("Frequency", "Hz")),
    )
    _require_same_axis_values_for_context(
        "Stark Pulse Gain", gains, excited_gains, "legacy AC Stark sidecars"
    )
    _require_same_axis_values_for_context(
        "Frequency", freq_hz, excited_freq_hz, "legacy AC Stark sidecars"
    )

    expected_shape = (len(freq_hz), len(gains))
    ground_pop = _legacy_real_z(
        ground,
        ground_path,
        expected_label="Signal",
        expected_unit="a.u.",
        expected_shape=expected_shape,
    )
    excited_pop = _legacy_real_z(
        excited,
        excited_path,
        expected_label="Signal",
        expected_unit="a.u.",
        expected_shape=expected_shape,
    )

    result = AcStarkResult(
        gains=gains.astype(np.float64),
        freqs=(freq_hz * 1e-6).astype(np.float64),
        populations=np.stack([ground_pop.T, excited_pop.T], axis=-1),
        population_states=np.array([0, 1], dtype=np.int64),
    )
    axes_spec = AcStarkExp.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError("AcStarkExp has no AXES_SPEC")
    _save_axes_spec_result_exact(output_path, axes_spec, result, comment=ground.comment)


def _convert_mist_power_freq_sidecars(input_path: Path, output_path: Path) -> None:
    ground_path, excited_path = _legacy_sidecar_paths_exact_by_name(
        input_path, ("_g_population", "_e_population")
    )
    ground = load_labber_data(str(ground_path))
    excited = load_labber_data(str(excited_path))
    _require_same_metadata(ground, (excited,), "legacy MIST power-freq sidecars")

    gains, freq_hz = _legacy_axes(
        ground,
        ground_path,
        expected=(("Drive gain", "a.u."), ("Drive freq", "Hz")),
    )
    excited_gains, excited_freq_hz = _legacy_axes(
        excited,
        excited_path,
        expected=(("Drive gain", "a.u."), ("Drive freq", "Hz")),
    )
    _require_same_axis_values_for_context(
        "Drive gain", gains, excited_gains, "legacy MIST power-freq sidecars"
    )
    _require_same_axis_values_for_context(
        "Drive freq", freq_hz, excited_freq_hz, "legacy MIST power-freq sidecars"
    )

    expected_shape = (len(freq_hz), len(gains))
    ground_pop = _legacy_real_z(
        ground,
        ground_path,
        expected_label="Population",
        expected_unit="a.u.",
        expected_shape=expected_shape,
    )
    excited_pop = _legacy_real_z(
        excited,
        excited_path,
        expected_label="Population",
        expected_unit="a.u.",
        expected_shape=expected_shape,
    )

    result = FreqPowerResult(
        gains=gains.astype(np.float64),
        freqs=(freq_hz * 1e-6).astype(np.float64),
        signals=np.stack([ground_pop.T, excited_pop.T], axis=-1),
        population_states=np.array([0, 1], dtype=np.int64),
    )
    axes_spec = FreqPowerExp.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError("FreqPowerExp has no AXES_SPEC")
    _save_axes_spec_result_exact(output_path, axes_spec, result, comment=ground.comment)


def _convert_mist_power_labber(input_path: Path, output_path: Path) -> None:
    data = load_labber_data(str(input_path))
    gains, population_values = _legacy_two_axes(
        data,
        input_path,
        expected=(("Drive gain", "a.u."), ("GE population", "a.u.")),
    )
    _require_population_axis(population_values, input_path)
    legacy_populations = _legacy_population_z(
        data,
        input_path,
        expected_shape=(len(population_values), len(gains)),
    )

    result = PowerResult(
        gains=gains.astype(np.float64),
        signals=legacy_populations.T.astype(np.float64),
        population_states=population_values.astype(np.int64),
    )
    axes_spec = PowerExp.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError("PowerExp has no AXES_SPEC")

    _save_axes_spec_result_exact(output_path, axes_spec, result, comment=data.comment)


def _convert_mist_freq_labber(input_path: Path, output_path: Path) -> None:
    data = load_labber_data(str(input_path))
    freq_hz, population_values = _legacy_two_axes(
        data,
        input_path,
        expected=(("Drive Freq", "Hz"), ("GE population", "a.u.")),
    )
    _require_population_axis(population_values, input_path)
    legacy_populations = _legacy_population_z(
        data,
        input_path,
        expected_shape=(len(population_values), len(freq_hz)),
    )

    result = FreqResult(
        freqs=(freq_hz * 1e-6).astype(np.float64),
        signals=legacy_populations.T.astype(np.float64),
        population_states=population_values.astype(np.int64),
    )
    axes_spec = FreqDepExp.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError("FreqDepExp has no AXES_SPEC")

    _save_axes_spec_result_exact(output_path, axes_spec, result, comment=data.comment)


def _convert_mist_pre_freq_labber(input_path: Path, output_path: Path) -> None:
    data = load_labber_data(str(input_path))
    freq_hz, population_values = _legacy_two_axes(
        data,
        input_path,
        expected=(("PrePulse frequency", "Hz"), ("GE population", "a.u.")),
    )
    _require_population_axis(population_values, input_path)
    legacy_populations = _legacy_population_z(
        data,
        input_path,
        expected_shape=(len(population_values), len(freq_hz)),
    )

    result = PreFreqResult(
        freqs=(freq_hz * 1e-6).astype(np.float64),
        signals=legacy_populations.T.astype(np.float64),
        population_states=population_values.astype(np.int64),
    )
    axes_spec = PreFreqExp.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError("PreFreqExp has no AXES_SPEC")

    _save_axes_spec_result_exact(output_path, axes_spec, result, comment=data.comment)


def _convert_t1_sidecars(input_path: Path, output_path: Path) -> None:
    result, comment = _load_legacy_t1_sidecars(
        input_path,
        result_type=T1Result,
        context="legacy T1 sidecars",
    )
    axes_spec = T1Exp.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError("T1Exp has no AXES_SPEC")
    _save_axes_spec_result_exact(output_path, axes_spec, result, comment=comment)


def _convert_t1_with_tone_sidecars(input_path: Path, output_path: Path) -> None:
    result, comment = _load_legacy_t1_sidecars(
        input_path,
        result_type=T1WithToneResult,
        context="legacy T1-with-tone sidecars",
    )
    axes_spec = T1WithToneExp.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError("T1WithToneExp has no AXES_SPEC")
    _save_axes_spec_result_exact(output_path, axes_spec, result, comment=comment)


def _convert_t1_with_tone_sweep_sidecars(input_path: Path, output_path: Path) -> None:
    paths = _legacy_sidecar_paths_by_name(
        input_path, ("_gg_pop", "_ge_pop", "_eg_pop", "_ee_pop")
    )
    gg_path, ge_path, eg_path, ee_path = paths
    gg, ge, eg, ee = tuple(load_labber_data(str(path)) for path in paths)
    _require_same_metadata(gg, (ge, eg, ee), "legacy T1-with-tone-sweep sidecars")

    time_s, xs = _legacy_axes(
        gg,
        gg_path,
        expected=(("Time", "s"), ("sweep value", "a.u.")),
    )
    for path, data in ((ge_path, ge), (eg_path, eg), (ee_path, ee)):
        candidate_time_s, candidate_xs = _legacy_axes(
            data,
            path,
            expected=(("Time", "s"), ("sweep value", "a.u.")),
        )
        _require_same_axis_values_for_context(
            "Time", time_s, candidate_time_s, "legacy T1-with-tone-sweep sidecars"
        )
        _require_same_axis_values_for_context(
            "sweep value", xs, candidate_xs, "legacy T1-with-tone-sweep sidecars"
        )

    expected_shape = (len(xs), len(time_s))
    gg_pop = _legacy_real_z(
        gg,
        gg_path,
        expected_label="Ground Populations",
        expected_unit="a.u.",
        expected_shape=expected_shape,
    )
    ge_pop = _legacy_real_z(
        ge,
        ge_path,
        expected_label="Ground Populations",
        expected_unit="a.u.",
        expected_shape=expected_shape,
    )
    eg_pop = _legacy_real_z(
        eg,
        eg_path,
        expected_label="Ground Populations",
        expected_unit="a.u.",
        expected_shape=expected_shape,
    )
    ee_pop = _legacy_real_z(
        ee,
        ee_path,
        expected_label="Ground Populations",
        expected_unit="a.u.",
        expected_shape=expected_shape,
    )

    signals = np.empty((len(xs), 2, len(time_s), 2), dtype=np.float64)
    signals[:, 0, :, 0] = gg_pop
    signals[:, 0, :, 1] = ge_pop
    signals[:, 1, :, 0] = eg_pop
    signals[:, 1, :, 1] = ee_pop
    result = T1WithToneSweepResult(
        xs=xs.astype(np.float64),
        lengths=(time_s * 1e6).astype(np.float64),
        signals=signals,
        initial_states=np.array([0, 1], dtype=np.int64),
        population_states=np.array([0, 1], dtype=np.int64),
    )
    axes_spec = T1WithToneSweepExp.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError("T1WithToneSweepExp has no AXES_SPEC")
    _save_axes_spec_result_exact(output_path, axes_spec, result, comment=gg.comment)


def _validate_ge_output(path: str) -> GE_Result:
    return GE_Exp().load(path)


def _validate_bath_length_output(path: str) -> LengthResult:
    return LengthExp().load(path)


def _validate_len_rabi_output(path: str) -> LenRabiResult:
    return LenRabiExp().load(path)


def _validate_ac_stark_output(path: str) -> AcStarkResult:
    return AcStarkExp().load(path)


def _validate_mist_power_freq_output(path: str) -> FreqPowerResult:
    return FreqPowerExp().load(path)


def _validate_mist_power_output(path: str) -> PowerResult:
    return PowerExp().load(path)


def _validate_mist_freq_output(path: str) -> FreqResult:
    return FreqDepExp().load(path)


def _validate_mist_pre_freq_output(path: str) -> PreFreqResult:
    return PreFreqExp().load(path)


def _validate_ckp_output(path: str) -> CKP_Result:
    return CKP_Exp().load(path)


def _validate_ro_auto_output(path: str) -> AutoOptResult:
    return load_auto_opt_grouped_result(path)


def _validate_jpa_auto_output(path: str) -> JPAOptimizeResult:
    return load_jpa_auto_grouped_result(path)


def _validate_t1_output(path: str) -> T1Result:
    return T1Exp().load(path)


def _validate_t1_with_tone_output(path: str) -> T1WithToneResult:
    return T1WithToneExp().load(path)


def _validate_t1_with_tone_sweep_output(path: str) -> T1WithToneSweepResult:
    return T1WithToneSweepExp().load(path)


def _validate_ac_stark_sidecar_input(input_path: Path) -> None:
    _validate_sidecar_input(
        _legacy_sidecar_paths_by_name(input_path, ("_g_pop", "_e_pop")),
        "legacy AC Stark sidecar",
    )


def _validate_mist_power_freq_sidecar_input(input_path: Path) -> None:
    _validate_sidecar_input(
        _legacy_sidecar_paths_exact_by_name(
            input_path, ("_g_population", "_e_population")
        ),
        "legacy MIST power-freq sidecar",
    )


def _validate_t1_sidecar_input(input_path: Path) -> None:
    _validate_sidecar_input(
        _legacy_sidecar_paths_by_stem(input_path, ("_initg", "_inite")),
        "legacy T1 sidecar",
    )


def _validate_t1_with_tone_sweep_sidecar_input(input_path: Path) -> None:
    _validate_sidecar_input(
        _legacy_sidecar_paths_by_name(
            input_path, ("_gg_pop", "_ge_pop", "_eg_pop", "_ee_pop")
        ),
        "legacy T1-with-tone-sweep sidecar",
    )


def _validate_ckp_sidecar_input(input_path: Path) -> None:
    for sidecar_path in _legacy_ckp_sidecar_paths(input_path):
        if not sidecar_path.is_file():
            raise FileNotFoundError(
                f"legacy CKP sidecar does not exist: {sidecar_path}"
            )


def _validate_ro_auto_sidecar_input(input_path: Path) -> None:
    _validate_sidecar_input(
        _legacy_sidecar_paths_by_name(input_path, ("_params", "_signals")),
        "legacy RO auto-optimize sidecar",
    )


def _validate_jpa_auto_sidecar_input(input_path: Path) -> None:
    _validate_sidecar_input(
        _legacy_sidecar_paths_by_name(input_path, ("_params", "_phases", "_signals")),
        "legacy JPA auto-optimize sidecar",
    )


def _legacy_ckp_sidecar_paths(input_path: Path) -> tuple[Path, Path]:
    return (
        Path(format_ext(str(input_path.with_name(input_path.name + "_ground")))),
        Path(format_ext(str(input_path.with_name(input_path.name + "_excited")))),
    )


def _legacy_sidecar_paths_by_name(
    input_path: Path, suffixes: tuple[str, ...]
) -> tuple[Path, ...]:
    return tuple(
        _resolve_legacy_sidecar_path(input_path.with_name(input_path.name + suffix))
        for suffix in suffixes
    )


def _legacy_sidecar_paths_exact_by_name(
    input_path: Path, suffixes: tuple[str, ...]
) -> tuple[Path, ...]:
    return tuple(
        Path(format_ext(str(input_path.with_name(input_path.name + suffix))))
        for suffix in suffixes
    )


def _legacy_sidecar_paths_by_stem(
    input_path: Path, suffixes: tuple[str, ...]
) -> tuple[Path, ...]:
    return tuple(
        _resolve_legacy_sidecar_path(input_path.with_name(input_path.stem + suffix))
        for suffix in suffixes
    )


def _resolve_legacy_sidecar_path(sidecar_base: Path) -> Path:
    exact = Path(format_ext(str(sidecar_base)))
    candidates: list[Path] = []
    if exact.is_file():
        candidates.append(exact)

    numbered_prefix = exact.stem + "_"
    if exact.parent.exists():
        for child in exact.parent.iterdir():
            suffix_number = child.stem.removeprefix(numbered_prefix)
            if (
                child.is_file()
                and child.suffix == exact.suffix
                and child.stem.startswith(numbered_prefix)
                and suffix_number.isdigit()
            ):
                candidates.append(child)

    candidates = sorted(set(candidates))
    if len(candidates) > 1:
        formatted = ", ".join(str(path) for path in candidates)
        raise ValueError(f"ambiguous legacy sidecar for {exact}: {formatted}")
    if candidates:
        return candidates[0]
    return exact


def _validate_sidecar_input(paths: tuple[Path, ...], label: str) -> None:
    for sidecar_path in paths:
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {sidecar_path}")


def _load_legacy_ckp_sidecars(input_path: Path) -> tuple[CKP_Result, str]:
    ground_path, excited_path = _legacy_ckp_sidecar_paths(input_path)
    ground = load_labber_data(str(ground_path))
    excited = load_labber_data(str(excited_path))
    _require_same_ckp_metadata(ground, excited)

    ground_res_hz, ground_qub_hz = _legacy_ckp_axes(ground, ground_path)
    excited_res_hz, excited_qub_hz = _legacy_ckp_axes(excited, excited_path)
    _require_same_axis_values("Resonator Frequency", ground_res_hz, excited_res_hz)
    _require_same_axis_values("Qubit Frequency", ground_qub_hz, excited_qub_hz)

    expected_shape = (len(ground_qub_hz), len(ground_res_hz))
    ground_signals = _legacy_ckp_z(ground, ground_path, expected_shape)
    excited_signals = _legacy_ckp_z(excited, excited_path, expected_shape)

    result = CKP_Result(
        res_freqs=(ground_res_hz * 1e-6).astype(np.float64),
        qub_freqs=(ground_qub_hz * 1e-6).astype(np.float64),
        signals=np.stack([ground_signals.T, excited_signals.T], axis=0).astype(
            np.complex128
        ),
    )
    return result, ground.comment


def _load_legacy_ro_auto_sidecars(input_path: Path) -> tuple[AutoOptResult, str]:
    params_path, signals_path = _legacy_sidecar_paths_by_name(
        input_path, ("_params", "_signals")
    )
    params_data = load_labber_data(str(params_path))
    signals_data = load_labber_data(str(signals_path))
    _require_same_comment_metadata(
        params_data,
        (signals_data,),
        "legacy RO auto-optimize sidecars",
    )

    iterations, params = _legacy_auto_params(
        params_data,
        params_path,
        context="legacy RO auto-optimize params sidecar",
    )
    signals = _legacy_auto_1d_real_z(
        signals_data,
        signals_path,
        iterations,
        expected_label="Signal",
        expected_unit="a.u.",
        context="legacy RO auto-optimize signals sidecar",
    )
    return AutoOptResult(params=params, signals=signals), params_data.comment


def _load_legacy_jpa_auto_sidecars(input_path: Path) -> tuple[JPAOptimizeResult, str]:
    params_path, phases_path, signals_path = _legacy_sidecar_paths_by_name(
        input_path, ("_params", "_phases", "_signals")
    )
    params_data = load_labber_data(str(params_path))
    phases_data = load_labber_data(str(phases_path))
    signals_data = load_labber_data(str(signals_path))
    _require_same_comment_metadata(
        params_data,
        (phases_data, signals_data),
        "legacy JPA auto-optimize sidecars",
    )

    iterations, params = _legacy_auto_params(
        params_data,
        params_path,
        context="legacy JPA auto-optimize params sidecar",
    )
    phases_real = _legacy_auto_1d_real_z(
        phases_data,
        phases_path,
        iterations,
        expected_label="Phase",
        expected_unit="a.u.",
        context="legacy JPA auto-optimize phases sidecar",
    )
    phases = _require_integer_like_values(
        phases_real,
        "legacy JPA auto-optimize phase",
        phases_path,
    ).astype(np.int32)
    signals = _legacy_auto_1d_real_z(
        signals_data,
        signals_path,
        iterations,
        expected_label="Signal",
        expected_unit="a.u.",
        context="legacy JPA auto-optimize signals sidecar",
    )
    return (
        JPAOptimizeResult(params=params, phases=phases, signals=signals),
        params_data.comment,
    )


def _legacy_auto_params(
    data: LabberData,
    path: Path,
    *,
    context: str,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    iteration_values, parameter_type_values = _legacy_axes(
        data,
        path,
        expected=(("Iteration", "a.u."), ("Parameter Type", "a.u.")),
    )
    iterations = _require_iteration_axis_values(iteration_values, path, context)
    _require_axis_values(
        "parameter type",
        parameter_type_values,
        np.array([0.0, 1.0, 2.0], dtype=np.float64),
        path=path,
    )
    z = _legacy_real_z(
        data,
        path,
        expected_label="Parameters",
        expected_unit="a.u.",
        expected_shape=(3, len(iterations)),
    )
    return iterations, z.T.astype(np.float64)


def _legacy_auto_1d_real_z(
    data: LabberData,
    path: Path,
    iterations: NDArray[np.int64],
    *,
    expected_label: str,
    expected_unit: str,
    context: str,
) -> NDArray[np.float64]:
    (iteration_values,) = _legacy_axes(
        data,
        path,
        expected=(("Iteration", "a.u."),),
    )
    candidate_iterations = _require_iteration_axis_values(
        iteration_values,
        path,
        context,
    )
    if not np.array_equal(candidate_iterations, iterations):
        raise ValueError(f"{context} disagrees on Iteration axis values")
    return _legacy_real_z(
        data,
        path,
        expected_label=expected_label,
        expected_unit=expected_unit,
        expected_shape=(len(iterations),),
    )


def _load_legacy_t1_sidecars(
    input_path: Path,
    *,
    result_type: type[T1Result] | type[T1WithToneResult],
    context: str,
) -> tuple[T1Result | T1WithToneResult, str]:
    initg_path, inite_path = _legacy_sidecar_paths_by_stem(
        input_path, ("_initg", "_inite")
    )
    initg = load_labber_data(str(initg_path))
    inite = load_labber_data(str(inite_path))
    _require_same_metadata(initg, (inite,), context)

    time_s, population_values = _legacy_axes(
        initg,
        initg_path,
        expected=(("Time", "s"), ("GE population", "a.u.")),
    )
    inite_time_s, inite_population_values = _legacy_axes(
        inite,
        inite_path,
        expected=(("Time", "s"), ("GE population", "a.u.")),
    )
    _require_population_axis(population_values, initg_path)
    _require_population_axis(inite_population_values, inite_path)
    _require_same_axis_values_for_context("Time", time_s, inite_time_s, context)
    _require_same_axis_values_for_context(
        "GE population", population_values, inite_population_values, context
    )

    expected_shape = (len(population_values), len(time_s))
    initg_pop = _legacy_real_z(
        initg,
        initg_path,
        expected_label="Signal",
        expected_unit="a.u.",
        expected_shape=expected_shape,
    )
    inite_pop = _legacy_real_z(
        inite,
        inite_path,
        expected_label="Signal",
        expected_unit="a.u.",
        expected_shape=expected_shape,
    )
    signals = np.stack([initg_pop.T, inite_pop.T], axis=1)
    result = result_type(
        lengths=(time_s * 1e6).astype(np.float64),
        signals=signals,
        initial_states=np.array([0, 1], dtype=np.int64),
        population_states=population_values.astype(np.int64),
    )
    return result, initg.comment


def _legacy_axes(
    data: LabberData,
    path: Path,
    *,
    expected: tuple[tuple[str, str], ...],
) -> tuple[NDArray[np.float64], ...]:
    if len(data.axes) != len(expected):
        raise ValueError(
            f"legacy file {path} has {len(data.axes)} axes; expected {len(expected)}"
        )

    values: list[NDArray[np.float64]] = []
    for index, (axis, (expected_name, expected_unit)) in enumerate(
        zip(data.axes, expected, strict=True)
    ):
        if axis.name != expected_name or axis.unit != expected_unit:
            raise ValueError(
                f"legacy file {path} axis {index} is "
                f"{axis.name!r} [{axis.unit!r}], expected "
                f"{expected_name!r} [{expected_unit!r}]"
            )
        axis_values = np.asarray(axis.values, dtype=np.float64)
        if axis_values.ndim != 1:
            raise ValueError(
                f"legacy file {path} axis {expected_name!r} is "
                f"{axis_values.ndim}D; expected 1D"
            )
        values.append(axis_values)

    return tuple(values)


def _legacy_ckp_axes(
    data: LabberData, path: Path
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    expected = (
        ("Resonator Frequency", "Hz"),
        ("Qubit Frequency", "Hz"),
    )
    if len(data.axes) != len(expected):
        raise ValueError(
            f"legacy CKP sidecar {path} has {len(data.axes)} axes; "
            f"expected {len(expected)}"
        )

    values: list[NDArray[np.float64]] = []
    for index, (axis, (expected_name, expected_unit)) in enumerate(
        zip(data.axes, expected, strict=True)
    ):
        if axis.name != expected_name or axis.unit != expected_unit:
            raise ValueError(
                f"legacy CKP sidecar {path} axis {index} is "
                f"{axis.name!r} [{axis.unit!r}], expected "
                f"{expected_name!r} [{expected_unit!r}]"
            )
        axis_values = np.asarray(axis.values, dtype=np.float64)
        if axis_values.ndim != 1:
            raise ValueError(
                f"legacy CKP sidecar {path} axis {expected_name!r} is "
                f"{axis_values.ndim}D; expected 1D"
            )
        values.append(axis_values)

    return values[0], values[1]


def _legacy_ckp_z(
    data: LabberData,
    path: Path,
    expected_shape: tuple[int, int],
) -> NDArray[np.complex128]:
    if data.data.name != "Signal" or data.data.unit != "a.u.":
        raise ValueError(
            f"legacy CKP sidecar {path} z channel is "
            f"{data.data.name!r} [{data.data.unit!r}], expected 'Signal' ['a.u.']"
        )

    signals = np.asarray(data.z, dtype=np.complex128)
    if signals.shape != expected_shape:
        raise ValueError(
            f"legacy CKP sidecar {path} z shape {signals.shape} != "
            f"expected legacy shape {expected_shape}"
        )
    return signals


def _legacy_two_axes(
    data: LabberData,
    path: Path,
    *,
    expected: tuple[tuple[str, str], tuple[str, str]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if len(data.axes) != len(expected):
        raise ValueError(
            f"legacy file {path} has {len(data.axes)} axes; expected {len(expected)}"
        )

    values: list[NDArray[np.float64]] = []
    for index, (axis, (expected_name, expected_unit)) in enumerate(
        zip(data.axes, expected, strict=True)
    ):
        if axis.name != expected_name or axis.unit != expected_unit:
            raise ValueError(
                f"legacy file {path} axis {index} is "
                f"{axis.name!r} [{axis.unit!r}], expected "
                f"{expected_name!r} [{expected_unit!r}]"
            )
        axis_values = np.asarray(axis.values, dtype=np.float64)
        if axis_values.ndim != 1:
            raise ValueError(
                f"legacy file {path} axis {expected_name!r} is "
                f"{axis_values.ndim}D; expected 1D"
            )
        values.append(axis_values)

    return values[0], values[1]


def _legacy_z(
    data: LabberData,
    path: Path,
    *,
    expected_shape: tuple[int, ...],
) -> NDArray[np.complex128]:
    if data.data.name != "Signal" or data.data.unit != "a.u.":
        raise ValueError(
            f"legacy file {path} z channel is "
            f"{data.data.name!r} [{data.data.unit!r}], expected 'Signal' ['a.u.']"
        )

    signals = np.asarray(data.z, dtype=np.complex128)
    if signals.shape != expected_shape:
        raise ValueError(
            f"legacy file {path} z shape {signals.shape} != "
            f"expected legacy shape {expected_shape}"
        )
    return signals


def _legacy_real_z(
    data: LabberData,
    path: Path,
    *,
    expected_label: str,
    expected_unit: str,
    expected_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    if data.data.name != expected_label or data.data.unit != expected_unit:
        raise ValueError(
            f"legacy file {path} z channel is "
            f"{data.data.name!r} [{data.data.unit!r}], expected "
            f"{expected_label!r} [{expected_unit!r}]"
        )

    values = np.asarray(data.z, dtype=np.complex128)
    if values.shape != expected_shape:
        raise ValueError(
            f"legacy file {path} z shape {values.shape} != "
            f"expected legacy shape {expected_shape}"
        )
    if np.any(values.imag != 0.0):
        raise ValueError(f"legacy file {path} z contains imaginary components")
    return values.real.astype(np.float64)


def _legacy_population_z(
    data: LabberData,
    path: Path,
    *,
    expected_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    if data.data.name != "Population" or data.data.unit != "a.u.":
        raise ValueError(
            f"legacy file {path} z channel is "
            f"{data.data.name!r} [{data.data.unit!r}], expected 'Population' ['a.u.']"
        )

    populations = np.asarray(data.z, dtype=np.complex128)
    if populations.shape != expected_shape:
        raise ValueError(
            f"legacy file {path} z shape {populations.shape} != "
            f"expected legacy shape {expected_shape}"
        )
    if np.any(populations.imag != 0.0):
        raise ValueError(
            f"legacy file {path} population z contains imaginary components"
        )
    return populations.real.astype(np.float64)


def _require_population_axis(values: NDArray[np.float64], path: Path) -> None:
    _require_axis_values(
        "population state",
        values,
        np.array([0, 1], dtype=np.float64),
        path=path,
    )


def _require_iteration_axis_values(
    values: NDArray[np.float64],
    path: Path,
    context: str,
) -> NDArray[np.int64]:
    rounded = _require_integer_like_values(values, f"{context} Iteration axis", path)
    expected = np.arange(len(rounded), dtype=np.int64)
    if not np.array_equal(rounded, expected):
        raise ValueError(
            f"{context} Iteration axis values {rounded.tolist()} != "
            f"expected {expected.tolist()}"
        )
    return rounded


def _require_integer_like_values(
    values: NDArray[np.float64],
    label: str,
    path: Path,
) -> NDArray[np.int64]:
    rounded = np.round(values)
    if not np.allclose(values, rounded):
        raise ValueError(f"legacy file {path} {label} values must be integers")
    return rounded.astype(np.int64)


def _require_axis_values(
    axis_name: str,
    values: NDArray[np.float64],
    expected: NDArray[Any],
    *,
    path: Path,
) -> None:
    if values.shape != expected.shape or not np.array_equal(values, expected):
        raise ValueError(
            f"legacy file {path} {axis_name} values {values.tolist()} != "
            f"expected {expected.tolist()}"
        )


def _require_same_axis_values(
    axis_name: str,
    ground_values: NDArray[np.float64],
    excited_values: NDArray[np.float64],
) -> None:
    if ground_values.shape != excited_values.shape or not np.array_equal(
        ground_values, excited_values
    ):
        raise ValueError(f"legacy CKP sidecars disagree on {axis_name} axis values")


def _require_same_axis_values_for_context(
    axis_name: str,
    reference_values: NDArray[np.float64],
    candidate_values: NDArray[np.float64],
    context: str,
) -> None:
    if reference_values.shape != candidate_values.shape or not np.array_equal(
        reference_values, candidate_values
    ):
        raise ValueError(f"{context} disagree on {axis_name} axis values")


def _require_same_metadata(
    reference: LabberData, others: Sequence[LabberData], context: str
) -> None:
    for other in others:
        if reference.comment != other.comment:
            raise ValueError(f"{context} disagree on comment metadata")
        if reference.tags != other.tags:
            raise ValueError(f"{context} disagree on tags")


def _require_same_comment_metadata(
    reference: LabberData, others: Sequence[LabberData], context: str
) -> None:
    for other in others:
        if reference.comment != other.comment:
            raise ValueError(f"{context} disagree on comment metadata")


def _require_same_ckp_metadata(ground: LabberData, excited: LabberData) -> None:
    if ground.comment != excited.comment:
        raise ValueError("legacy CKP sidecars disagree on comment metadata")
    if ground.tags != excited.tags:
        raise ValueError("legacy CKP sidecars disagree on tags")


def _npz_comment(data: NpzFile) -> str:
    if "comment" not in data.files:
        return ""
    comment = data["comment"]
    if comment.shape == ():
        return str(comment.tolist())
    return str(comment)


CONVERTERS: dict[str, ConverterSpec] = {
    **SINGLE_FILE_CONVERTERS,
    "singleshot/ac_stark": ConverterSpec(
        convert=_convert_ac_stark_sidecars,
        validate=_validate_ac_stark_output,
        validate_input=_validate_ac_stark_sidecar_input,
    ),
    "singleshot/ge": ConverterSpec(
        convert=_convert_ge_labber,
        validate=_validate_ge_output,
    ),
    "singleshot/len_rabi": ConverterSpec(
        convert=_convert_len_rabi_labber,
        validate=_validate_len_rabi_output,
    ),
    "singleshot/mist/freq": ConverterSpec(
        convert=_convert_mist_freq_labber,
        validate=_validate_mist_freq_output,
    ),
    "singleshot/mist/power": ConverterSpec(
        convert=_convert_mist_power_labber,
        validate=_validate_mist_power_output,
    ),
    "singleshot/mist/power_freq": ConverterSpec(
        convert=_convert_mist_power_freq_sidecars,
        validate=_validate_mist_power_freq_output,
        validate_input=_validate_mist_power_freq_sidecar_input,
    ),
    "singleshot/mist/pre_freq": ConverterSpec(
        convert=_convert_mist_pre_freq_labber,
        validate=_validate_mist_pre_freq_output,
    ),
    "singleshot/t1/t1": ConverterSpec(
        convert=_convert_t1_sidecars,
        validate=_validate_t1_output,
        validate_input=_validate_t1_sidecar_input,
    ),
    "singleshot/t1/t1_with_tone": ConverterSpec(
        convert=_convert_t1_with_tone_sidecars,
        validate=_validate_t1_with_tone_output,
        validate_input=_validate_t1_sidecar_input,
    ),
    "singleshot/t1/t1_with_tone_sweep": ConverterSpec(
        convert=_convert_t1_with_tone_sweep_sidecars,
        validate=_validate_t1_with_tone_sweep_output,
        validate_input=_validate_t1_with_tone_sweep_sidecar_input,
    ),
    "twotone/reset/bath/length": ConverterSpec(
        convert=_convert_bath_length_labber,
        validate=_validate_bath_length_output,
    ),
    "twotone/cpmg": ConverterSpec(
        convert=_convert_cpmg_npz,
        validate=load_cpmg_grouped_result,
    ),
    "twotone/ckp": ConverterSpec(
        convert=_convert_ckp_sidecars,
        validate=_validate_ckp_output,
        validate_input=_validate_ckp_sidecar_input,
    ),
    "twotone/ro_optimize/auto_optimize": ConverterSpec(
        convert=_convert_ro_auto_sidecars,
        validate=_validate_ro_auto_output,
        validate_input=_validate_ro_auto_sidecar_input,
    ),
    "jpa/jpa_auto_optimize": ConverterSpec(
        convert=_convert_jpa_auto_sidecars,
        validate=_validate_jpa_auto_output,
        validate_input=_validate_jpa_auto_sidecar_input,
    ),
}


if __name__ == "__main__":
    raise SystemExit(main())
