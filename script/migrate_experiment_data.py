from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment import AxesSpec
from zcu_tools.experiment.v2.singleshot.ge import GE_Exp, GE_Result
from zcu_tools.experiment.v2.singleshot.len_rabi import LenRabiExp, LenRabiResult
from zcu_tools.experiment.v2.singleshot.mist.freq import FreqDepExp, FreqResult
from zcu_tools.experiment.v2.singleshot.mist.power import PowerExp, PowerResult
from zcu_tools.experiment.v2.singleshot.mist.pre_freq import (
    PreFreqExp,
    PreFreqResult,
)
from zcu_tools.experiment.v2.twotone.ckp import CKP_Exp, CKP_Result
from zcu_tools.experiment.v2.twotone.reset.bath.length import LengthExp, LengthResult
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


@dataclass(frozen=True)
class ConverterSpec:
    convert: Callable[[Path, Path], None]
    validate: Callable[[str], object]
    validate_input: Callable[[Path], None] | None = None


def migrate_experiment_data(
    *,
    experiment: str,
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    try:
        spec = CONVERTERS[experiment]
    except KeyError:
        supported = ", ".join(sorted(CONVERTERS))
        raise ValueError(
            f"unsupported experiment {experiment!r}; supported: {supported}"
        ) from None

    if spec.validate_input is None:
        _validate_regular_input_file(input_path)
    else:
        spec.validate_input(input_path)

    output_path = Path(format_ext(str(output_path)))
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output file already exists: {output_path}; pass --overwrite to replace it"
        )
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"output directory does not exist: {output_path.parent}"
        )

    temp_path = _make_temp_path(output_path)
    try:
        spec.convert(input_path, temp_path)
        spec.validate(str(temp_path))
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return output_path


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


def _make_temp_path(output_path: Path) -> Path:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".hdf5",
        dir=output_path.parent,
    )
    os.close(fd)
    return Path(temp_name)


def _validate_regular_input_file(input_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")


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


def _validate_ge_output(path: str) -> GE_Result:
    return GE_Exp().load(path)


def _validate_bath_length_output(path: str) -> LengthResult:
    return LengthExp().load(path)


def _validate_len_rabi_output(path: str) -> LenRabiResult:
    return LenRabiExp().load(path)


def _validate_mist_power_output(path: str) -> PowerResult:
    return PowerExp().load(path)


def _validate_mist_freq_output(path: str) -> FreqResult:
    return FreqDepExp().load(path)


def _validate_mist_pre_freq_output(path: str) -> PreFreqResult:
    return PreFreqExp().load(path)


def _validate_ckp_output(path: str) -> CKP_Result:
    return CKP_Exp().load(path)


def _validate_ckp_sidecar_input(input_path: Path) -> None:
    for sidecar_path in _legacy_ckp_sidecar_paths(input_path):
        if not sidecar_path.is_file():
            raise FileNotFoundError(
                f"legacy CKP sidecar does not exist: {sidecar_path}"
            )


def _legacy_ckp_sidecar_paths(input_path: Path) -> tuple[Path, Path]:
    return (
        Path(format_ext(str(input_path.with_name(input_path.name + "_ground")))),
        Path(format_ext(str(input_path.with_name(input_path.name + "_excited")))),
    )


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


def _require_same_ckp_metadata(ground: LabberData, excited: LabberData) -> None:
    if ground.comment != excited.comment:
        raise ValueError("legacy CKP sidecars disagree on comment metadata")
    if ground.tags != excited.tags:
        raise ValueError("legacy CKP sidecars disagree on tags")


def _npz_comment(data: np.lib.npyio.NpzFile) -> str:
    if "comment" not in data.files:
        return ""
    comment = data["comment"]
    if comment.shape == ():
        return str(comment.tolist())
    return str(comment)


CONVERTERS: dict[str, ConverterSpec] = {
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
    "singleshot/mist/pre_freq": ConverterSpec(
        convert=_convert_mist_pre_freq_labber,
        validate=_validate_mist_pre_freq_output,
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
}


if __name__ == "__main__":
    raise SystemExit(main())
