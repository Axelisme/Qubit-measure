from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment.axes_spec import AxesSpec
from zcu_tools.experiment.v2.onetone.flux_dep import (
    FluxDepExp as OneToneFluxDepExp,
)
from zcu_tools.experiment.v2.onetone.freq import FreqExp as OneToneFreqExp
from zcu_tools.experiment.v2.twotone.fluxdep import FreqFluxExp
from zcu_tools.experiment.v2.twotone.freq import FreqExp as TwoToneFreqExp
from zcu_tools.utils.datasaver import (
    LabberData,
    format_ext,
    load_labber_data,
    save_labber_data,
)

__all__ = [
    "CONVERTERS",
    "ConverterSpec",
    "migrate_experiment_data",
    "migrated_legacy_tempfile",
    "save_axes_spec_result_exact",
]


@dataclass(frozen=True)
class ConverterSpec:
    convert: Callable[[Path, Path], None]
    validate: Callable[[str], object]
    validate_input: Callable[[Path], None] | None = None


@dataclass(frozen=True)
class LegacyAxisSpec:
    field_name: str
    labels: tuple[str, ...]
    unit_to_memory_scale: Mapping[str, float]
    dtype: type = np.float64


@dataclass(frozen=True)
class LegacyZSpec:
    labels: tuple[str, ...]
    units: tuple[str, ...]
    dtype: type = np.complex128


@dataclass(frozen=True)
class SingleFileLegacySpec:
    axes_spec: AxesSpec[Any, Any]
    axes: tuple[LegacyAxisSpec, ...]
    z: LegacyZSpec
    validate: Callable[[str], object]

    def to_converter(self) -> ConverterSpec:
        return ConverterSpec(convert=self.convert, validate=self.validate)

    def convert(self, input_path: Path, output_path: Path) -> None:
        data = load_labber_data(str(input_path))
        loaded_axes = _load_legacy_axes(data, input_path, self.axes)
        loaded_z = _load_legacy_z(data, input_path, self.z)
        expected_shape = tuple(axis.shape[0] for axis in reversed(loaded_axes))
        if loaded_z.shape != expected_shape:
            raise ValueError(
                f"legacy file {input_path} z shape {loaded_z.shape} != "
                f"expected legacy shape {expected_shape}"
            )

        kwargs: dict[str, Any] = {
            axis.field_name: values for axis, values in zip(self.axes, loaded_axes)
        }
        kwargs[self.axes_spec.z.field_name] = _cast_z(
            loaded_z,
            self.axes_spec.z.dtype,
            context=f"legacy file {input_path} z channel",
        )
        result = self.axes_spec.result_type(**kwargs)
        save_axes_spec_result_exact(
            output_path,
            self.axes_spec,
            result,
            comment=data.comment,
            tag=self.axes_spec.tag,
        )


def migrate_experiment_data(
    *,
    experiment: str,
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
    converters: Mapping[str, ConverterSpec] | None = None,
) -> Path:
    registry = CONVERTERS if converters is None else converters
    try:
        spec = registry[experiment]
    except KeyError:
        supported = ", ".join(sorted(registry))
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


@contextmanager
def migrated_legacy_tempfile(
    *,
    experiment: str,
    input_path: str | Path,
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="zcu_legacy_migration_") as temp_dir:
        output_path = Path(temp_dir) / "canonical.hdf5"
        yield migrate_experiment_data(
            experiment=experiment,
            input_path=Path(input_path),
            output_path=output_path,
        )


def save_axes_spec_result_exact(
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


def _make_temp_path(output_path: Path) -> Path:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".hdf5",
        dir=output_path.parent,
    )
    os.close(fd)
    os.unlink(temp_name)
    return Path(temp_name)


def _validate_regular_input_file(input_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")


def _axes_spec(exp_cls: type[Any]) -> AxesSpec[Any, Any]:
    axes_spec = exp_cls.AXES_SPEC
    if axes_spec is None:
        raise RuntimeError(f"{exp_cls.__name__} has no AXES_SPEC")
    return axes_spec


def _load_legacy_axes(
    data: LabberData,
    path: Path,
    axes: tuple[LegacyAxisSpec, ...],
) -> tuple[NDArray[Any], ...]:
    if len(data.axes) != len(axes):
        raise ValueError(
            f"legacy file {path} has {len(data.axes)} axes; expected {len(axes)}"
        )

    loaded: list[NDArray[Any]] = []
    for index, (axis, axis_spec) in enumerate(zip(data.axes, axes, strict=True)):
        if axis.name not in axis_spec.labels:
            expected_labels = ", ".join(repr(label) for label in axis_spec.labels)
            raise ValueError(
                f"legacy file {path} axis {index} label is {axis.name!r}; "
                f"expected one of: {expected_labels}"
            )
        try:
            scale = axis_spec.unit_to_memory_scale[axis.unit]
        except KeyError:
            expected_units = ", ".join(
                repr(unit) for unit in axis_spec.unit_to_memory_scale
            )
            raise ValueError(
                f"legacy file {path} axis {index} unit is {axis.unit!r}; "
                f"expected one of: {expected_units}"
            ) from None

        values = np.asarray(axis.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(
                f"legacy file {path} axis {axis.name!r} is {values.ndim}D; expected 1D"
            )
        loaded.append((values * scale).astype(axis_spec.dtype))

    return tuple(loaded)


def _load_legacy_z(
    data: LabberData,
    path: Path,
    z_spec: LegacyZSpec,
) -> NDArray[Any]:
    if data.data.name not in z_spec.labels:
        expected_labels = ", ".join(repr(label) for label in z_spec.labels)
        raise ValueError(
            f"legacy file {path} z channel label is {data.data.name!r}; "
            f"expected one of: {expected_labels}"
        )
    if data.data.unit not in z_spec.units:
        expected_units = ", ".join(repr(unit) for unit in z_spec.units)
        raise ValueError(
            f"legacy file {path} z channel unit is {data.data.unit!r}; "
            f"expected one of: {expected_units}"
        )
    return _cast_z(
        data.z,
        z_spec.dtype,
        context=f"legacy file {path} z channel",
    )


def _cast_z(values: Any, dtype: type, *, context: str) -> NDArray[Any]:
    target_dtype = np.dtype(dtype)
    z_values = np.asarray(values)
    if target_dtype.kind != "c" and np.iscomplexobj(z_values):
        if np.any(np.imag(z_values) != 0.0):
            raise ValueError(f"{context} contains non-zero imaginary component")
        z_values = np.real(z_values)
    return z_values.astype(target_dtype)


_FREQ_MHZ_UNITS: Mapping[str, float] = {
    "Hz": 1e-6,
    "kHz": 1e-3,
    "MHz": 1.0,
    "GHz": 1e3,
}
_DEVICE_VALUE_UNITS: Mapping[str, float] = {
    "": 1.0,
    "a.u.": 1.0,
    "a.u": 1.0,
    "V": 1.0,
    "A": 1.0,
}
_SIGNAL_Z = LegacyZSpec(
    labels=("Signal",),
    units=("a.u.", "ADC unit", ""),
)


def _freq_axis(*, labels: tuple[str, ...] = ("Frequency",)) -> LegacyAxisSpec:
    return LegacyAxisSpec(
        field_name="freqs",
        labels=labels,
        unit_to_memory_scale=_FREQ_MHZ_UNITS,
        dtype=np.float64,
    )


def _flux_value_axis() -> LegacyAxisSpec:
    return LegacyAxisSpec(
        field_name="values",
        labels=("Flux device value", "Yoko", "Yokogawa", "Flux", "Flux Value"),
        unit_to_memory_scale=_DEVICE_VALUE_UNITS,
        dtype=np.float64,
    )


_ONETONE_FREQ = SingleFileLegacySpec(
    axes_spec=_axes_spec(OneToneFreqExp),
    axes=(_freq_axis(),),
    z=_SIGNAL_Z,
    validate=lambda path: OneToneFreqExp().load(path),
)
_ONETONE_FLUX_DEP = SingleFileLegacySpec(
    axes_spec=_axes_spec(OneToneFluxDepExp),
    axes=(_freq_axis(), _flux_value_axis()),
    z=_SIGNAL_Z,
    validate=lambda path: OneToneFluxDepExp().load(path),
)
_TWOTONE_FREQ = SingleFileLegacySpec(
    axes_spec=_axes_spec(TwoToneFreqExp),
    axes=(_freq_axis(labels=("Frequency", "Qubit Frequency")),),
    z=_SIGNAL_Z,
    validate=lambda path: TwoToneFreqExp().load(path),
)
_TWOTONE_FLUX_DEP = SingleFileLegacySpec(
    axes_spec=_axes_spec(FreqFluxExp),
    axes=(_freq_axis(labels=("Frequency", "Qubit Frequency")), _flux_value_axis()),
    z=_SIGNAL_Z,
    validate=lambda path: FreqFluxExp().load(path),
)

CONVERTERS: dict[str, ConverterSpec] = {
    "onetone/freq": _ONETONE_FREQ.to_converter(),
    "onetone/flux_dep": _ONETONE_FLUX_DEP.to_converter(),
    "twotone/freq": _TWOTONE_FREQ.to_converter(),
    "twotone/flux_dep": _TWOTONE_FLUX_DEP.to_converter(),
    "twotone/flux_dep/freq": _TWOTONE_FLUX_DEP.to_converter(),
    "twotone/fluxdep": _TWOTONE_FLUX_DEP.to_converter(),
}
