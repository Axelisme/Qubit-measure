from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

_LOG10 = np.log(10.0)


@dataclass(frozen=True, slots=True)
class ThermalAttenuatorStage:
    name: str
    Temp_K: float
    attenuation_db: float


@dataclass(frozen=True, slots=True)
class ThermalProbeResult:
    frequency_hz: float
    log10_psd_v2_per_hz: float
    effective_temperature_K: float
    photon_number: float


@dataclass(frozen=True, slots=True)
class ThermalChainResult:
    frequencies_hz: NDArray[np.float64]
    impedance_ohm: float
    input_temperature_K: float
    stages: tuple[ThermalAttenuatorStage, ...]
    contribution_labels: tuple[str, ...]
    contribution_log10_psd_v2_per_hz: NDArray[np.float64]
    total_log10_psd_v2_per_hz: NDArray[np.float64]
    effective_temperature_K: NDArray[np.float64]
    effective_photon_number: NDArray[np.float64]

    def probe(self, frequency_hz: float) -> ThermalProbeResult:
        return evaluate_thermal_chain_at_frequency(
            frequency_hz,
            self.stages,
            input_temperature_K=self.input_temperature_K,
            impedance_ohm=self.impedance_ohm,
        )


def blackbody_photon_number(
    Temp_K: float,
    frequencies_hz: ArrayLike,
) -> NDArray[np.float64]:
    Temp_K = _validate_positive_scalar("Temp_K", Temp_K)
    frequencies = _as_frequency_array(frequencies_hz)
    x = sc.h * frequencies / (sc.k * Temp_K)
    with np.errstate(over="ignore"):
        photons = 1.0 / np.expm1(x)
    return np.asarray(photons, dtype=np.float64)


def effective_temperature_from_photon_number(
    photon_number: ArrayLike,
    frequencies_hz: ArrayLike,
) -> NDArray[np.float64]:
    photons = np.asarray(photon_number, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if np.any(~np.isfinite(photons)) or np.any(photons < 0.0):
        raise ValueError("photon_number must be finite and non-negative")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must be finite and positive")

    photons, frequencies = np.broadcast_arrays(photons, frequencies)
    result = np.zeros_like(photons, dtype=np.float64)
    mask = photons > 0.0
    result[mask] = sc.h * frequencies[mask] / (sc.k * np.log1p(1.0 / photons[mask]))
    return result


def thermal_psd_log10_v2_per_hz(
    Temp_K: float,
    frequencies_hz: ArrayLike,
    *,
    impedance_ohm: float = 50.0,
) -> NDArray[np.float64]:
    Temp_K = _validate_positive_scalar("Temp_K", Temp_K)
    impedance_ohm = _validate_positive_scalar("impedance_ohm", impedance_ohm)
    frequencies = _as_frequency_array(frequencies_hz)
    x = sc.h * frequencies / (sc.k * Temp_K)
    return (
        np.log10(4.0 * sc.k * Temp_K * impedance_ohm)
        + np.log10(x)
        - _log_expm1(x) / _LOG10
    )


def calculate_thermal_chain(
    frequencies_hz: ArrayLike,
    stages: Sequence[ThermalAttenuatorStage],
    *,
    input_temperature_K: float = 300.0,
    impedance_ohm: float = 50.0,
) -> ThermalChainResult:
    frequencies = _as_frequency_array(frequencies_hz)
    input_temperature_K = _validate_positive_scalar(
        "input_temperature_K",
        input_temperature_K,
    )
    impedance_ohm = _validate_positive_scalar("impedance_ohm", impedance_ohm)
    normalized_stages = _validate_stages(stages)

    attenuations = np.array(
        [stage.attenuation_db for stage in normalized_stages],
        dtype=np.float64,
    )
    downstream_attenuation_db = np.array(
        [
            float(np.sum(attenuations[index + 1 :]))
            for index in range(len(normalized_stages))
        ],
        dtype=np.float64,
    )
    total_attenuation_db = float(np.sum(attenuations))

    labels: list[str] = [
        f"input source ({_format_temperature(input_temperature_K)})",
    ]
    contributions = [
        thermal_psd_log10_v2_per_hz(
            input_temperature_K,
            frequencies,
            impedance_ohm=impedance_ohm,
        )
        - total_attenuation_db / 10.0,
    ]

    for stage, downstream_db in zip(normalized_stages, downstream_attenuation_db):
        emissivity = 1.0 - 10.0 ** (-stage.attenuation_db / 10.0)
        stage_log_psd = thermal_psd_log10_v2_per_hz(
            stage.Temp_K,
            frequencies,
            impedance_ohm=impedance_ohm,
        )
        if emissivity == 0.0:
            contribution = np.full_like(frequencies, -np.inf, dtype=np.float64)
        else:
            contribution = stage_log_psd + np.log10(emissivity) - downstream_db / 10.0
        labels.append(f"{stage.name} emission ({_format_temperature(stage.Temp_K)})")
        contributions.append(contribution)

    contribution_array = np.vstack(contributions)
    total_log10_psd = _log10sumexp(contribution_array, axis=0)
    effective_photons = _photon_number_from_psd_log10(
        total_log10_psd,
        frequencies,
        impedance_ohm=impedance_ohm,
    )
    effective_temperature = effective_temperature_from_photon_number(
        effective_photons,
        frequencies,
    )

    return ThermalChainResult(
        frequencies_hz=frequencies,
        impedance_ohm=impedance_ohm,
        input_temperature_K=input_temperature_K,
        stages=normalized_stages,
        contribution_labels=tuple(labels),
        contribution_log10_psd_v2_per_hz=contribution_array,
        total_log10_psd_v2_per_hz=total_log10_psd,
        effective_temperature_K=effective_temperature,
        effective_photon_number=effective_photons,
    )


def evaluate_thermal_chain_at_frequency(
    frequency_hz: float,
    stages: Sequence[ThermalAttenuatorStage],
    *,
    input_temperature_K: float = 300.0,
    impedance_ohm: float = 50.0,
) -> ThermalProbeResult:
    frequency_hz = _validate_positive_scalar("frequency_hz", frequency_hz)
    result = calculate_thermal_chain(
        np.array([frequency_hz], dtype=np.float64),
        stages,
        input_temperature_K=input_temperature_K,
        impedance_ohm=impedance_ohm,
    )
    return ThermalProbeResult(
        frequency_hz=frequency_hz,
        log10_psd_v2_per_hz=float(result.total_log10_psd_v2_per_hz[0]),
        effective_temperature_K=float(result.effective_temperature_K[0]),
        photon_number=float(result.effective_photon_number[0]),
    )


def plot_thermal_chain_psd(
    result: ThermalChainResult,
    *,
    probe_frequency_hz: float | None = None,
    ax: Axes | None = None,
    ylim: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
    else:
        fig = cast(Figure, ax.figure)

    raw_source_labels = [
        _format_temperature_compact(result.input_temperature_K),
        *(stage.name for stage in result.stages),
    ]
    raw_source_temperatures = [
        result.input_temperature_K,
        *(stage.Temp_K for stage in result.stages),
    ]
    for label, Temp_K in zip(raw_source_labels, raw_source_temperatures):
        raw_psd = thermal_psd_log10_v2_per_hz(
            Temp_K,
            result.frequencies_hz,
            impedance_ohm=result.impedance_ohm,
        )
        ax.plot(result.frequencies_hz, raw_psd, linewidth=1.2, label=label)

    ax.plot(
        result.frequencies_hz,
        result.total_log10_psd_v2_per_hz,
        color="k",
        linewidth=2.2,
        label="Effective",
    )

    if probe_frequency_hz is not None:
        probe = result.probe(probe_frequency_hz)
        equivalent_log_psd = thermal_psd_log10_v2_per_hz(
            probe.effective_temperature_K,
            result.frequencies_hz,
            impedance_ohm=result.impedance_ohm,
        )
        ax.axvline(
            probe.frequency_hz,
            color="k",
            linestyle="--",
            linewidth=1.0,
            label=f"freq = {probe.frequency_hz * 1e-9:.3g}GHz",
        )
        ax.plot(
            result.frequencies_hz,
            equivalent_log_psd,
            linestyle="--",
            linewidth=1.5,
            label=f"T_eff = {probe.effective_temperature_K * 1e3:.1f} mK",
        )
        ax.scatter(
            [probe.frequency_hz],
            [probe.log10_psd_v2_per_hz],
            color="tab:red",
            marker="*",
            s=140,
            zorder=5,
            label=f"n = {probe.photon_number:.3g}",
        )

    ax.set_xscale("log")
    ax.set_xlim(
        float(np.min(result.frequencies_hz)), float(np.max(result.frequencies_hz))
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, which="both", alpha=0.35)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$\log_{10} S_{VV}$ [$V^2$/Hz]")
    ax.legend(fontsize="small", loc="best")
    fig.tight_layout()
    return fig, ax


def plot_effective_temperature_vs_frequency(
    result: ThermalChainResult,
    *,
    probe_frequency_hz: float | None = None,
    highlight_range_hz: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
    else:
        fig = cast(Figure, ax.figure)

    ax.plot(
        result.frequencies_hz,
        result.effective_temperature_K * 1e3,
        color="k",
        linewidth=2.0,
        label="effective temperature",
    )

    if probe_frequency_hz is not None:
        probe = result.probe(probe_frequency_hz)
        ax.axvline(
            probe.frequency_hz,
            color="k",
            linestyle="--",
            linewidth=1.0,
            label=f"probe = {probe.frequency_hz * 1e-9:.3g} GHz",
        )
        ax.scatter(
            [probe.frequency_hz],
            [probe.effective_temperature_K * 1e3],
            color="tab:red",
            marker="*",
            s=120,
            zorder=5,
            label=f"{probe.effective_temperature_K * 1e3:.1f} mK",
        )

    if highlight_range_hz is not None:
        f_min, f_max = _validate_frequency_range(highlight_range_hz)
        ax.axvspan(f_min, f_max, color="tab:blue", alpha=0.08, label="target range")
        mask = (result.frequencies_hz >= f_min) & (result.frequencies_hz <= f_max)
        if np.any(mask):
            range_temperatures = result.effective_temperature_K[mask]
            min_index = int(np.argmin(range_temperatures))
            min_temperature = float(range_temperatures[min_index])
            ax.axhline(
                min_temperature * 1e3,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                label=f"range min = {min_temperature * 1e3:.1f} mK",
            )

    ax.set_xscale("log")
    ax.set_xlim(
        float(np.min(result.frequencies_hz)), float(np.max(result.frequencies_hz))
    )
    ax.grid(True, which="both", alpha=0.35)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Effective temperature [mK]")
    ax.legend(fontsize="small", loc="best")
    fig.tight_layout()
    return fig, ax


def _as_frequency_array(frequencies_hz: ArrayLike) -> NDArray[np.float64]:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim == 0:
        frequencies = frequencies.reshape(1)
    if frequencies.ndim != 1:
        raise ValueError("frequencies_hz must be a one-dimensional array")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must be finite and positive")
    return frequencies


def _validate_positive_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_stages(
    stages: Sequence[ThermalAttenuatorStage],
) -> tuple[ThermalAttenuatorStage, ...]:
    normalized: list[ThermalAttenuatorStage] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage.name, str) or stage.name == "":
            raise ValueError(f"stages[{index}].name must be a non-empty string")
        Temp_K = _validate_positive_scalar(f"stages[{index}].Temp_K", stage.Temp_K)
        attenuation_db = float(stage.attenuation_db)
        if not np.isfinite(attenuation_db) or attenuation_db < 0.0:
            raise ValueError(
                f"stages[{index}].attenuation_db must be finite and non-negative"
            )
        normalized.append(
            ThermalAttenuatorStage(
                name=stage.name,
                Temp_K=Temp_K,
                attenuation_db=attenuation_db,
            )
        )
    return tuple(normalized)


def _log_expm1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.empty_like(x, dtype=np.float64)
    small = x < 20.0
    result[small] = np.log(np.expm1(x[small]))
    result[~small] = x[~small] + np.log1p(-np.exp(-x[~small]))
    return result


def _log10sumexp(values: NDArray[np.float64], *, axis: int) -> NDArray[np.float64]:
    natural_logs = values * _LOG10
    max_log = np.max(natural_logs, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore"):
        shifted_sum = np.sum(np.exp(natural_logs - max_log), axis=axis)
    squeezed_max = np.squeeze(max_log, axis=axis)
    result = (squeezed_max + np.log(shifted_sum)) / _LOG10
    all_negative_infinite = np.all(np.isneginf(values), axis=axis)
    return np.where(all_negative_infinite, -np.inf, result)


def _photon_number_from_psd_log10(
    log10_psd_v2_per_hz: NDArray[np.float64],
    frequencies_hz: NDArray[np.float64],
    *,
    impedance_ohm: float,
) -> NDArray[np.float64]:
    log10_prefactor = np.log10(4.0 * impedance_ohm * sc.h * frequencies_hz)
    return np.power(10.0, log10_psd_v2_per_hz - log10_prefactor)


def _validate_frequency_range(
    frequency_range_hz: tuple[float, float],
) -> tuple[float, float]:
    f_min = _validate_positive_scalar("highlight_range_hz[0]", frequency_range_hz[0])
    f_max = _validate_positive_scalar("highlight_range_hz[1]", frequency_range_hz[1])
    if f_min >= f_max:
        raise ValueError("highlight_range_hz must be increasing")
    return f_min, f_max


def _format_temperature(Temp_K: float) -> str:
    if Temp_K >= 1.0:
        return f"{Temp_K:g} K"
    return f"{Temp_K * 1e3:g} mK"


def _format_temperature_compact(Temp_K: float) -> str:
    if Temp_K >= 1.0:
        return f"{Temp_K:g}K"
    return f"{Temp_K * 1e3:g}mK"


__all__ = [
    "ThermalAttenuatorStage",
    "ThermalChainResult",
    "ThermalProbeResult",
    "blackbody_photon_number",
    "calculate_thermal_chain",
    "effective_temperature_from_photon_number",
    "evaluate_thermal_chain_at_frequency",
    "plot_effective_temperature_vs_frequency",
    "plot_thermal_chain_psd",
    "thermal_psd_log10_v2_per_hz",
]
