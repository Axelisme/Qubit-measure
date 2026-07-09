---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
  kernelspec:
    display_name: zcu-tools
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.13.11
---

# T2 curve

This notebook follows the T1-curve workflow style but analyzes echo/Ramsey T2
columns in `result/Q12_2D[7]/Q4/samples.csv`.

The physical model follows Nguyen et al., *High-Coherence Fluxonium Qubit*,
Phys. Rev. X 9, 041041 (2019):

- T1 is the energy-relaxation channel; any homogeneous T2 is bounded by
  `T2 <= 2*T1`.
- Echo T2 is used to reject slow setup drift. Away from a sweet spot, the paper
  models Gaussian echo decay from first-order 1/f flux noise as
  `Gamma_echo = sqrt(ln 2) * A_phi * |d omega01 / d Phi|`.
- Residual pure dephasing is summarized by
  `Tphi = 1 / (1/T2 - 1/(2*T1))`.
- Near a sweet spot, first-order flux sensitivity should vanish; residual
  dephasing can then come from pulse/flux drift, imperfect echo, higher-order
  flux noise, or photon shot noise.
- Residual readout thermal photons are estimated with Appendix A15:
  `Gamma_phi_th = n_th*kappa*chi^2/(kappa^2 + chi^2)`, then
  `T2_limit = 1/(1/(2*T1) + Gamma_phi_th)`.

For the current `Q12_2D[7]/Q4` sample table, the canonical T2 columns are
`T2r (us)` and `T2e (us)`. The weighted flux-noise fit uses rows that also have
`T2e err (us)` and `T1err (us)`; rows without fit uncertainty are still shown in
the branch-coverage and sample plots.

# Import

```python
%load_ext autoreload

import os
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import least_squares

%autoreload 2
from zcu_tools.meta_tool import QubitParams
import zcu_tools.notebook.analysis.t1_curve as zt1
from zcu_tools.simulate import value2flux
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux
import zcu_tools.simulate.fluxonium as zf
```

```python
chip_name = "Q12_2D[7]"
qub_name = "Q4"

result_dir = os.path.join("..", "..", "result", chip_name, qub_name)
t2_curve_dir = os.path.join(result_dir, "t2_curve")
image_dir = t2_curve_dir
os.makedirs(result_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
```

# Helpers

```python
def predict_f01_mhz(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    *,
    cutoff: int = 40,
) -> NDArray[np.float64]:
    _, energies = calculate_energy_vs_flux(
        params, np.asarray(fluxs, dtype=np.float64), cutoff=cutoff, evals_count=4
    )
    return np.asarray(1e3 * (energies[:, 1] - energies[:, 0]), dtype=np.float64)


def predict_domega_dflux(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    *,
    step: float = 1e-5,
    cutoff: int = 40,
) -> NDArray[np.float64]:
    fluxs_arr = np.asarray(fluxs, dtype=np.float64)
    f_plus = predict_f01_mhz(params, fluxs_arr + step, cutoff=cutoff)
    f_minus = predict_f01_mhz(params, fluxs_arr - step, cutoff=cutoff)
    df_dflux = (f_plus - f_minus) / (2.0 * step)  # MHz / Phi0
    return np.asarray(2.0 * np.pi * df_dflux, dtype=np.float64)  # rad / us / Phi0


def choose_current_scale(
    raw_values: NDArray[np.float64],
    measured_freqs_mhz: NDArray[np.float64],
    *,
    params: tuple[float, float, float],
    flux_half: float,
    flux_period: float,
    candidates: tuple[float, ...] = (1.0, 1000.0),
) -> tuple[float, pd.DataFrame]:
    rows = []
    for scale in candidates:
        trial_values = raw_values * scale
        trial_fluxs = value2flux(trial_values, flux_half, flux_period)
        model_freqs = predict_f01_mhz(params, trial_fluxs)
        residuals = model_freqs - measured_freqs_mhz
        rows.append(
            {
                "scale": scale,
                "rms_MHz": float(np.sqrt(np.mean(residuals**2))),
                "median_abs_MHz": float(np.median(np.abs(residuals))),
                "flux_min": float(np.min(trial_fluxs)),
                "flux_max": float(np.max(trial_fluxs)),
            }
        )

    report = pd.DataFrame(rows).sort_values("rms_MHz").reset_index(drop=True)
    return float(report.loc[0, "scale"]), report


@dataclass(frozen=True)
class FluxNoiseFit:
    A_phi: float
    reduced_chi2: float
    success: bool
    message: str


def fit_first_order_flux_noise(
    gamma_phi: NDArray[np.float64],
    gamma_phi_err: NDArray[np.float64],
    domega_dflux: NDArray[np.float64],
    *,
    init_A_phi: float = 2e-6,
    loss: str = "soft_l1",
) -> FluxNoiseFit:
    sensitivity = np.sqrt(np.log(2.0)) * np.abs(domega_dflux)
    valid = (
        np.isfinite(gamma_phi)
        & np.isfinite(gamma_phi_err)
        & np.isfinite(sensitivity)
        & (gamma_phi > 0.0)
        & (gamma_phi_err > 0.0)
        & (sensitivity > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError("Need at least two valid positive-dephasing points.")

    def residual(log_A: NDArray[np.float64]) -> NDArray[np.float64]:
        A_phi = float(np.exp(log_A[0]))
        model = A_phi * sensitivity[valid]
        return (model - gamma_phi[valid]) / gamma_phi_err[valid]

    result = least_squares(
        residual,
        np.log([init_A_phi]),
        loss=loss,
        max_nfev=10000,
    )
    A_phi = float(np.exp(result.x[0]))
    chi2 = float(np.sum(residual(np.log([A_phi])) ** 2))
    dof = max(1, int(np.count_nonzero(valid)) - 1)
    return FluxNoiseFit(
        A_phi=A_phi,
        reduced_chi2=chi2 / dof,
        success=bool(result.success),
        message=str(result.message),
    )


def thermal_photon_gamma_phi_per_us(
    n_th: float | NDArray[np.float64],
    *,
    kappa_over_2pi_mhz: float,
    chi_over_2pi_mhz: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """PRX 9, 041041 Eq. A15, using MHz inputs for kappa/2pi and chi/2pi."""
    kappa = 2.0 * np.pi * kappa_over_2pi_mhz
    chi = 2.0 * np.pi * np.asarray(chi_over_2pi_mhz, dtype=np.float64)
    gamma_per_photon = kappa * chi**2 / (kappa**2 + chi**2)
    gamma = np.asarray(n_th, dtype=np.float64) * gamma_per_photon
    if np.ndim(gamma) == 0:
        return float(gamma)
    return gamma


def thermal_photon_t2_limit_us(
    n_th: float | NDArray[np.float64],
    *,
    T1_us: float | NDArray[np.float64],
    kappa_over_2pi_mhz: float,
    chi_over_2pi_mhz: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    gamma_phi = thermal_photon_gamma_phi_per_us(
        n_th,
        kappa_over_2pi_mhz=kappa_over_2pi_mhz,
        chi_over_2pi_mhz=chi_over_2pi_mhz,
    )
    t2_limit = 1.0 / (1.0 / (2.0 * np.asarray(T1_us)) + gamma_phi)
    if np.ndim(t2_limit) == 0:
        return float(t2_limit)
    return np.asarray(t2_limit, dtype=np.float64)


def equivalent_n_th_from_t2(
    *,
    T1_us: float,
    T2_us: float,
    kappa_over_2pi_mhz: float,
    chi_over_2pi_mhz: float,
) -> float:
    gamma_phi = 1.0 / T2_us - 1.0 / (2.0 * T1_us)
    gamma_per_photon = thermal_photon_gamma_phi_per_us(
        1.0,
        kappa_over_2pi_mhz=kappa_over_2pi_mhz,
        chi_over_2pi_mhz=chi_over_2pi_mhz,
    )
    if gamma_phi <= 0.0:
        return float("nan")
    return float(gamma_phi / gamma_per_photon)
```

# Load data

```python
params_file = QubitParams(os.path.join(result_dir, "params.json"), readonly=True)
fit_inputs = params_file.require_dispersive_inputs(default_bare_rf=5.0)
dispersive_fit = params_file.get_dispersive_fit()
t1_curve_fit = params_file.get_t1_curve_fit()
if dispersive_fit is None:
    raise RuntimeError("params.json needs a dispersive section with g and bare_rf.")

params = fit_inputs.params
EJ, EC, EL = params
flx_half = fit_inputs.flux_half
flx_int = fit_inputs.flux_int
flx_period = fit_inputs.flux_period
bare_rf = dispersive_fit.bare_rf
g = dispersive_fit.g
readout_kappa_over_2pi_ghz = 14.75412896809815e-3  # From T1_curve.md rf_w; assumed kappa/2pi.
readout_kappa_over_2pi_mhz = 1e3 * readout_kappa_over_2pi_ghz

print("params = ", params, " GHz")
print("flx_half = ", flx_half)
print("flx_int = ", flx_int)
print("flx_period = ", flx_period)
print("bare_rf = ", bare_rf, " GHz")
print("g = ", g, " GHz")
print("readout kappa/2pi = ", readout_kappa_over_2pi_mhz, " MHz")
print("has t1_curve_fit =", t1_curve_fit is not None)

analysis_flux_range = (0.4, 0.6)
t_fluxs = np.linspace(analysis_flux_range[0], analysis_flux_range[1], 1000)
print(f"analysis flux range = {analysis_flux_range}")
```

```python
loadpath = os.path.join(result_dir, "samples.csv")
samples_df = pd.read_csv(loadpath)

required_columns = [
    "calibrated mA",
    "Freq (MHz)",
    "T1 (us)",
    "T1err (us)",
    "T2e (us)",
    "T2e err (us)",
]
missing_columns = [name for name in required_columns if name not in samples_df.columns]
if missing_columns:
    raise KeyError(f"Missing required sample columns: {missing_columns}")

t2e_df = samples_df[~np.isnan(samples_df["T2e (us)"])].copy()
t2r_df = samples_df[~np.isnan(samples_df["T2r (us)"])].copy()

print(f"T2e rows = {len(t2e_df)}")
print(f"T2r rows = {len(t2r_df)}")
display(t2e_df.head(10))
```

## Current-scale guard

The regenerated `samples.csv` stores `calibrated mA` in mA. This guard compares
the measured f01 against the fitted fluxonium model and should select
`current_scale = 1`.

```python
freq_rows = samples_df[
    np.isfinite(samples_df["calibrated mA"]) & np.isfinite(samples_df["Freq (MHz)"])
].copy()
raw_current_values = freq_rows["calibrated mA"].to_numpy(dtype=np.float64)
measured_f01_mhz = freq_rows["Freq (MHz)"].to_numpy(dtype=np.float64)

current_scale, scale_report = choose_current_scale(
    raw_current_values,
    measured_f01_mhz,
    params=params,
    flux_half=flx_half,
    flux_period=flx_period,
)
display(scale_report)

print(f"selected current_scale = {current_scale:g}")
if current_scale != 1.0:
    print(
        "Unexpected current scale: samples.csv may still contain a mixed-unit "
        "current column."
    )
```

# Prepare sample arrays

```python
valid_t2e = np.isfinite(t2e_df["T2e (us)"].to_numpy(dtype=np.float64))
valid_t2e &= t2e_df["T2e (us)"].to_numpy(dtype=np.float64) > 0.0
t2e_df = t2e_df[valid_t2e].copy()

s_current_raw = t2e_df["calibrated mA"].to_numpy(dtype=np.float64)
s_dev_values = s_current_raw * current_scale
s_f01_mhz = t2e_df["Freq (MHz)"].to_numpy(dtype=np.float64)
s_T2e_us = t2e_df["T2e (us)"].to_numpy(dtype=np.float64)
s_T2e_err_us = t2e_df["T2e err (us)"].to_numpy(dtype=np.float64)
s_T1_us = t2e_df["T1 (us)"].to_numpy(dtype=np.float64)
s_T1_err_us = t2e_df["T1err (us)"].to_numpy(dtype=np.float64)

s_raw_fluxs = value2flux(s_dev_values, flx_half, flx_period)
s_corr_values = s_dev_values.copy()
s_corr_fluxs = np.asarray(s_raw_fluxs, dtype=np.float64).copy()
f01_correction_accepted = np.zeros_like(s_dev_values, dtype=bool)

max_abs_flx_correction = 0.03  # Phi0; same guard as T1_curve.md
finite_f01 = np.isfinite(s_f01_mhz)
if np.any(finite_f01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f01_correction = zt1.correct_flux_from_f01(
            s_dev_values[finite_f01],
            1e-3 * s_f01_mhz[finite_f01],
            params,
            flx_half,
            flx_period,
            max_abs_flux_correction=max_abs_flx_correction,
        )
    s_corr_values[finite_f01] = f01_correction.corrected_dev_values
    s_corr_fluxs[finite_f01] = f01_correction.corrected_fluxs
    f01_correction_accepted[finite_f01] = f01_correction.accepted

flux_corrections = s_corr_fluxs - s_raw_fluxs
print(
    "f01 flux correction: "
    f"accepted={np.count_nonzero(f01_correction_accepted)}/{len(s_corr_fluxs)}, "
    f"median={np.nanmedian(flux_corrections):+.4f} Phi0, "
    f"max={np.nanmax(np.abs(flux_corrections)):.4f} Phi0"
)
print(
    "f01-corrected branch coverage: "
    f"below={np.count_nonzero(s_corr_values < flx_half)}, "
    f"at={np.count_nonzero(np.isclose(s_corr_values, flx_half))}, "
    f"above={np.count_nonzero(s_corr_values > flx_half)}"
)
print(
    "f01-corrected current range: "
    f"{np.nanmin(s_corr_values):.4f} to {np.nanmax(s_corr_values):.4f} mA"
)

flux_window_mask = (
    np.isfinite(s_corr_fluxs)
    & (s_corr_fluxs >= analysis_flux_range[0])
    & (s_corr_fluxs <= analysis_flux_range[1])
)
print(
    "analysis flux-window filter: "
    f"kept={np.count_nonzero(flux_window_mask)}/{len(s_corr_fluxs)} "
    f"for {analysis_flux_range[0]:.3f} <= flux <= {analysis_flux_range[1]:.3f}"
)

s_current_raw = s_current_raw[flux_window_mask]
s_dev_values = s_dev_values[flux_window_mask]
s_corr_values = s_corr_values[flux_window_mask]
s_raw_fluxs = s_raw_fluxs[flux_window_mask]
s_corr_fluxs = s_corr_fluxs[flux_window_mask]
s_f01_mhz = s_f01_mhz[flux_window_mask]
s_T2e_us = s_T2e_us[flux_window_mask]
s_T2e_err_us = s_T2e_err_us[flux_window_mask]
s_T1_us = s_T1_us[flux_window_mask]
s_T1_err_us = s_T1_err_us[flux_window_mask]

sort_order = np.argsort(s_corr_fluxs)
s_current_raw = s_current_raw[sort_order]
s_dev_values = s_dev_values[sort_order]
s_corr_values = s_corr_values[sort_order]
s_raw_fluxs = s_raw_fluxs[sort_order]
s_corr_fluxs = s_corr_fluxs[sort_order]
s_f01_mhz = s_f01_mhz[sort_order]
s_T2e_us = s_T2e_us[sort_order]
s_T2e_err_us = s_T2e_err_us[sort_order]
s_T1_us = s_T1_us[sort_order]
s_T1_err_us = s_T1_err_us[sort_order]
```

## Branch coverage after f01 calibration

Check which T2 rows are actually on each side of `flux_half` after the same
measured-f01 correction used by `T1_curve.md`.

```python
def f01_corrected_coverage_row(label: str, frame: pd.DataFrame) -> dict[str, object]:
    finite = np.isfinite(frame["calibrated mA"]) & np.isfinite(frame["Freq (MHz)"])
    usable = frame[finite].copy()
    if usable.empty:
        return {
            "subset": label,
            "n_with_f01": 0,
            "n_in_flux_window": 0,
            "accepted": 0,
            "skipped": 0,
            "raw_min_mA": np.nan,
            "raw_max_mA": np.nan,
            "window_corr_min_mA": np.nan,
            "window_corr_max_mA": np.nan,
            "window_below_flux_half": 0,
            "window_at_flux_half": 0,
            "window_above_flux_half": 0,
        }

    raw_values = usable["calibrated mA"].to_numpy(dtype=np.float64) * current_scale
    f01_freqs = 1e-3 * usable["Freq (MHz)"].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        correction = zt1.correct_flux_from_f01(
            raw_values,
            f01_freqs,
            params,
            flx_half,
            flx_period,
            max_abs_flux_correction=max_abs_flx_correction,
        )
    corrected_values = correction.corrected_dev_values
    corrected_fluxs = correction.corrected_fluxs
    in_window = (
        np.isfinite(corrected_fluxs)
        & (corrected_fluxs >= analysis_flux_range[0])
        & (corrected_fluxs <= analysis_flux_range[1])
    )
    window_values = corrected_values[in_window]
    return {
        "subset": label,
        "n_with_f01": len(usable),
        "n_in_flux_window": int(np.count_nonzero(in_window)),
        "accepted": int(np.count_nonzero(correction.accepted)),
        "skipped": correction.skipped_count,
        "raw_min_mA": float(np.nanmin(raw_values)),
        "raw_max_mA": float(np.nanmax(raw_values)),
        "window_corr_min_mA": (
            np.nan if len(window_values) == 0 else float(np.nanmin(window_values))
        ),
        "window_corr_max_mA": (
            np.nan if len(window_values) == 0 else float(np.nanmax(window_values))
        ),
        "window_below_flux_half": int(np.count_nonzero(window_values < flx_half)),
        "window_at_flux_half": int(np.count_nonzero(np.isclose(window_values, flx_half))),
        "window_above_flux_half": int(np.count_nonzero(window_values > flx_half)),
    }


coverage_rows = [
    f01_corrected_coverage_row("T2e rows", t2e_df),
    f01_corrected_coverage_row("T2r rows", t2r_df),
]
branch_coverage_df = pd.DataFrame(coverage_rows)
display(branch_coverage_df)

half_preview_df = t2e_df[
    np.isfinite(t2e_df["calibrated mA"]) & np.isfinite(t2e_df["Freq (MHz)"])
].copy()
if not half_preview_df.empty:
    half_preview_values = (
        half_preview_df["calibrated mA"].to_numpy(dtype=np.float64) * current_scale
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        half_preview_correction = zt1.correct_flux_from_f01(
            half_preview_values,
            1e-3 * half_preview_df["Freq (MHz)"].to_numpy(dtype=np.float64),
            params,
            flx_half,
            flx_period,
            max_abs_flux_correction=max_abs_flx_correction,
        )
    half_preview = half_preview_df[
        ["calibrated mA", "Freq (MHz)", "T2r (us)", "T2e (us)", "T2e err (us)"]
    ].copy()
    half_preview["raw current (mA)"] = half_preview_values
    half_preview["f01-corrected current (mA)"] = (
        half_preview_correction.corrected_dev_values
    )
    half_preview["f01-corrected flux"] = half_preview_correction.corrected_fluxs
    half_preview["f01 correction accepted"] = half_preview_correction.accepted
    half_preview = half_preview[
        (half_preview["f01-corrected flux"] >= analysis_flux_range[0])
        & (half_preview["f01-corrected flux"] <= analysis_flux_range[1])
    ]
    display(
        half_preview[
            np.isclose(half_preview["f01-corrected current (mA)"], flx_half)
        ]
    )
```

```python
summary_rows = [
    ("T2e", np.isfinite(s_T2e_us), s_T2e_us),
    ("T2e err", np.isfinite(s_T2e_err_us), s_T2e_err_us),
    ("T1", np.isfinite(s_T1_us), s_T1_us),
    ("Freq", np.isfinite(s_f01_mhz), s_f01_mhz),
]
for name, mask, values in summary_rows:
    finite = values[mask]
    print(
        f"{name:7s}: n={len(finite):2d}, "
        f"min={np.min(finite):.4g}, median={np.median(finite):.4g}, "
        f"max={np.max(finite):.4g}"
    )

for col in ("T2r (us)", "T2e (us)"):
    values = samples_df[col].to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    print(
        f"{col:10s}: n={len(finite):2d}, "
        f"min={np.min(finite):.4g}, median={np.median(finite):.4g}, "
        f"max={np.max(finite):.4g}"
    )
```

# Derived dephasing

```python
sample_mask = (
    np.isfinite(s_corr_fluxs)
    & np.isfinite(s_f01_mhz)
    & np.isfinite(s_T1_us)
    & np.isfinite(s_T2e_us)
    & (s_T1_us > 0.0)
    & (s_T2e_us > 0.0)
)
fit_mask = (
    sample_mask
    & np.isfinite(s_T1_err_us)
    & np.isfinite(s_T2e_err_us)
    & (s_T1_err_us >= 0.0)
    & (s_T2e_err_us > 0.0)
    & (s_T2e_err_us < 0.5 * s_T2e_us)
)

sample_values = s_corr_values[sample_mask]
sample_fluxs = s_corr_fluxs[sample_mask]
sample_f01_mhz = s_f01_mhz[sample_mask]
sample_T1_us = s_T1_us[sample_mask]
sample_T2e_us = s_T2e_us[sample_mask]
sample_T2e_err_us = s_T2e_err_us[sample_mask]

fit_values = s_corr_values[fit_mask]
fit_fluxs = s_corr_fluxs[fit_mask]
fit_f01_mhz = s_f01_mhz[fit_mask]
fit_T1_us = s_T1_us[fit_mask]
fit_T1_err_us = s_T1_err_us[fit_mask]
fit_T2e_us = s_T2e_us[fit_mask]
fit_T2e_err_us = s_T2e_err_us[fit_mask]

gamma_2 = 1.0 / fit_T2e_us
gamma_relax_half = 1.0 / (2.0 * fit_T1_us)
gamma_phi = gamma_2 - gamma_relax_half
gamma_phi_err = np.sqrt(
    (fit_T2e_err_us / fit_T2e_us**2) ** 2
    + (0.5 * fit_T1_err_us / fit_T1_us**2) ** 2
)
Tphi_us = np.where(gamma_phi > 0.0, 1.0 / gamma_phi, np.nan)

sample_gamma_phi = 1.0 / sample_T2e_us - 1.0 / (2.0 * sample_T1_us)
sample_Tphi_us = np.where(sample_gamma_phi > 0.0, 1.0 / sample_gamma_phi, np.nan)

print(f"sample rows = {len(sample_T2e_us)}")
print(f"fit rows = {len(fit_T2e_us)}")
print(
    "sample 2T1/T2e: "
    f"min={np.nanmin(2.0 * sample_T1_us / sample_T2e_us):.3g}, "
    f"median={np.nanmedian(2.0 * sample_T1_us / sample_T2e_us):.3g}, "
    f"max={np.nanmax(2.0 * sample_T1_us / sample_T2e_us):.3g}"
)
print(
    "fit 2T1/T2e: "
    f"min={np.nanmin(2.0 * fit_T1_us / fit_T2e_us):.3g}, "
    f"median={np.nanmedian(2.0 * fit_T1_us / fit_T2e_us):.3g}, "
    f"max={np.nanmax(2.0 * fit_T1_us / fit_T2e_us):.3g}"
)
print(
    "fit Tphi: "
    f"min={np.nanmin(Tphi_us):.3g} us, "
    f"median={np.nanmedian(Tphi_us):.3g} us, "
    f"max={np.nanmax(Tphi_us):.3g} us"
)

sample_peak_idx = int(np.nanargmax(sample_T2e_us))
print(
    "sample peak T2e point: "
    f"current={sample_values[sample_peak_idx]:.4f} mA, "
    f"flux={sample_fluxs[sample_peak_idx]:.6f}, "
    f"f01={sample_f01_mhz[sample_peak_idx]:.3f} MHz, "
    f"T2e={sample_T2e_us[sample_peak_idx]:.3f} us, "
    f"T1={sample_T1_us[sample_peak_idx]:.3f} us, "
    f"2T1={2.0 * sample_T1_us[sample_peak_idx]:.3f} us, "
    f"Tphi={sample_Tphi_us[sample_peak_idx]:.3f} us"
)

fit_peak_idx = int(np.nanargmax(fit_T2e_us))
print(
    "weighted-fit peak T2e point: "
    f"current={fit_values[fit_peak_idx]:.4f} mA, "
    f"flux={fit_fluxs[fit_peak_idx]:.6f}, "
    f"f01={fit_f01_mhz[fit_peak_idx]:.3f} MHz, "
    f"T2e={fit_T2e_us[fit_peak_idx]:.3f} +/- {fit_T2e_err_us[fit_peak_idx]:.3f} us, "
    f"T1={fit_T1_us[fit_peak_idx]:.3f} us, "
    f"2T1={2.0 * fit_T1_us[fit_peak_idx]:.3f} us, "
    f"Tphi={Tphi_us[fit_peak_idx]:.3f} us"
)
```

```python
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(
    sample_fluxs,
    sample_T2e_us,
    "o",
    color="tab:blue",
    label="T2 echo sample",
)
ax.errorbar(
    fit_fluxs,
    fit_T2e_us,
    yerr=fit_T2e_err_us,
    fmt="none",
    ecolor="tab:blue",
    alpha=0.7,
    capsize=3,
)
ax.scatter(
    sample_fluxs,
    2.0 * sample_T1_us,
    s=24,
    color="tab:green",
    label="2*T1 sample",
)
ax.set_xlabel(r"Flux $\Phi_\mathrm{ext}/\Phi_0$")
ax.set_ylabel("Time (us)")
ax.set_title("Echo T2 vs T1 ceiling")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(image_dir, "T2e_vs_flux.png"), dpi=160)
plt.show()
plt.close(fig)
```

# Readout thermal-photon shot-noise limit

Use the PRX Appendix A15 low-photon estimate for residual photons in the
readout mode:

`Gamma_phi_th = n_th * kappa * chi^2 / (kappa^2 + chi^2)`.

Here `chi/2pi` is taken as the simulated qubit-state-dependent readout
frequency separation `|rf_1 - rf_0|`, matching the `chi01/2pi` convention used
in the PRX paper's Table I. The linewidth is not stored in `params.json`, so
this notebook uses the existing `T1_curve.md` value `rf_w = 4.2e-3 GHz` as
`kappa/2pi = 4.2 MHz`.

```python
def dispersive_chi01_over_2pi_mhz(fluxs: NDArray[np.float64]) -> NDArray[np.float64]:
    rf_0_ghz, rf_1_ghz = zf.calculate_dispersive_vs_flux_fast(
        params,
        np.asarray(fluxs, dtype=np.float64),
        bare_rf,
        g,
        res_dim=5,
        qub_dim=15,
        return_dim=2,
    )
    return np.asarray(1e3 * np.abs(rf_1_ghz - rf_0_ghz), dtype=np.float64)


half_sample_idx = int(np.nanargmin(np.abs(sample_fluxs - 0.5)))
half_flux = float(sample_fluxs[half_sample_idx])
half_T1_us = float(sample_T1_us[half_sample_idx])
half_T2e_us = float(sample_T2e_us[half_sample_idx])
half_gamma_phi_per_us = float(sample_gamma_phi[half_sample_idx])
half_chi_over_2pi_mhz = float(
    dispersive_chi01_over_2pi_mhz(np.asarray([half_flux], dtype=np.float64))[0]
)
half_gamma_per_photon_us = thermal_photon_gamma_phi_per_us(
    1.0,
    kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
    chi_over_2pi_mhz=half_chi_over_2pi_mhz,
)
half_equivalent_n_th = equivalent_n_th_from_t2(
    T1_us=half_T1_us,
    T2_us=half_T2e_us,
    kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
    chi_over_2pi_mhz=half_chi_over_2pi_mhz,
)

fit_peak_chi_over_2pi_mhz = float(
    dispersive_chi01_over_2pi_mhz(
        np.asarray([fit_fluxs[fit_peak_idx]], dtype=np.float64)
    )[0]
)
fit_peak_equivalent_n_th = equivalent_n_th_from_t2(
    T1_us=float(fit_T1_us[fit_peak_idx]),
    T2_us=float(fit_T2e_us[fit_peak_idx]),
    kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
    chi_over_2pi_mhz=fit_peak_chi_over_2pi_mhz,
)

print(
    "half-flux thermal photon estimate: "
    f"flux={half_flux:.6f}, "
    f"T1={half_T1_us:.3f} us, "
    f"T2e={half_T2e_us:.3f} us"
)
print(f"chi01/2pi = {half_chi_over_2pi_mhz:.3f} MHz")
print(f"kappa/2pi = {readout_kappa_over_2pi_mhz:.3f} MHz")
print(f"Gamma_phi observed = {half_gamma_phi_per_us:.5f} 1/us")
print(f"Gamma_phi per thermal photon = {half_gamma_per_photon_us:.3f} 1/us")
print(f"equivalent n_th = {half_equivalent_n_th:.3e}")
print(
    "weighted-fit peak equivalent n_th = "
    f"{fit_peak_equivalent_n_th:.3e} "
    f"(flux={fit_fluxs[fit_peak_idx]:.6f})"
)

thermal_probe_n_th = np.asarray(
    [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, half_equivalent_n_th],
    dtype=np.float64,
)
thermal_probe_n_th = np.unique(np.sort(thermal_probe_n_th[np.isfinite(thermal_probe_n_th)]))
thermal_limit_table = pd.DataFrame(
    {
        "n_th": thermal_probe_n_th,
        "Gamma_phi_th (1/us)": thermal_photon_gamma_phi_per_us(
            thermal_probe_n_th,
            kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
            chi_over_2pi_mhz=half_chi_over_2pi_mhz,
        ),
        "T2_limit (us)": thermal_photon_t2_limit_us(
            thermal_probe_n_th,
            T1_us=half_T1_us,
            kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
            chi_over_2pi_mhz=half_chi_over_2pi_mhz,
        ),
    }
)
display(thermal_limit_table)
```

```python
n_th_axis = np.logspace(-5, -1, 400)
thermal_T2_limit_axis_us = thermal_photon_t2_limit_us(
    n_th_axis,
    T1_us=half_T1_us,
    kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
    chi_over_2pi_mhz=half_chi_over_2pi_mhz,
)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.semilogx(n_th_axis, thermal_T2_limit_axis_us, label="thermal photon + T1 limit")
ax.axhline(2.0 * half_T1_us, color="tab:green", linestyle="--", label="2*T1")
ax.axhline(half_T2e_us, color="tab:orange", linestyle=":", label="measured T2 echo")
ax.axvline(
    half_equivalent_n_th,
    color="tab:red",
    linestyle=":",
    label=rf"equiv. $n_{{th}}={half_equivalent_n_th:.2e}$",
)
ax.set_xlabel(r"Residual readout thermal photons $n_{th}$")
ax.set_ylabel("T2 limit (us)")
ax.set_title("Half-flux thermal photon shot-noise ceiling")
ax.grid(True, which="both", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(image_dir, "T2e_thermal_photon_limit.png"), dpi=160)
plt.show()
plt.close(fig)
```

# First-order 1/f flux-noise fit

Fit only the flux-noise residual. The target rate is:

`Gamma_flux,target = 1/T2e - 1/(2*T1) - Gamma_photon(flux)`.

Rows whose residual target is nonpositive are excluded by the same positive-rate
guard used by the fit helper.

```python
domega_dflux = predict_domega_dflux(params, fit_fluxs)
fit_chi_over_2pi_mhz = dispersive_chi01_over_2pi_mhz(fit_fluxs)
fit_gamma_phi_photon = thermal_photon_gamma_phi_per_us(
    half_equivalent_n_th,
    kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
    chi_over_2pi_mhz=fit_chi_over_2pi_mhz,
)
gamma_phi_flux = gamma_phi - fit_gamma_phi_photon

flux_noise_fit = fit_first_order_flux_noise(
    gamma_phi_flux,
    gamma_phi_err,
    domega_dflux,
    init_A_phi=2e-6,
    loss="soft_l1",
)

print("fit success =", flux_noise_fit.success)
print("fit message =", flux_noise_fit.message)
print(
    f"A_phi = {flux_noise_fit.A_phi:.3e} Phi0/sqrt(Hz) "
    f"= {flux_noise_fit.A_phi * 1e6:.3f} uPhi0/sqrt(Hz)"
)
print(f"reduced chi2 = {flux_noise_fit.reduced_chi2:.3g}")

positive_flux_target = gamma_phi_flux[np.isfinite(gamma_phi_flux) & (gamma_phi_flux > 0.0)]
print(
    "photon-subtracted flux-dephasing target: "
    f"positive_n={len(positive_flux_target)}/{len(gamma_phi_flux)}, "
    f"min={np.min(positive_flux_target):.4g}, "
    f"median={np.median(positive_flux_target):.4g}, "
    f"max={np.max(positive_flux_target):.4g} 1/us"
)

pointwise_A = gamma_phi_flux / (np.sqrt(np.log(2.0)) * np.abs(domega_dflux))
finite_A = pointwise_A[np.isfinite(pointwise_A) & (pointwise_A > 0.0)]
print(
    "pointwise A_phi: "
    f"min={np.min(finite_A) * 1e6:.3f}, "
    f"median={np.median(finite_A) * 1e6:.3f}, "
    f"max={np.max(finite_A) * 1e6:.3f} uPhi0/sqrt(Hz)"
)
```

```python
x_sensitivity = np.sqrt(np.log(2.0)) * np.abs(domega_dflux)
x_line = np.linspace(0.0, 1.05 * np.nanmax(x_sensitivity), 200)
y_line = flux_noise_fit.A_phi * x_line

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.errorbar(
    x_sensitivity,
    gamma_phi_flux,
    yerr=gamma_phi_err,
    fmt="o",
    capsize=3,
    label="sample dephasing after T1 + photon subtraction",
)
ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
ax.plot(
    x_line,
    y_line,
    label=rf"$A_\Phi$ = {flux_noise_fit.A_phi * 1e6:.2f} $\mu\Phi_0/\sqrt{{Hz}}$",
)
ax.set_xlabel(r"$\sqrt{\ln 2}\,|\partial\omega_{01}/\partial\Phi|$ (1/us/Phi0)")
ax.set_ylabel(r"$\Gamma_\phi - \Gamma_\mathrm{photon}$ (1/us)")
ax.set_title("Photon-subtracted echo dephasing vs flux sensitivity")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(image_dir, "Gamma_phi_vs_flux_sensitivity.png"), dpi=160)
plt.show()
plt.close(fig)
```

## Overlay fitted T2 curve

Use the stored T1-curve fit when available, so the continuous T2 curve keeps the
same T1 mechanism as `T1_curve.md`. If no T1 fit exists, the notebook falls back
to a simple interpolation of measured T1 values over the plotted flux window.
The final model overlay separates the relaxation-only ceiling, the pure
first-order flux-noise dephasing scale, the pure readout thermal photon
shot-noise dephasing scale, and the effective T2 from adding all three rates.

```python
grid_fluxs = t_fluxs
grid_domega_dflux = predict_domega_dflux(params, grid_fluxs)
grid_gamma_phi = (
    np.sqrt(np.log(2.0)) * flux_noise_fit.A_phi * np.abs(grid_domega_dflux)
)

if t1_curve_fit is not None:
    t1_noise_channels = []
    if t1_curve_fit.params.Q_cap is not None:
        t1_noise_channels.append(
            ("t1_capacitive", {"Q_cap": t1_curve_fit.params.Q_cap})
        )
    if t1_curve_fit.params.x_qp is not None:
        t1_noise_channels.append(
            ("t1_quasiparticle_tunneling", {"x_qp": t1_curve_fit.params.x_qp})
        )
    if t1_curve_fit.params.Q_ind is not None:
        t1_noise_channels.append(
            ("t1_inductive", {"Q_ind": t1_curve_fit.params.Q_ind})
        )
    grid_T1_us = 1e-3 * zf.calculate_eff_t1_vs_flux_fast(
        params,
        grid_fluxs,
        t1_noise_channels,
        t1_curve_fit.params.Temp,
    )
    t1_label = "T1-curve fit"
else:
    order = np.argsort(fit_fluxs)
    grid_T1_us = np.interp(grid_fluxs, fit_fluxs[order], fit_T1_us[order])
    t1_label = "interpolated measured T1"

grid_chi_over_2pi_mhz = dispersive_chi01_over_2pi_mhz(grid_fluxs)
grid_gamma_phi_thermal = thermal_photon_gamma_phi_per_us(
    half_equivalent_n_th,
    kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
    chi_over_2pi_mhz=grid_chi_over_2pi_mhz,
)
grid_T2_relax_us = 2.0 * grid_T1_us
grid_Tphi_flux_us = np.divide(
    1.0,
    grid_gamma_phi,
    out=np.full_like(grid_gamma_phi, np.inf),
    where=grid_gamma_phi > 0.0,
)
grid_Tphi_photon_us = np.divide(
    1.0,
    grid_gamma_phi_thermal,
    out=np.full_like(grid_gamma_phi_thermal, np.inf),
    where=grid_gamma_phi_thermal > 0.0,
)
grid_T2e_effective_us = 1.0 / (
    1.0 / (2.0 * grid_T1_us) + grid_gamma_phi + grid_gamma_phi_thermal
)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(
    sample_fluxs,
    sample_T2e_us,
    "o",
    color="tab:blue",
    label="T2 echo sample",
)
ax.errorbar(
    fit_fluxs,
    fit_T2e_us,
    yerr=fit_T2e_err_us,
    fmt="none",
    ecolor="tab:blue",
    alpha=0.7,
    capsize=3,
)
ax.scatter(
    sample_fluxs,
    2.0 * sample_T1_us,
    s=24,
    color="tab:green",
    alpha=0.65,
    label="2*T1 sample",
)
ax.plot(
    grid_fluxs,
    grid_T2_relax_us,
    "--",
    color="tab:green",
    label=f"2*T1 ({t1_label})",
)
ax.plot(
    grid_fluxs,
    grid_Tphi_flux_us,
    ":",
    color="tab:blue",
    label="pure 1/f flux noise",
)
ax.plot(
    grid_fluxs,
    grid_Tphi_photon_us,
    "-.",
    color="tab:red",
    label=rf"photon shot noise ($n_{{th}}={half_equivalent_n_th:.2e}$)",
)
ax.plot(
    grid_fluxs,
    grid_T2e_effective_us,
    "-",
    color="black",
    linewidth=2.0,
    label="effective: 2*T1 + flux + photons",
)
ax.set_xlabel(r"Flux $\Phi_\mathrm{ext}/\Phi_0$")
ax.set_ylabel("Time (us)")
ax.set_title(
    rf"Echo T2 channel limits, $A_\Phi$ = {flux_noise_fit.A_phi * 1e6:.2f} "
    r"$\mu\Phi_0/\sqrt{Hz}$"
)
ax.grid(True, alpha=0.25)
ax.legend()
ax.set_xlim(0.49, 0.53)
ax.set_ylim(0.0, 130)
fig.tight_layout()
fig.savefig(os.path.join(image_dir, "T2e_flux_noise_fit.png"), dpi=160)
plt.show()
plt.close(fig)
```

# Readout and conclusion

For the current regenerated `samples.csv`:

- `T2e (us)` has 77 finite rows in the full table.
- The current-scale guard selects `current_scale = 1`, confirming that
  `samples.csv` is now consistently in mA.
- Measured f01 is used to calibrate flux before branch coverage and fitting,
  using the same `correct_flux_from_f01` path as `T1_curve.md`.
- The analysis filter keeps only `0.4 <= flux <= 0.6` after f01 correction:
  61 `T2e (us)` rows remain, 58 of them have finite f01, 55 have enough fields
  for sample dephasing, and 53 have full uncertainty columns for the weighted
  fit.
- Within the filtered finite-f01 `T2e (us)` rows, half-flux coverage is
  `below=0`, `at=1`, `above=57` for `flux_half = -11.1 mA`.
- Plots use flux quanta on the x-axis; continuous simulated curves use the
  uniform `t_fluxs` grid over `0.4..0.6`, not the measured sample positions.
- `T2e` samples are shown as one series; rows with finite fit errors get
  error-bar overlays but are not split into a separate no-error series.
- The measured `2*T1` ceiling is shown as scatter in both the direct data plot
  and the final model overlay. The final model overlay is split into four lines:
  `2*T1`, pure first-order flux-noise `Tphi`, pure photon-shot-noise `Tphi`, and
  the effective T2 from adding all three rates.
- The best sample echo point is at corrected flux `0.5000`, with
  `T2e = 30.36 us`, `T1 = 59.10 us`, `2*T1 = 118.20 us`, and
  `Tphi = 40.85 us`. The best point in the weighted-fit subset is around
  corrected flux `0.5021`, with `T2e = 21.50 us`.
- The readout thermal-photon estimate uses `bare_rf = 5.7931 GHz`,
  `g = 73.91 MHz`, and the `T1_curve.md` linewidth assumption
  `kappa/2pi = 4.2 MHz`. At half flux, the dispersive model gives
  `chi01/2pi = 2.915 MHz`, so PRX Eq. A15 gives
  `Gamma_phi = 8.58*n_th 1/us`.
- Matching the half-flux sample's observed
  `Gamma_phi = 1/T2e - 1/(2*T1) = 0.02448 1/us` requires
  `n_th = 2.85e-3`. The weighted-fit peak point requires
  `n_th = 3.93e-3`.
- For the half-flux sample, the thermal-photon-limited `T2` ceiling is
  `118.2 us` at `n_th = 0`, `58.7 us` at `n_th = 1e-3`, `29.2 us` at
  `n_th = 3e-3`, and `10.6 us` at `n_th = 1e-2`.
- The median sample `2*T1/T2e` ratio is about `28`, so this dataset is not
  relaxation-limited; pure dephasing dominates most measured T2e points.
- The first-order 1/f flux-noise fit first subtracts both the relaxation rate
  `1/(2*T1)` and the photon-shot-noise rate `Gamma_photon(flux)`. The remaining
  flux-dephasing target is positive for all 53 weighted-fit rows.
- After photon subtraction, the first-order 1/f flux-noise fit gives
  `A_phi ~= 3.36 uPhi0/sqrt(Hz)` with reduced chi2 about `2.10`, comparable in
  order of magnitude to the PRX paper's `~2 uPhi0/sqrt(Hz)` flux-noise amplitude.
- Figures are saved under `result/Q12_2D[7]/Q4/t2_curve/`, including the new
  `T2e_thermal_photon_limit.png`.

```python
print(f"Images saved under: {image_dir}")
```

```python

```
