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
    display_name: Python 3
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

# T1 curve with dynamic temperature

**Readers:** operator, lab-author. Fit T1 channels with a fixed frequency-dependent temperature profile and inspect parameter identifiability.

Intrinsic channels use `Eff_T(f01)` at each point. Purcell uses the fixed
`Eff_T(bare_rf)`. The author assigns these temperatures before calling the
shared fitter. The fitter does not read the temperature file or infer its units.
The top-fraction Q estimate below is a separate author analysis, not a fit constraint.

# Import

```python
%load_ext autoreload

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from zcu_tools.resources.qubit_params import QubitParams
from zcu_tools.resources.record_table import RecordFluxFrame, flux_from_frame
from zcu_tools.simulate.fluxonium import calculate_eff_t1_vs_flux_fast

%autoreload 2
import zcu_tools.notebook.analysis.fit_tools as zfit
import zcu_tools.notebook.analysis.t1_curve as zt1
from zcu_tools.notebook.analysis.t1_curve.purcell import (
    combine_t1_limits,
    relaxation_rates_from_t1,
)
```

# Project and author flux

```python
result_dir = "../../result/Q12_2D[7]/Q4"
samples_filename = "samples.csv"
image_dir = "../../result/Q12_2D[7]/Q4/t1_curve_dynamicT"
figure_paths = {}

samples = pd.read_csv(Path(result_dir) / samples_filename)
params = QubitParams(Path(result_dir) / "params.json", readonly=True).require_fluxdep_fit()
fallback_frame = None
# Explicitly supply a frame only for rows that have none:
# fallback_frame = RecordFluxFrame(
#     dev_unit="A", flux_int=params.flux_int, flux_period=params.flux_period
# )
fluxs = flux_from_frame(
    samples, device_name="bias_device", fallback_frame=fallback_frame
)
```

# Optional f01 correction

Replace the flux reconstruction above with your own row-aligned flux array if
needed. Correction is an explicit author choice. Missing observed frequencies
use the model f01; the analysis stage does not infer preprocessing history.

```python
apply_f01_correction = False
max_abs_flux_correction = 0.03
if apply_f01_correction:
    observed_ghz = (
        1e-3 * samples["freq_mhz"].to_numpy(dtype=np.float64)
        if "freq_mhz" in samples.columns else None
    )
    correction = zfit.correct_flux_from_f01(
        fluxs, observed_ghz, params.params,
        max_abs_flux_correction=max_abs_flux_correction,
    )
    display(pd.DataFrame({
        "author flux": correction.raw_fluxs,
        "corrected flux": correction.corrected_fluxs,
        "accepted": correction.accepted,
        "applied correction": correction.applied_flux_corrections,
    }))
    fluxs = correction.corrected_fluxs
```

# Load and select fit window

```python
ctx = zt1.load_t1_curve_context(
    result_dir=result_dir,
    samples_filename=samples_filename,
    fluxs=fluxs,
    image_dir=image_dir,
)
display(ctx.params_table)
display(ctx.samples_preview)
cal = zt1.calibrate_t1_flux(ctx)
analysis_flux_range = (0.48, 1.02)
data = zt1.prepare_t1_curve_data(
    cal,
    analysis_flux_range=analysis_flux_range,
    max_rel_t1_err=0.25,
    use_weighted_points_only=False,
)
display(data.summary_table)

fig, _ = zt1.plot_t1_flux_calibration(data)
figure_paths["flux_calibration"] = zt1.save_t1_curve_figure(
    fig, ctx, "flux_calibration.png", bbox_inches="tight"
)
fig, _ = zt1.plot_t1_curve_data(data)
figure_paths["t1_samples"] = zt1.save_t1_curve_figure(fig, ctx, "T1s.png")
```

# Optional device axis

Normalized-flux plots do not require a device frame. The author may add a
second axis with the chosen device's A, V or native values.

```python
show_device_axis = False
device_axis_name = "bias_device"
device_axis_frame = None
if show_device_axis:
    device_axis_frame = RecordFluxFrame(
        dev_unit="A", flux_int=params.flux_int, flux_period=params.flux_period
    )
```

# Purcell effect

```python
purcell_kappa_ghz = 14.8e-3
purcell = zt1.load_t1_purcell_params(ctx, kappa_ghz=purcell_kappa_ghz)
# purcell = None
print("Purcell enabled =", purcell is not None)
if purcell is not None:
    print(f"bare_rf = {purcell.bare_rf:.6g} GHz")
    print(f"g = {purcell.g:.6g} GHz")
    print(f"kappa = {purcell.kappa_ghz:.6g} GHz")
```

# Effective temperature assignment

The NPZ stores `fpts` in Hz and `T_effs` in K. This author helper filters
invalid calibration points, sorts by frequency and clips interpolation to
the endpoints with a warning. Analysis uses the prepared f01 values, after
selection and integer-period alignment.

```python
eff_temp_npz = Path("../../result/Eff_T/Eff_T_vs_Freq.npz")


def load_eff_temperature_curve(path):
    with np.load(path) as raw:
        freqs_hz = np.asarray(raw["fpts"], dtype=np.float64)
        Temps_K = np.asarray(raw["T_effs"], dtype=np.float64)
    if freqs_hz.ndim != 1 or Temps_K.ndim != 1 or freqs_hz.shape != Temps_K.shape:
        raise ValueError("fpts and T_effs must be matching 1D arrays")
    finite = np.isfinite(freqs_hz) & np.isfinite(Temps_K) & (freqs_hz > 0) & (Temps_K > 0)
    if not np.any(finite):
        raise ValueError("effective-temperature curve has no finite positive points")
    order = np.argsort(freqs_hz[finite])
    return freqs_hz[finite][order], Temps_K[finite][order]


def interp_eff_T_from_freq_hz(freq_hz, freqs_hz, Temps_K):
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    clipped = (freq_hz < freqs_hz[0]) | (freq_hz > freqs_hz[-1])
    if np.any(clipped):
        print(f"warning: {np.count_nonzero(clipped)} points clipped to Eff_T range")
    return np.interp(freq_hz, freqs_hz, Temps_K)


eff_freqs_hz, eff_Temps_K = load_eff_temperature_curve(eff_temp_npz)
sample_Temp_K = interp_eff_T_from_freq_hz(data.sample.f01_mhz * 1e6, eff_freqs_hz, eff_Temps_K)
fit_Temp_K = interp_eff_T_from_freq_hz(data.fit.f01_mhz * 1e6, eff_freqs_hz, eff_Temps_K)
purcell_Temp_K = None
if purcell is not None:
    purcell_Temp_K = float(interp_eff_T_from_freq_hz(
        np.asarray([purcell.bare_rf * 1e9]), eff_freqs_hz, eff_Temps_K,
    )[0])
print(f"fit-point Eff_T range = {fit_Temp_K.min() * 1e3:.2f} - {fit_Temp_K.max() * 1e3:.2f} mK")
if purcell_Temp_K is not None:
    print(f"Purcell Temp = Eff_T(bare_rf) = {purcell_Temp_K * 1e3:.2f} mK")
```

```python
fig, ax = plt.subplots(constrained_layout=True, figsize=(7.5, 4.0))
ax.plot(eff_freqs_hz * 1e-9, eff_Temps_K * 1e3, color="black", label="Eff_T(f)")
ax.scatter(data.sample.f01_mhz * 1e-3, sample_Temp_K * 1e3, s=18, alpha=0.45, label="sample points")
ax.scatter(data.fit.f01_mhz * 1e-3, fit_Temp_K * 1e3, s=28, facecolors="none", edgecolors="tab:red", label="fit points")
if purcell is not None:
    ax.axvline(purcell.bare_rf, color="tab:purple", linestyle="--", label="bare_rf")
ax.set_xlabel("f01 (GHz)")
ax.set_ylabel("Eff_T (mK)")
ax.grid()
ax.legend(fontsize="small")
figure_paths["effective_temperature_assignment"] = zt1.save_t1_curve_figure(
    fig, ctx, "Eff_T_assignment.png", bbox_inches="tight"
)
```

# Fit settings and fixed Purcell rates

The shared parameter whitelist selects the intrinsic channels. The scalar
`init.Temp` remains the callback's reference value; `fixed_temperatures`
replaces it for intrinsic T1 and excludes Temp from covariance. Here the
callback closes over fixed Purcell rates and does not use `init.Temp`.

```python
active_mechanisms = ("capacitive", "quasiparticle", "inductive")
fixed_mechanisms = ()
fit_init = {"Q_cap": 1.0e6, "x_qp": 1.0e-8, "Q_ind": 1.0e8}
fit_bounds = {
    "Q_cap": (1.0, 1.0e10),
    "x_qp": (1.0e-12, 1.0),
    "Q_ind": (1.0, 1.0e12),
}
init = zt1.make_t1_fit_init(
    active_mechanisms=active_mechanisms, Temp=0.06, **fit_init,
)
T1_error_policy = zfit.MeasurementErrorPolicy(
    nan_policy="bin_median", relative_floor=0.05, fallback_error=1000.0,
)
flux_weighting = zfit.FluxResidualWeighting(
    mode="equal_flux_bin", bin_width=0.01, origin=analysis_flux_range[0],
)
residual_mode = "log"
loss = "linear"
max_nfev = 1000
Q_lower_top_fraction = 0.10

t_flux_count = 250
t_fluxs = np.linspace(*analysis_flux_range, t_flux_count)
t_f01_mhz = zfit.predict_f01_mhz(ctx.params, t_fluxs)
t_Temp_K = interp_eff_T_from_freq_hz(t_f01_mhz * 1e6, eff_freqs_hz, eff_Temps_K)
fit_purcell_T1_ns = grid_purcell_T1_ns = None
fit_extra_rates = np.zeros_like(data.fit.fluxs)
if purcell is not None:
    fit_purcell_T1_ns = zt1.calculate_purcell_t1_limit(
        ctx, data.fit.fluxs, purcell, Temp=purcell_Temp_K,
    )
    grid_purcell_T1_ns = zt1.calculate_purcell_t1_limit(
        ctx, t_fluxs, purcell, Temp=purcell_Temp_K,
    )
    fit_extra_rates = relaxation_rates_from_t1(fit_purcell_T1_ns, name="Purcell T1")
```

# Author Q lower-bound estimate and mechanism plots

Subtract the fixed Purcell rate from observed T1. Ignore points with
nonpositive intrinsic rate. For each mechanism, take the mean of the highest
10% of pointwise Q values, rounding the count upward. Quasiparticle Q is
`1/x_qp`. These author estimates do not change combined-fit initial values or
bounds. If Purcell exceeds every observed rate, this estimate has no valid
points and raises instead of changing the physics settings.

```python
if not np.isfinite(Q_lower_top_fraction) or not 0 < Q_lower_top_fraction <= 1:
    raise ValueError("Q_lower_top_fraction must be in (0, 1]")
target_T1_ns = data.fit.T1_ns.copy()
valid_intrinsic = np.isfinite(target_T1_ns) & (target_T1_ns > 0)
if fit_purcell_T1_ns is not None:
    correction = zt1.subtract_relaxation_limit(
        data.fit.T1_ns, data.fit.T1err_ns, fit_purcell_T1_ns,
    )
    target_T1_ns = correction.intrinsic_T1_ns
    valid_intrinsic = correction.valid_mask

mechanism_rows = []
mechanism_estimates = {}
for mechanism in active_mechanisms:
    reference = zt1.make_t1_fit_init(
        active_mechanisms=(mechanism,), Temp=init.Temp, **fit_init,
    )
    channels = zt1.t1_noise_channels_from_params(reference)
    param_name = zt1.mechanisms_to_fixed_params((mechanism,))[0]
    reference_value = fit_init[param_name]
    reference_Q = 1 / reference_value if param_name == "x_qp" else reference_value
    reference_T1_ns = calculate_eff_t1_vs_flux_fast(
        ctx.params, data.fit.fluxs, channels, fit_Temp_K,
    )
    valid = valid_intrinsic & np.isfinite(reference_T1_ns) & (reference_T1_ns > 0)
    if not np.any(valid):
        raise ValueError(f"{mechanism} has no valid points for Q lower-bound estimate")
    pointwise_Q = reference_Q * target_T1_ns[valid] / reference_T1_ns[valid]
    top_count = max(1, int(np.ceil(pointwise_Q.size * Q_lower_top_fraction)))
    Q_lower = float(np.mean(np.sort(pointwise_Q)[-top_count:]))
    estimate = 1 / Q_lower if param_name == "x_qp" else Q_lower
    mechanism_estimates[mechanism] = estimate
    mechanism_rows.append({
        "mechanism": mechanism, "parameter": param_name, "value": estimate,
        "Q lower bound": Q_lower, "top fraction": Q_lower_top_fraction,
        "top count": top_count, "valid points": int(np.count_nonzero(valid)),
    })
    estimated_params = zt1.make_t1_fit_init(
        active_mechanisms=(mechanism,), Temp=init.Temp, **{param_name: estimate},
    )
    intrinsic = calculate_eff_t1_vs_flux_fast(
        ctx.params, t_fluxs, zt1.t1_noise_channels_from_params(estimated_params), t_Temp_K,
    )
    components = {mechanism: intrinsic}
    effective = intrinsic
    if grid_purcell_T1_ns is not None:
        components["Purcell"] = grid_purcell_T1_ns
        effective = combine_t1_limits(intrinsic, grid_purcell_T1_ns)
    fig, ax = zt1.plot_eff_t1_with_sample(
        data.sample.fluxs, data.sample.T1_ns, data.sample.T1err_ns,
        effective, t_fluxs, label=f"{mechanism} lower bound",
        component_t1s=components,
        parameter_text=f"{param_name} = {estimate:.3e}\nTemp = Eff_T(f01)\nQ lower = {Q_lower:.3e}\nconstraint = top {Q_lower_top_fraction:.0%} mean",
    )
    if device_axis_frame is not None:
        zfit.add_device_axis(ax, device_axis_frame, device_name=device_axis_name)
    figure_paths[f"T1_fit_{param_name}"] = zt1.save_t1_curve_figure(
        fig, ctx, f"T1s_fit_{param_name}_profile_Temp.png", bbox_inches="tight"
    )
display(pd.DataFrame(mechanism_rows))
```

# Combined fit and identifiability

```python
profile_fit = zt1.fit_t1_noise_params(
    data.fit.fluxs, data.fit.T1_ns, ctx.params,
    init=init, fixed_temperatures=fit_Temp_K,
    bounds={name: fit_bounds[name] for name in zt1.mechanisms_to_fixed_params(active_mechanisms)},
    fixed=zt1.mechanisms_to_fixed_params(fixed_mechanisms),
    T1errs=data.fit.T1err_ns, T1_error_policy=T1_error_policy,
    flux_weighting=flux_weighting, residual_mode=residual_mode,
    loss=loss, max_nfev=max_nfev,
    extra_relaxation_rate_fn=lambda current: fit_extra_rates,
)
print("success =", profile_fit.success)
print(profile_fit.message)
print("free =", profile_fit.free)
print("fixed =", profile_fit.fixed)
print("effective flux bins =", profile_fit.flux_weights.effective_observation_count)
parameter_values = {
    "Q_cap": profile_fit.params.Q_cap,
    "x_qp": profile_fit.params.x_qp,
    "Q_ind": profile_fit.params.Q_ind,
}
display(pd.DataFrame([
    {"parameter": name, "value": value}
    for name, value in parameter_values.items() if value is not None
]))
weak = zfit.find_weak_parameters(profile_fit.correlation)
display(zfit.make_weak_parameter_table(weak))
if profile_fit.correlation is not None:
    fig, _ = zfit.plot_fit_correlation(profile_fit.correlation)
    figure_paths["correlation"] = zt1.save_t1_curve_figure(
        fig, ctx, "fit_correlation.png", bbox_inches="tight"
    )
else:
    print("No covariance available: inspect fit status and free parameters.")
```

# Channel curves

```python
intrinsic = calculate_eff_t1_vs_flux_fast(
    ctx.params, t_fluxs, zt1.t1_noise_channels_from_params(profile_fit.params), t_Temp_K,
)
component_T1s = {}
for mechanism in active_mechanisms:
    component_params = zt1.make_t1_fit_init(
        active_mechanisms=(mechanism,), Temp=init.Temp, **parameter_values,
    )
    component_T1s[mechanism] = calculate_eff_t1_vs_flux_fast(
        ctx.params, t_fluxs, zt1.t1_noise_channels_from_params(component_params), t_Temp_K,
    )
grid_effective_T1_ns = intrinsic
if grid_purcell_T1_ns is not None:
    component_T1s["Purcell"] = grid_purcell_T1_ns
    grid_effective_T1_ns = combine_t1_limits(intrinsic, grid_purcell_T1_ns)
parameter_lines = [
    f"{name} = {value:.3e}" + (" fixed" if name in profile_fit.fixed else "")
    for name, value in parameter_values.items() if value is not None
]
parameter_lines.append("Temp = Eff_T(f01)")
if purcell_Temp_K is not None:
    parameter_lines.append(f"Purcell Temp = {purcell_Temp_K * 1e3:.2f} mK")
fig, ax = zt1.plot_eff_t1_with_sample(
    data.sample.fluxs, data.sample.T1_ns, data.sample.T1err_ns,
    grid_effective_T1_ns, t_fluxs, label="effective",
    component_t1s=component_T1s, parameter_text="\n".join(parameter_lines),
)
if device_axis_frame is not None:
    zfit.add_device_axis(ax, device_axis_frame, device_name=device_axis_name)
figure_paths["channel_overlay"] = zt1.save_t1_curve_figure(
    fig, ctx, "T1s_fit_eff_profile_Temp.png", bbox_inches="tight"
)
```

# Figures

```python
figure_paths
```
