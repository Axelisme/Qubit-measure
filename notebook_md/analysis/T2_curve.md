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

# T2 Curve

Model:

`1/T2e = 1/(2*T1) + sqrt(ln 2) * A_phi * |domega01/dPhi| + n_th * kappa * chi^2/(kappa^2 + chi^2)`.

# Import

```python
%load_ext autoreload

from IPython.display import display

%autoreload 2
import zcu_tools.notebook.analysis.t2_curve as zt2
```

# Project

```python
chip_name = "Q12_2D[7]"
qub_name = "Q4"

result_dir = f"../../result/{chip_name}/{qub_name}"
samples_filename = "samples.csv"
image_dir = None  # None: use result_dir/t2_curve.
default_bare_rf = 5.0

figure_paths = {}
```

```python
ctx = zt2.load_t2_curve_context(
    result_dir=result_dir,
    samples_filename=samples_filename,
    image_dir=image_dir,
    default_bare_rf=default_bare_rf,
)
```

```python
display(ctx.params_table)
display(ctx.samples_preview)
```

# Flux Calibration

```python
current_scale_candidates = (1.0, 1000.0)
```

```python
cal = zt2.calibrate_t2_flux(
    ctx,
    current_scale_candidates=current_scale_candidates,
)
```

```python
print(f"current scale = {cal.current_scale:g}")
print(f"finite f01 rows = {len(cal.freq_rows)}")
```

# Fit Window

```python
analysis_flux_range = (0.49, 0.53)
max_abs_flux_correction = 0.03
max_rel_t2e_err = 0.5
use_weighted_points_only = False

T1_error_policy = zt2.MeasurementErrorPolicy(
    nan_policy="bin_median",
    fallback_error=1.0,
)
T2e_error_policy = zt2.MeasurementErrorPolicy(
    nan_policy="bin_median",
    absolute_floor=0.2,
    relative_floor=0.05,
)
flux_weighting = zt2.FluxResidualWeighting(
    mode="equal_flux_bin",
    bin_width=0.002,
    origin=analysis_flux_range[0],
)
```

```python
data = zt2.prepare_t2_dephasing_data(
    cal,
    analysis_flux_range=analysis_flux_range,
    max_abs_flux_correction=max_abs_flux_correction,
    max_rel_t2e_err=max_rel_t2e_err,
    use_weighted_points_only=use_weighted_points_only,
)
```

```python
fig, _ = zt2.plot_t2_flux_calibration(data)
figure_paths["flux_calibration"] = zt2.save_t2_curve_figure(
    fig, ctx, "flux_calibration.png"
)

fig, _ = zt2.plot_t2_dephasing_data(data)
figure_paths["t2e_vs_flux"] = zt2.save_t2_curve_figure(fig, ctx, "T2e_vs_flux.png")
```

# Flux-Noise-Only Probe

```python
readout_kappa_over_2pi_mhz = 14.75412896809815

flux_assumed_n_th = 0.0
flux_probe_statistic = "median"
flux_A_phi_init = None
flux_A_phi_bounds = None
flux_probe_min_sensitivity_fraction = 1e-3
```

```python
flux_probe = zt2.analyze_flux_noise_limit(
    data,
    readout_kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
    assumed_n_th=flux_assumed_n_th,
    A_phi_init=flux_A_phi_init,
    A_phi_bounds=flux_A_phi_bounds,
    statistic=flux_probe_statistic,
    min_sensitivity_fraction=flux_probe_min_sensitivity_fraction,
    T1_error_policy=T1_error_policy,
    T2_error_policy=T2e_error_policy,
    flux_weighting=flux_weighting,
    residual_mode="gamma",
    loss="linear",
    max_nfev=10000,
    progress=True,
)
```

```python
print(f"A_phi probe = {1e6 * flux_probe.A_phi_fit:.3f} uPhi0/sqrtHz")
print(f"A_phi pointwise upper = {1e6 * flux_probe.A_phi_upper:.3f} uPhi0/sqrtHz")

fig, _ = zt2.plot_flux_noise_probe(flux_probe)
figure_paths["flux_noise_probe"] = zt2.save_t2_curve_figure(
    fig, ctx, "Gamma_phi_vs_flux_sensitivity.png"
)
```

# Photon-Shot-Noise-Only Probe

```python
photon_assumed_A_phi = 0.0
photon_probe_statistic = "min"
photon_n_th_init = None
photon_n_th_bounds = None
thermal_probe_n_th = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
```

```python
photon_probe = zt2.analyze_photon_shot_noise_limit(
    data,
    readout_kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
    assumed_A_phi=photon_assumed_A_phi,
    n_th_init=photon_n_th_init,
    n_th_bounds=photon_n_th_bounds,
    statistic=photon_probe_statistic,
    thermal_probe_n_th=thermal_probe_n_th,
    T1_error_policy=T1_error_policy,
    T2_error_policy=T2e_error_policy,
    flux_weighting=flux_weighting,
    residual_mode="gamma",
    loss="linear",
    max_nfev=10000,
    progress=True,
)
```

```python
print(f"n_th probe = {photon_probe.n_th_init:.3e}")
print(f"n_th photon-only fit = {photon_probe.n_th_fit:.3e}")
print(f"half-flux equivalent n_th = {photon_probe.thermal.half_equivalent_n_th:.3e}")

fig, _ = zt2.plot_photon_shot_noise_probe(photon_probe)
figure_paths["photon_shot_noise_probe"] = zt2.save_t2_curve_figure(
    fig, ctx, "T2e_thermal_photon_limit.png"
)

fig, _ = zt2.plot_nth_limit_vs_flux(photon_probe)
figure_paths["n_th_vs_flux"] = zt2.save_t2_curve_figure(fig, ctx, "n_th_vs_flux.png")
```

# Combined Fit

```python
active_mechanisms = ("flux_noise", "photon_shot_noise")
fixed_mechanisms = ()

fit_A_phi_override = None
fit_n_th_override = None

residual_mode = "gamma"
loss = "linear"
max_nfev = 10000
```

```python
fit_init = zt2.make_t2_fit_init(
    active_mechanisms=active_mechanisms,
    flux_probe=flux_probe,
    photon_probe=photon_probe,
    A_phi=fit_A_phi_override,
    n_th=fit_n_th_override,
)
fit_bounds = zt2.make_t2_fit_bounds(
    fit_init,
    factor=1000.0,
    A_phi_lower_floor=1e-8,
    n_th_lower_floor=1e-8,
    A_phi_upper_cap=1e-4,
    n_th_upper_cap=1e-1,
)
fixed = zt2.mechanisms_to_fixed_params(fixed_mechanisms)
```

```python
combined_fit = zt2.fit_t2_curve(
    data,
    readout_kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
    init=fit_init,
    bounds=fit_bounds,
    fixed=fixed,
    residual_mode=residual_mode,
    loss=loss,
    max_nfev=max_nfev,
    progress=True,
    T1_error_policy=T1_error_policy,
    T2_error_policy=T2e_error_policy,
    flux_weighting=flux_weighting,
)

print(zt2.t2_parameter_text(combined_fit.fit_result))
print("fixed =", combined_fit.fit_result.fixed)
print("free =", combined_fit.fit_result.free)
print("flux weighting =", combined_fit.fit_result.flux_weights.mode)
print(
    "effective flux bins =",
    f"{combined_fit.fit_result.flux_weights.effective_observation_count:g}",
)
```

# Channel Curves

```python
t_flux_count = 1000
plot_flux_range = analysis_flux_range
use_t1_curve_fit = True
```

```python
channel_analysis = zt2.build_t2_channel_curves(
    combined_fit,
    t_flux_count=t_flux_count,
    flux_range=plot_flux_range,
    use_t1_curve_fit=use_t1_curve_fit,
)
```

```python
fig, _ = zt2.plot_t2_channel_analysis(channel_analysis)
figure_paths["channel_overlay"] = zt2.save_t2_curve_figure(
    fig, ctx, "T2e_flux_noise_fit.png", bbox_inches="tight"
)
```

# Result Object

```python
analysis = zt2.collect_t2_curve_result(
    context=ctx,
    calibration=cal,
    data=data,
    flux_probe=flux_probe,
    photon_probe=photon_probe,
    combined_fit=combined_fit,
    channel_analysis=channel_analysis,
    figure_paths=figure_paths,
)
```

```python
analysis.figure_paths
```
