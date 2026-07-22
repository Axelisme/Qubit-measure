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

# T1 Curve

Model:

`1/T1_eff = 1/T1_capacitive + 1/T1_quasiparticle + 1/T1_inductive + 1/T1_Purcell`.

# Import

```python
%load_ext autoreload

from IPython.display import display

%autoreload 2
import zcu_tools.notebook.analysis.t1_curve as zt1
```

# Project

```python
figure_paths = {}
```

```python
ctx = zt1.load_t1_curve_context(
    result_dir="../../result/Q12_2D[7]/Q4",
    samples_filename="samples.csv",
    image_dir=None,
    default_bare_rf=5.0,
    default_g=0.1,
)
```

```python
display(ctx.params_table)
display(ctx.samples_preview)
```

# Flux Calibration

```python
cal = zt1.calibrate_t1_flux(
    ctx,
    current_scale_candidates=(1.0, 1000.0),
)
print(f"current scale = {cal.current_scale:g}")
print(f"finite f01 rows = {len(cal.freq_rows)}")
print(f"finite T1 rows = {len(cal.t1_df)}")
```

# Fit Window

```python
analysis_flux_range = (0.49, 1.0)
```

```python
data = zt1.prepare_t1_curve_data(
    cal,
    analysis_flux_range=analysis_flux_range,
    max_abs_flux_correction=0.03,
    max_rel_t1_err=0.25,
    use_weighted_points_only=False,
)
fig, _ = zt1.plot_t1_flux_calibration(data)
figure_paths["flux_calibration"] = zt1.save_t1_curve_figure(fig, ctx, "flux_calibration.png")

fig, _ = zt1.plot_t1_curve_data(data)
figure_paths["t1_samples"] = zt1.save_t1_curve_figure(fig, ctx, "T1s.png")
```

# Purcell Effect

```python
purcell = zt1.PurcellEffectParams(
    kappa_ghz=14.8e-3,
    bare_rf=None,
    g=None,
)
# purcell = None  # Disable Purcell in probe, combined fit, and plots.

print("Purcell enabled =", purcell is not None)
```

# Capacitive-Loss Probe

```python
Temp = 60e-3
```

```python
cap_probe = zt1.analyze_t1_capacitive_limit(
    data,
    Temp=Temp,
    purcell=purcell,
    omega_range=(None, None),
    fit_temperature=False,
    fit_constant=True,
    statistic="median",
    parameter_init=None,
)

print(f"Q_cap probe = {cap_probe.parameter_init:.3e}")
```

```python
fig, _ = zt1.plot_t1_mechanism_probe(cap_probe)
figure_paths["Qcap_vs_omega"] = zt1.save_t1_curve_figure(
    fig, ctx, "Qcap_vs_omega.png"
)

fig, _ = zt1.plot_t1_mechanism_limit(
    cap_probe,
    t_flux_count=1000,
    flux_range=analysis_flux_range,
    purcell=purcell,
)
figure_paths["T1_fit_Qcap"] = zt1.save_t1_curve_figure(
    fig, ctx, "T1s_fit_Qcap.png", bbox_inches="tight"
)
```

# Quasiparticle-Loss Probe

```python
qp_probe = zt1.analyze_t1_quasiparticle_limit(
    data,
    Temp=Temp,
    purcell=purcell,
    omega_range=(6.0, None),
    fit_temperature=False,
    fit_constant=True,
    statistic="median",
    parameter_init=None,
)

print(f"x_qp probe = {qp_probe.parameter_init:.3e}")
```

```python
fig, _ = zt1.plot_t1_mechanism_probe(qp_probe)
figure_paths["Qqp_vs_omega"] = zt1.save_t1_curve_figure(
    fig, ctx, "Qqp_vs_omega.png"
)

fig, _ = zt1.plot_t1_mechanism_limit(
    qp_probe,
    t_flux_count=1000,
    flux_range=analysis_flux_range,
    purcell=purcell,
)
figure_paths["T1_fit_xqp"] = zt1.save_t1_curve_figure(
    fig, ctx, "T1s_fit_xqp.png", bbox_inches="tight"
)
```

# Inductive-Loss Probe

```python
ind_probe = zt1.analyze_t1_inductive_limit(
    data,
    Temp=Temp,
    purcell=purcell,
    omega_range=(None, 4.0),
    fit_temperature=False,
    fit_constant=True,
    statistic="median",
    parameter_init=None,
)

print(f"Q_ind probe = {ind_probe.parameter_init:.3e}")
```

```python
fig, _ = zt1.plot_t1_mechanism_probe(ind_probe)
figure_paths["Qind_vs_omega"] = zt1.save_t1_curve_figure(
    fig, ctx, "Qind_vs_omega.png"
)

fig, _ = zt1.plot_t1_mechanism_limit(
    ind_probe,
    t_flux_count=1000,
    flux_range=analysis_flux_range,
    purcell=purcell,
)
figure_paths["T1_fit_Qind"] = zt1.save_t1_curve_figure(
    fig, ctx, "T1s_fit_Qind.png", bbox_inches="tight"
)
```

# Combined Fit

```python
fit_init = zt1.make_t1_fit_init(
    active_mechanisms=("capacitive", "inductive"),
    Temp=Temp,
    cap_probe=cap_probe,
    qp_probe=qp_probe,
    ind_probe=ind_probe,
    Q_cap=None,
    x_qp=None,
    Q_ind=None,
)
fit_bounds = zt1.make_t1_fit_bounds(
    fit_init,
    factor=100.0,
    Temp_bounds=(10e-3, 300e-3),
    Q_lower_floor=1.0,
    x_qp_lower_floor=1e-12,
    x_qp_upper_cap=1.0,
)
```

```python
combined_fit = zt1.fit_t1_curve(
    data,
    init=fit_init,
    purcell=purcell,
    bounds=fit_bounds,
    fixed=zt1.mechanisms_to_fixed_params(()),
    T1_error_policy=zt1.MeasurementErrorPolicy(
        nan_policy="bin_median",
        relative_floor=0.05,
        fallback_error=1000.0,  # ns
    ),
    flux_weighting=zt1.FluxResidualWeighting(
        mode="equal_flux_bin",
        bin_width=0.01,
        origin=analysis_flux_range[0],
    ),
    residual_mode="log",
    loss="linear",
    max_nfev=10000,
    progress=True,
)

print(zt1.t1_parameter_text(combined_fit.fit_result))
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
channel_analysis = zt1.build_t1_channel_curves(
    combined_fit,
    t_flux_count=1000,
    flux_range=analysis_flux_range,
    purcell=purcell,
)
```

```python
fig, _ = zt1.plot_t1_channel_analysis(channel_analysis)
figure_paths["channel_overlay"] = zt1.save_t1_curve_figure(
    fig, ctx, "T1s_fit_eff.png", bbox_inches="tight"
)
```

# Writeback

```python
zt1.write_t1_curve_fit(combined_fit)
```

# Result Object

```python
analysis = zt1.collect_t1_curve_result(
    context=ctx,
    calibration=cal,
    data=data,
    cap_probe=cap_probe,
    qp_probe=qp_probe,
    ind_probe=ind_probe,
    combined_fit=combined_fit,
    channel_analysis=channel_analysis,
    figure_paths=figure_paths,
)
analysis.figure_paths
```
