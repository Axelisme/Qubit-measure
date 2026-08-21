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
)
```

```python
display(ctx.params_table)
display(ctx.samples_preview)
```

# Flux Resolution (v2)

```python
cal = zt1.calibrate_t1_flux(
    ctx,
    fallback_frame_unit="A",  # params.json fluxdep_fit frame declared in A
)
print(f"explicit flux rows = {sum(s == 'explicit' for s in cal.resolution.sources)}")
print(f"row-frame flux rows = {sum(s == 'row-frame' for s in cal.resolution.sources)}")
print(f"fallback-frame flux rows = {sum(s == 'fallback-frame' for s in cal.resolution.sources)}")
print(f"finite f01 rows = {len(cal.freq_rows)}")
print(f"finite T1 rows = {len(cal.t1_df)}")
```

# Fit Window

```python
analysis_flux_range = (0.48, 1.02)

Temp_lower = 56.6e-3
```

```python
data = zt1.prepare_t1_curve_data(
    cal,
    analysis_flux_range=analysis_flux_range,
    max_abs_flux_correction=0.03,
    max_rel_t1_err=0.25,
    use_weighted_points_only=False,
)
# fig, _ = zt1.plot_t1_flux_calibration(data)
# figure_paths["flux_calibration"] = zt1.save_t1_curve_figure(fig, ctx, "flux_calibration.png")

fig, _ = zt1.plot_t1_curve_data(data)
figure_paths["t1_samples"] = zt1.save_t1_curve_figure(fig, ctx, "T1s.png")
```

```python
display(data.summary_table)  # prepared frequency-source and branch-shift diagnostics
```

# Purcell Effect

```python
purcell_kappa_ghz = 14.8e-3  # GHz

purcell = zt1.load_t1_purcell_params(ctx, kappa_ghz=purcell_kappa_ghz)
# purcell = None  # Disable Purcell in probe, combined fit, and plots.

print("Purcell enabled =", purcell is not None)
if purcell is not None:
    print(f"bare_rf = {purcell.bare_rf:.6g} GHz")
    print(f"g = {purcell.g:.6g} GHz")
    print(f"kappa = {purcell.kappa_ghz:.6g} GHz")
```

```python
purcell_Temp_upper = 150e-3
if purcell is not None:
    purcell_Temp_upper = zt1.estimate_purcell_Temp_upper_bound(
        data,
        purcell,
        Temp_range=(20e-3, 150e-3),
        tolerance=1e-3,
        max_iter=12,
        progress=True,
    )

print(f"Purcell Temp upper = {purcell_Temp_upper * 1e3:.2f} mK")

if purcell is not None:
    fig, _ = zt1.plot_purcell_Temp_upper_bound(
        data,
        purcell,
        Temp=purcell_Temp_upper,
        t_flux_count=1000,
        flux_range=analysis_flux_range,
    )
    figure_paths["purcell_temp_upper"] = zt1.save_t1_curve_figure(
        fig, ctx, "Purcell_Temp_upper.png", bbox_inches="tight"
    )
```

# Capacitive-Loss Probe

```python
probe_Temp_bounds = (Temp_lower, purcell_Temp_upper)
Temp = 60e-3
```

```python
cap_probe = zt1.analyze_t1_capacitive_limit(
    data,
    Temp=Temp,
    purcell=purcell,
    omega_range=(None, None),
    # fit_temperature=True,
    Temp_bounds=probe_Temp_bounds,
    fit_constant=True,
    statistic="median",
    parameter_init=None,
)
fig, _ = zt1.plot_t1_mechanism_probe(cap_probe)
figure_paths["Qcap_vs_omega"] = zt1.save_t1_curve_figure(
    fig, ctx, "Qcap_vs_omega.png"
)

print(f"Q_cap probe = {cap_probe.parameter_init:.3e}")
```

```python
fig, _ = zt1.plot_t1_mechanism_dipole(cap_probe)
figure_paths["T1_vs_dipole_cap"] = zt1.save_t1_curve_figure(
    fig, ctx, "T1s_vs_|d01|_cap.png"
)
```

```python
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
    # fit_temperature=True,
    Temp_bounds=probe_Temp_bounds,
    fit_constant=True,
    statistic="median",
    parameter_init=None,
)
fig, _ = zt1.plot_t1_mechanism_probe(qp_probe)
figure_paths["Qqp_vs_omega"] = zt1.save_t1_curve_figure(
    fig, ctx, "Qqp_vs_omega.png"
)

print(f"x_qp probe = {qp_probe.parameter_init:.3e}")
```

```python
fig, _ = zt1.plot_t1_mechanism_dipole(qp_probe)
figure_paths["T1_vs_dipole_qp"] = zt1.save_t1_curve_figure(
    fig, ctx, "T1s_vs_|d01|_qp.png"
)
```

```python
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
    # fit_temperature=True,
    Temp_bounds=probe_Temp_bounds,
    fit_constant=True,
    statistic="median",
    parameter_init=None,
)
fig, _ = zt1.plot_t1_mechanism_probe(ind_probe)
figure_paths["Qind_vs_omega"] = zt1.save_t1_curve_figure(
    fig, ctx, "Qind_vs_omega.png"
)

print(f"Q_ind probe = {ind_probe.parameter_init:.3e}")
```

```python
fig, _ = zt1.plot_t1_mechanism_dipole(ind_probe)
figure_paths["T1_vs_dipole_ind"] = zt1.save_t1_curve_figure(
    fig, ctx, "T1s_vs_|d01|_ind.png"
)
```

```python
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
active_mechanisms = ("capacitive", "inductive")
# fix_mechanisms = ("Temp",)
fix_mechanisms = ()

fit_init = zt1.make_t1_fit_init(
    active_mechanisms=active_mechanisms,
    Temp=Temp,
    # Temp=60e-3,
    cap_probe=cap_probe,
    qp_probe=qp_probe,
    ind_probe=ind_probe,
)
fit_bounds = zt1.make_t1_fit_bounds(
    fit_init,
    factor=100.0,
    Temp_bounds=probe_Temp_bounds,
    Q_lower_floor=1.0,
    x_qp_lower_floor=1e-12,
    x_qp_upper_cap=1.0,
)

combined_fit = zt1.fit_t1_curve(
    data,
    init=fit_init,
    purcell=purcell,
    bounds=fit_bounds,
    fixed=zt1.mechanisms_to_fixed_params(fix_mechanisms),
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

fig, _ = zt1.plot_t1_channel_analysis(channel_analysis)
figure_paths["channel_overlay"] = zt1.save_t1_curve_figure(
    fig, ctx, "T1s_fit_eff.png", bbox_inches="tight"
)
```

# Writeback

```python
zt1.write_t1_curve_fit(combined_fit)
```

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

```python

```
