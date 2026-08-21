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

# T1 Curve 2

Use `Eff_T(f01)` as a fixed temperature profile for intrinsic T1 channels.
Purcell uses a fixed `Eff_T(bare_rf)`.

# Import

```python
%load_ext autoreload

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.optimize import least_squares

%autoreload 2
import zcu_tools.notebook.analysis.fit_tools as zfit
import zcu_tools.notebook.analysis.t1_curve as zt1
from zcu_tools.simulate.fluxonium import (
    calculate_eff_t1_vs_flux_fast,
    calculate_purcell_t1_vs_flux,
)
```

# Project

```python
figure_paths = {}
```

```python
ctx = zt1.load_t1_curve_context(
    result_dir="../../result/Q12_2D[7]/Q4",
    samples_filename="samples.csv",
    image_dir="../../result/Q12_2D[7]/Q4/t1_curve2",
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
figure_paths["flux_calibration"] = zt1.save_t1_curve_figure(
    fig, ctx, "flux_calibration.png", bbox_inches="tight"
)

fig, _ = zt1.plot_t1_curve_data(data)
figure_paths["t1_samples"] = zt1.save_t1_curve_figure(fig, ctx, "T1s.png")
```

```python
display(data.summary_table)  # prepared frequency-source and branch-shift diagnostics
```

# Purcell Effect

```python
purcell_kappa_ghz = 14.8e-3

purcell = zt1.load_t1_purcell_params(ctx, kappa_ghz=purcell_kappa_ghz)
# purcell = None
```

```python
print("Purcell enabled =", purcell is not None)
if purcell is not None:
    print(f"bare_rf = {purcell.bare_rf:.6g} GHz")
    print(f"g = {purcell.g:.6g} GHz")
    print(f"kappa = {purcell.kappa_ghz:.6g} GHz")
```

# Effective Temperature

```python
eff_temp_npz = Path("../../result/Eff_T/Eff_T_vs_Freq.npz")
```

```python
def load_eff_temperature_curve(path):
    raw = np.load(path)
    freqs_hz = np.asarray(raw["fpts"], dtype=np.float64)
    Temps_K = np.asarray(raw["T_effs"], dtype=np.float64)
    if freqs_hz.ndim != 1 or Temps_K.ndim != 1 or freqs_hz.shape != Temps_K.shape:
        raise ValueError("fpts and T_effs must be matching 1D arrays")
    finite = np.isfinite(freqs_hz) & np.isfinite(Temps_K) & (freqs_hz > 0.0) & (Temps_K > 0.0)
    if not np.any(finite):
        raise ValueError("effective-temperature curve has no finite positive points")
    order = np.argsort(freqs_hz[finite])
    return freqs_hz[finite][order], Temps_K[finite][order]


def interp_eff_T_from_freq_hz(freq_hz, freqs_hz, Temps_K, *, name="frequency"):
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    clipped = (freq_hz < freqs_hz[0]) | (freq_hz > freqs_hz[-1])
    if np.any(clipped):
        print(f"warning: {np.count_nonzero(clipped)} {name} points clipped to Eff_T range")
    return np.interp(freq_hz, freqs_hz, Temps_K)


def interp_eff_T_from_f01_mhz(f01_mhz, freqs_hz, Temps_K):
    return interp_eff_T_from_freq_hz(
        np.asarray(f01_mhz, dtype=np.float64) * 1e6,
        freqs_hz,
        Temps_K,
        name="f01",
    )


def plot_eff_T_assignment(
    data,
    freqs_hz,
    Temps_K,
    sample_Temp_K,
    fit_Temp_K,
    *,
    bare_rf_ghz=None,
):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(7.5, 4.0))
    ax.plot(freqs_hz * 1e-9, Temps_K * 1e3, color="black", linewidth=1.8, label="Eff_T(f)")
    if bare_rf_ghz is not None:
        ax.axvline(
            bare_rf_ghz,
            color="tab:purple",
            linestyle="--",
            linewidth=1.6,
            label=f"bare_rf = {bare_rf_ghz:.4g} GHz",
        )
    ax.scatter(
        data.sample.f01_mhz * 1e-3,
        sample_Temp_K * 1e3,
        s=18,
        alpha=0.45,
        label="sample points",
    )
    ax.scatter(
        data.fit.f01_mhz * 1e-3,
        fit_Temp_K * 1e3,
        s=28,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.0,
        label="fit points",
    )
    ax.set_xlabel("f01 (GHz)")
    ax.set_ylabel("Eff_T (mK)")
    ax.grid()
    ax.legend(fontsize="small")
    return fig, ax
```

```python
eff_freqs_hz, eff_Temps_K = load_eff_temperature_curve(eff_temp_npz)
sample_Temp_K = interp_eff_T_from_f01_mhz(data.sample.f01_mhz, eff_freqs_hz, eff_Temps_K)
fit_Temp_K = interp_eff_T_from_f01_mhz(data.fit.f01_mhz, eff_freqs_hz, eff_Temps_K)
purcell_Temp_K = None
if purcell is not None:
    purcell_Temp_K = float(
        interp_eff_T_from_freq_hz(
            np.asarray([purcell.bare_rf * 1e9], dtype=np.float64),
            eff_freqs_hz,
            eff_Temps_K,
            name="bare_rf",
        )[0]
    )

print(
    "fit-point Eff_T range =",
    f"{np.min(fit_Temp_K) * 1e3:.2f} - {np.max(fit_Temp_K) * 1e3:.2f} mK",
)
if purcell_Temp_K is not None:
    print(f"Purcell Temp = Eff_T(bare_rf) = {purcell_Temp_K * 1e3:.2f} mK")

fig, _ = plot_eff_T_assignment(
    data,
    eff_freqs_hz,
    eff_Temps_K,
    sample_Temp_K,
    fit_Temp_K,
    bare_rf_ghz=None if purcell is None else purcell.bare_rf,
)
figure_paths["effective_temperature_assignment"] = zt1.save_t1_curve_figure(
    fig, ctx, "Eff_T_assignment.png", bbox_inches="tight"
)
```

# Inline Fit Tools

```python
PARAM_BY_MECH = {
    "capacitive": "Q_cap",
    "quasiparticle": "x_qp",
    "inductive": "Q_ind",
}
CHANNEL_BY_PARAM = {
    "Q_cap": ("t1_capacitive", "Q_cap"),
    "x_qp": ("t1_quasiparticle_tunneling", "x_qp"),
    "Q_ind": ("t1_inductive", "Q_ind"),
}


def combine_T1_limits(*T1_limits):
    rates = None
    for limit in T1_limits:
        if limit is None:
            continue
        arr = np.asarray(limit, dtype=np.float64)
        if rates is None:
            rates = np.zeros_like(arr)
        rates = rates + 1.0 / arr
    if rates is None:
        raise ValueError("at least one T1 limit is required")
    return 1.0 / rates


def make_noise_channels(param_values, active_mechanisms):
    channels = []
    for mechanism in active_mechanisms:
        name = PARAM_BY_MECH[mechanism]
        value = param_values.get(name)
        if value is None:
            continue
        channel_name, value_name = CHANNEL_BY_PARAM[name]
        channels.append((channel_name, {value_name: float(value)}))
    if not channels:
        raise ValueError("at least one active mechanism is required")
    return channels


def intrinsic_T1_profile_ns(fluxs, Temp_profile_K, param_values, active_mechanisms):
    fluxs = np.asarray(fluxs, dtype=np.float64)
    Temps = np.asarray(Temp_profile_K, dtype=np.float64)
    if fluxs.shape != Temps.shape:
        raise ValueError("fluxs and Temp profile must have the same shape")
    channels = make_noise_channels(param_values, active_mechanisms)
    return np.asarray(
        [
            calculate_eff_t1_vs_flux_fast(
                ctx.params,
                np.asarray([flux], dtype=np.float64),
                channels,
                float(Temp),
            )[0]
            for flux, Temp in zip(fluxs, Temps, strict=True)
        ],
        dtype=np.float64,
    )


def mechanism_T1_profile_ns(fluxs, Temp_profile_K, mechanism, value):
    name = PARAM_BY_MECH[mechanism]
    return intrinsic_T1_profile_ns(
        fluxs,
        Temp_profile_K,
        {name: value},
        (mechanism,),
    )


def purcell_T1_profile_ns(fluxs, Temp_profile_K, purcell):
    if purcell is None:
        return None
    fluxs = np.asarray(fluxs, dtype=np.float64)
    Temps = np.asarray(Temp_profile_K, dtype=np.float64)
    if fluxs.shape != Temps.shape:
        raise ValueError("fluxs and Temp profile must have the same shape")

    out = np.empty_like(fluxs, dtype=np.float64)
    for Temp in np.unique(Temps):
        mask = Temps == Temp
        out[mask] = (
            calculate_purcell_t1_vs_flux(
                fluxs[mask],
                bare_rf=purcell.bare_rf,
                kappa=purcell.kappa_ghz,
                g=purcell.g,
                Temp=float(Temp),
                params=ctx.params,
                progress=False,
            )
        )
    return out


def fit_model_T1_ns(fluxs, Temp_profile_K, param_values, active_mechanisms, purcell_T1_ns):
    intrinsic = intrinsic_T1_profile_ns(
        fluxs,
        Temp_profile_K,
        param_values,
        active_mechanisms,
    )
    return combine_T1_limits(intrinsic, purcell_T1_ns)


def parameter_text(
    param_values,
    active_mechanisms,
    fixed_mechanisms,
    purcell_Temp_K=None,
    extra_lines=(),
):
    lines = []
    for mechanism in active_mechanisms:
        name = PARAM_BY_MECH[mechanism]
        suffix = " fixed" if mechanism in fixed_mechanisms or name in fixed_mechanisms else ""
        value = param_values[name]
        lines.append(f"{name} = {value:.3e}{suffix}")
    lines.append("Temp = Eff_T(f01)")
    if purcell_Temp_K is not None:
        lines.append(f"Purcell Temp = {purcell_Temp_K * 1e3:.2f} mK")
    lines.extend(extra_lines)
    return "\n".join(lines)
```

```python
def fit_T1_with_eff_temperature_profile(
    fit_data,
    Temp_profile_K,
    init,
    bounds,
    *,
    active_mechanisms,
    fixed_mechanisms=(),
    purcell_T1_ns=None,
    T1_error_policy=None,
    flux_weighting=None,
    residual_mode="log",
    loss="linear",
    max_nfev=1000,
):
    active_params = tuple(PARAM_BY_MECH[name] for name in active_mechanisms)
    fixed_params = {
        PARAM_BY_MECH[name] if name in PARAM_BY_MECH else name for name in fixed_mechanisms
    }
    free_params = tuple(name for name in active_params if name not in fixed_params)

    fluxs = np.asarray(fit_data.fluxs, dtype=np.float64)
    T1s = np.asarray(fit_data.T1_ns, dtype=np.float64)
    T1errs = np.asarray(fit_data.T1err_ns, dtype=np.float64)
    Temps = np.asarray(Temp_profile_K, dtype=np.float64)
    if fluxs.shape != T1s.shape or fluxs.shape != Temps.shape:
        raise ValueError("fit data and temperature profile shapes do not match")

    flux_weights = zfit.build_flux_residual_weights(fluxs, flux_weighting)
    error_resolution = zfit.resolve_measurement_errors(
        T1s,
        T1errs,
        policy=T1_error_policy,
        flux_weights=flux_weights,
        name="T1errs",
    )
    sigma = error_resolution.effective_errors

    lower = np.asarray([bounds[name][0] for name in free_params], dtype=np.float64)
    upper = np.asarray([bounds[name][1] for name in free_params], dtype=np.float64)
    x0 = np.asarray([init[name] for name in free_params], dtype=np.float64)
    if np.any(lower <= 0.0) or np.any(upper <= lower) or np.any(x0 <= 0.0):
        raise ValueError("init and bounds must be positive and ordered")

    def unpack(log_values):
        values = {name: float(init[name]) for name in active_params}
        for name, value in zip(free_params, np.exp(log_values), strict=True):
            values[name] = float(value)
        return values

    def residual(log_values):
        model = fit_model_T1_ns(
            fluxs,
            Temps,
            unpack(log_values),
            active_mechanisms,
            purcell_T1_ns,
        )
        if residual_mode == "log":
            scaled_sigma = np.maximum(sigma / T1s, 1e-12)
            raw = (np.log(model) - np.log(T1s)) / scaled_sigma
        elif residual_mode == "linear":
            raw = (model - T1s) / sigma
        else:
            raise ValueError("residual_mode must be 'log' or 'linear'")
        return raw * flux_weights.residual_weights

    if free_params:
        opt = least_squares(
            residual,
            np.log(x0),
            bounds=(np.log(lower), np.log(upper)),
            loss=loss,
            max_nfev=max_nfev,
        )
        fit_values = unpack(opt.x)
        residuals = residual(opt.x)
        success = bool(opt.success)
        message = opt.message
    else:
        fit_values = {name: float(init[name]) for name in active_params}
        residuals = residual(np.asarray([], dtype=np.float64))
        success = True
        message = "all parameters fixed"

    model_T1 = fit_model_T1_ns(
        fluxs,
        Temps,
        fit_values,
        active_mechanisms,
        purcell_T1_ns,
    )
    return {
        "params": fit_values,
        "free": free_params,
        "fixed": tuple(name for name in active_params if name in fixed_params),
        "model_T1_ns": model_T1,
        "residuals": residuals,
        "success": success,
        "message": message,
        "error_resolution": error_resolution,
        "flux_weights": flux_weights,
    }


def build_profile_T1_curves(fluxs, Temp_profile_K, fit_result, active_mechanisms, purcell_T1_ns):
    components = {
        mechanism: mechanism_T1_profile_ns(
            fluxs,
            Temp_profile_K,
            mechanism,
            fit_result["params"][PARAM_BY_MECH[mechanism]],
        )
        for mechanism in active_mechanisms
    }
    if purcell_T1_ns is not None:
        components["Purcell"] = purcell_T1_ns
    effective = fit_model_T1_ns(
        fluxs,
        Temp_profile_K,
        fit_result["params"],
        active_mechanisms,
        purcell_T1_ns,
    )
    return effective, components


def intrinsic_target_T1_after_purcell(observed_T1_ns, purcell_T1_ns):
    observed_T1_ns = np.asarray(observed_T1_ns, dtype=np.float64)
    if purcell_T1_ns is None:
        return observed_T1_ns.copy(), np.isfinite(observed_T1_ns) & (observed_T1_ns > 0.0)
    purcell_T1_ns = np.asarray(purcell_T1_ns, dtype=np.float64)
    residual_rate = 1.0 / observed_T1_ns - 1.0 / purcell_T1_ns
    valid = np.isfinite(residual_rate) & (residual_rate > 0.0)
    target = np.full_like(observed_T1_ns, np.nan, dtype=np.float64)
    target[valid] = 1.0 / residual_rate[valid]
    return target, valid


def mechanism_Q_lower_bound_fit(
    fit_data,
    Temp_profile_K,
    mechanism,
    reference_value,
    purcell_T1_ns,
    *,
    top_fraction=0.10,
):
    if not np.isfinite(top_fraction) or not (0.0 < top_fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1]")
    param_name = PARAM_BY_MECH[mechanism]
    reference_T1_ns = mechanism_T1_profile_ns(
        fit_data.fluxs,
        Temp_profile_K,
        mechanism,
        reference_value,
    )
    target_T1_ns, valid = intrinsic_target_T1_after_purcell(
        fit_data.T1_ns,
        purcell_T1_ns,
    )
    valid = valid & np.isfinite(reference_T1_ns) & (reference_T1_ns > 0.0)
    if not np.any(valid):
        raise ValueError(f"{mechanism} has no valid points for Q lower-bound fit")

    reference_Q = 1.0 / reference_value if param_name == "x_qp" else reference_value
    pointwise_Q_lower = np.full_like(fit_data.T1_ns, np.nan, dtype=np.float64)
    pointwise_Q_lower[valid] = reference_Q * target_T1_ns[valid] / reference_T1_ns[valid]
    valid_Q_lower = np.sort(pointwise_Q_lower[valid])
    top_count = max(1, int(np.ceil(valid_Q_lower.size * top_fraction)))
    top_Q_lower = valid_Q_lower[-top_count:]
    Q_lower = float(np.mean(top_Q_lower))
    param_value = 1.0 / Q_lower if param_name == "x_qp" else Q_lower

    return {
        "params": {param_name: param_value},
        "free": (),
        "fixed": (param_name,),
        "model_T1_ns": fit_model_T1_ns(
            fit_data.fluxs,
            Temp_profile_K,
            {param_name: param_value},
            (mechanism,),
            purcell_T1_ns,
        ),
        "residuals": np.asarray([], dtype=np.float64),
        "success": True,
        "message": "Q lower bound from top-fraction pointwise average",
        "Q_lower_bound": Q_lower,
        "pointwise_Q_lower": pointwise_Q_lower,
        "top_fraction": top_fraction,
        "top_count": top_count,
        "top_Q_lower": top_Q_lower,
        "valid_mask": valid,
    }
```

# Fit Settings

```python
active_mechanisms = ("capacitive", "quasiparticle", "inductive")
fixed_mechanisms = ()

fit_init = {
    "Q_cap": 1.0e6,
    "x_qp": 1.0e-8,
    "Q_ind": 1.0e8,
}
fit_bounds = {
    "Q_cap": (1.0, 1.0e10),
    "x_qp": (1.0e-12, 1.0),
    "Q_ind": (1.0, 1.0e12),
}

fit_purcell_Temp_K = (
    None if purcell_Temp_K is None else np.full_like(data.fit.fluxs, purcell_Temp_K)
)
fit_purcell_T1_ns = purcell_T1_profile_ns(
    data.fit.fluxs,
    fit_purcell_Temp_K,
    purcell,
)

t_flux_count = 250
t_fluxs = np.linspace(*analysis_flux_range, t_flux_count)
t_f01_mhz = zfit.predict_f01_mhz(ctx.params, t_fluxs)
t_Temp_K = interp_eff_T_from_f01_mhz(t_f01_mhz, eff_freqs_hz, eff_Temps_K)
grid_purcell_Temp_K = (
    None if purcell_Temp_K is None else np.full_like(t_fluxs, purcell_Temp_K)
)
grid_purcell_T1_ns = purcell_T1_profile_ns(
    t_fluxs,
    grid_purcell_Temp_K,
    purcell,
)

T1_error_policy = zfit.MeasurementErrorPolicy(
    nan_policy="bin_median",
    relative_floor=0.05,
    fallback_error=1000.0,
)
flux_weighting = zfit.FluxResidualWeighting(
    mode="equal_flux_bin",
    bin_width=0.01,
    origin=analysis_flux_range[0],
)
residual_mode = "log"
loss = "linear"
max_nfev = 1000

Q_lower_top_fraction = 0.10
```

# Mechanism Q Lower Bounds

```python
mechanism_fits = {}
mechanism_rows = []

for mechanism in active_mechanisms:
    param_name = PARAM_BY_MECH[mechanism]
    fit = mechanism_Q_lower_bound_fit(
        data.fit,
        fit_Temp_K,
        mechanism,
        fit_init[param_name],
        fit_purcell_T1_ns,
        top_fraction=Q_lower_top_fraction,
    )
    mechanism_fits[mechanism] = fit
    mechanism_rows.append(
        {
            "mechanism": mechanism,
            "parameter": param_name,
            "value": fit["params"][param_name],
            "Q lower bound": fit["Q_lower_bound"],
            "top fraction": fit["top_fraction"],
            "top count": fit["top_count"],
            "valid points": int(np.count_nonzero(fit["valid_mask"])),
            "success": fit["success"],
        }
    )

display(pd.DataFrame(mechanism_rows))
```

```python
for mechanism, fit in mechanism_fits.items():
    mechanism_effective_T1_ns, mechanism_component_T1s = build_profile_T1_curves(
        t_fluxs,
        t_Temp_K,
        fit,
        (mechanism,),
        grid_purcell_T1_ns,
    )
    fig, _ = zt1.plot_eff_t1_with_sample(
        data.sample.values,
        data.sample.T1_ns,
        data.sample.T1err_ns,
        mechanism_effective_T1_ns,
        ctx.flux_half,
        ctx.flux_period,
        t_fluxs,
        label=f"{mechanism} lower bound",
        component_t1s=mechanism_component_T1s,
        parameter_text=parameter_text(
            fit["params"],
            (mechanism,),
            (),
            purcell_Temp_K=purcell_Temp_K,
            extra_lines=(
                f"Q lower = {fit['Q_lower_bound']:.3e}",
                f"constraint = top {fit['top_fraction']:.0%} mean",
            ),
        ),
    )
    param_name = PARAM_BY_MECH[mechanism]
    figure_paths[f"T1_fit_{param_name}"] = zt1.save_t1_curve_figure(
        fig, ctx, f"T1s_fit_{param_name}_profile_Temp.png", bbox_inches="tight"
    )
```

# Combined Fit

```python
profile_fit = fit_T1_with_eff_temperature_profile(
    data.fit,
    fit_Temp_K,
    fit_init,
    fit_bounds,
    active_mechanisms=active_mechanisms,
    fixed_mechanisms=fixed_mechanisms,
    purcell_T1_ns=fit_purcell_T1_ns,
    T1_error_policy=T1_error_policy,
    flux_weighting=flux_weighting,
    residual_mode=residual_mode,
    loss=loss,
    max_nfev=max_nfev,
)

print("success =", profile_fit["success"])
print(profile_fit["message"])
print("free =", profile_fit["free"])
print("fixed =", profile_fit["fixed"])
print(
    "effective flux bins =",
    f"{profile_fit['flux_weights'].effective_observation_count:g}",
)
display(
    pd.DataFrame(
        [
            {"parameter": name, "value": value}
            for name, value in profile_fit["params"].items()
        ]
    )
)
```

# Channel Curves

```python
grid_effective_T1_ns, component_T1s = build_profile_T1_curves(
    t_fluxs,
    t_Temp_K,
    profile_fit,
    active_mechanisms,
    grid_purcell_T1_ns,
)
```

```python
fig, _ = zt1.plot_eff_t1_with_sample(
    data.sample.values,
    data.sample.T1_ns,
    data.sample.T1err_ns,
    grid_effective_T1_ns,
    ctx.flux_half,
    ctx.flux_period,
    t_fluxs,
    label="effective",
    component_t1s=component_T1s,
    parameter_text=parameter_text(
        profile_fit["params"],
        active_mechanisms,
        fixed_mechanisms,
        purcell_Temp_K=purcell_Temp_K,
    ),
)
figure_paths["channel_overlay"] = zt1.save_t1_curve_figure(
    fig, ctx, "T1s_fit_eff_profile_Temp.png", bbox_inches="tight"
)
```

# Figures

```python
figure_paths
```
