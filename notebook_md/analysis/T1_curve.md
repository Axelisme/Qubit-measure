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

# Import

```python
%load_ext autoreload
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray

%autoreload 2
from zcu_tools.meta_tool import (
    QubitParams,
    T1CurveFit,
    T1CurveFitParams,
    T1CurveFitUncertainty,
)
import zcu_tools.notebook.analysis.t1_curve as zt1
from zcu_tools.simulate import value2flux
import zcu_tools.simulate.fluxonium as zf
```

```python
chip_name = "Q12_2D[7]"
qub_name = "Q4"

result_dir = os.path.join("..", "..", "result", chip_name, qub_name)
t1_curve_dir = os.path.join(result_dir, "t1_curve")
image_dir = t1_curve_dir
os.makedirs(result_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
```

# Load data

## Parameters

```python
params_file = QubitParams(os.path.join(result_dir, "params.json"), readonly=True)
fit_inputs = params_file.require_dispersive_inputs(default_bare_rf=5.0)
prior_dispersive = params_file.get_dispersive_fit()

params = fit_inputs.params
EJ, EC, EL = params
flx_half = fit_inputs.flux_half
flx_int = fit_inputs.flux_int
flx_period = fit_inputs.flux_period

print("params = ", params, " GHz")
print("flx_half = ", flx_half)
print("flx_int = ", flx_int)
print("flx_period = ", flx_period)

sample_f = 9.58464

g = 0.1  # GHz
rf_w = 4.2e-3  # GHz
bare_rf = fit_inputs.bare_rf_seed
if prior_dispersive is not None:
    g = prior_dispersive.g
print(f"bare rf = {bare_rf}", "GHz")
print(f"g = {g}", "GHz")
```

## Load Sample Points

```python
# loading points
loadpath = os.path.join(result_dir, "samples.csv")

freqs_df = pd.read_csv(loadpath)
freqs_df = freqs_df[~np.isnan(freqs_df["T1 (us)"])]
s_mAs: NDArray[np.float64] = freqs_df["calibrated mA"].values  # type: ignore
s_fpts: NDArray[np.float64] = 1e-3 * freqs_df["Freq (MHz)"].values  # type: ignore
s_T1s: NDArray[np.float64] = 1e3 * freqs_df["T1 (us)"].values  # type: ignore
t1err_col = "T1err (us)"
if t1err_col in freqs_df.columns:
    s_T1errs: NDArray[np.float64] = 1e3 * freqs_df[t1err_col].values  # type: ignore
else:
    s_T1errs = np.full_like(s_T1s, np.nan, dtype=np.float64)
    print(f"No '{t1err_col}' column found; T1 fitting will be unweighted.")

# filter out bad points
finite_positive_err = np.isfinite(s_T1errs) & (s_T1errs > 0.0)
valid = np.isnan(s_T1errs) | (finite_positive_err & (s_T1errs < 0.25 * s_T1s))
s_mAs = s_mAs[valid]
s_fpts = s_fpts[valid]
s_T1s = s_T1s[valid]
s_T1errs = s_T1errs[valid]

# sort by flux
s_mAs, s_fpts, s_T1s, s_T1errs = tuple(
    np.array(a) for a in zip(*sorted(zip(s_mAs, s_fpts, s_T1s, s_T1errs)))
)
s_flxs = value2flux(s_mAs, flx_half, flx_period)
s_omegas = zt1.freq2omega(s_fpts)

freqs_df.head(10)
```

```python
fig, _ = zt1.plot_sample_t1(s_mAs, s_T1s, s_T1errs, flx_half, flx_period)
plt.show()
fig.savefig(os.path.join(image_dir, "T1s.png"))
plt.close(fig)
```

# T1 curve

```python
t_flxs = np.linspace(0.0, 1.0, 100)

s_n_spectrum_data = None
s_phi_spectrum_data = None
```

```python
Temp = 60e-3

plot_args = (
    s_mAs,
    s_T1s,
    s_T1errs,
    flx_half,
    flx_period,
    params,
    t_flxs
)
```

## Q_cap

```python
s_n_spectrum_data, s_n_elements = zf.calculate_n_oper_vs_flux(
    params, s_flxs, spectrum_data=s_n_spectrum_data
)

mask_range = (None, None)
masks = np.ones_like(s_omegas, dtype=bool)
if omega_lb := mask_range[0]:
    masks = np.logical_and(masks, s_omegas >= omega_lb)
if omega_ub := mask_range[1]:
    masks = np.logical_and(masks, s_omegas <= omega_ub)

mask_omegas = s_omegas[masks]
mask_T1s = s_T1s[masks]
mask_T1errs = s_T1errs[masks]
mask_n_elements = s_n_elements[masks]

Temp_Qcap = zt1.find_proper_Temp(
    Temp,
    lambda T: zt1.calc_Qcap_vs_omega(
        params, mask_omegas, mask_T1s, mask_n_elements, T1errs=mask_T1errs, Temp=T
    )[0],
)
print(f"Temp_Qcap = {Temp_Qcap * 1e3:.2f} mK")
```

```python
Temp_Qcap = Temp
s_Qcaps, s_Qcaps_err = zt1.calc_Qcap_vs_omega(
    params, s_omegas, s_T1s, s_n_elements, T1errs=s_T1errs, Temp=Temp_Qcap
)

fig, ax = zt1.plot_Q_vs_omega(s_omegas, s_Qcaps, s_Qcaps_err, Qname=r"$Q_{cap}$")
ax.set_title(f"Temp = {Temp_Qcap * 1e3:.2f} mK")
ax.set_ylim(1e4, 1e6)

_, fit_Qcaps = zt1.add_Q_fit(ax, s_omegas, s_Qcaps, omega_range=mask_range, fit_constant=True)


plt.show()
fig.savefig(os.path.join(image_dir, "Qcap_vs_omega.png"))
plt.close(fig)
```

```python
mask_cap_dipoles = zt1.calc_cap_dipole(params, mask_n_elements, mask_omegas, Temp_Qcap)

fig, ax = zt1.plot_t1_vs_elements(
    mask_cap_dipoles, mask_T1s, mask_T1errs, Q_name=r"$Q_{cap}$"
)
ax.set_title(f"Temp = {Temp_Qcap * 1e3:.2f} mK")
# ax.set_ylim(5e3, 5e5)

plt.show()
fig.savefig(os.path.join(image_dir, r"T1s_vs_|d01|_cap.png"))
plt.close(fig)
```

```python
Q_cap = float(np.mean(fit_Qcaps))

# Q_cap = 4e5
log_product = np.log(mask_T1s * mask_cap_dipoles)
up_Q = np.exp(np.mean(log_product) + 2.0 * np.std(log_product))
down_Q = np.exp(np.mean(log_product) - 2.0 * np.std(log_product))

fig, _ = zt1.plot_t1_with_sample(
    *plot_args,
    name="Q_cap",
    noise_name="t1_capacitive",
    noise_values=[down_Q, Q_cap, up_Q],
    Temp=Temp_Qcap,
)

plt.show()
fig.savefig(os.path.join(image_dir, "T1s_fit_Qcap.png"))
plt.close(fig)
```

## Q_qp

```python
s_sin2_elements = np.asarray([zt1.calc_qp_oper(params, flx) for flx in s_flxs])

mask_range = (6, None)
masks = np.ones_like(s_omegas, dtype=bool)
if omega_lb := mask_range[0]:
    masks = np.logical_and(masks, s_omegas >= omega_lb)
if omega_ub := mask_range[1]:
    masks = np.logical_and(masks, s_omegas <= omega_ub)

mask_omegas = s_omegas[masks]
mask_T1s = s_T1s[masks]
mask_T1errs = s_T1errs[masks]
mask_sin2_elements = s_sin2_elements[masks]

Temp_Xqp = zt1.find_proper_Temp(
    Temp,
    lambda T: zt1.calc_Qqp_vs_omega(
        params, mask_omegas, mask_T1s, mask_sin2_elements, T1errs=mask_T1errs, Temp=T
    )[0],
)
print(f"Temp_Xqp = {Temp_Xqp * 1e3:.2f} mK")
```

```python
Temp_Xqp = Temp

s_Qqps, s_Qqps_err = zt1.calc_Qqp_vs_omega(
    params, s_omegas, s_T1s, s_sin2_elements, T1errs=s_T1errs, Temp=Temp_Xqp
)

fig, ax = zt1.plot_Q_vs_omega(s_omegas, s_Qqps, s_Qqps_err, Qname=r"$Q_{qp}$")
ax.set_title(f"Temp = {Temp_Xqp * 1e3:.2f} mK")
ax.set_ylim(1e2,1e6)

_, fit_Qqps = zt1.add_Q_fit(ax, s_omegas, s_Qqps, omega_range=(6, None), fit_constant=True)

plt.show()
fig.savefig(os.path.join(image_dir, "Qqp_vs_omega.png"))
plt.close(fig)
```

```python
mask_qp_dipoles = zt1.calc_qp_dipole(params, mask_sin2_elements, mask_omegas, Temp_Xqp)

fig, ax = zt1.plot_t1_vs_elements(
    mask_qp_dipoles,
    mask_T1s,
    mask_T1errs,
    Q_name=r"$x_{qp}$",
    product2val=lambda x: 1 / x
)
ax.set_title(f"Temp = {Temp_Xqp * 1e3:.2f} mK")
# ax.set_ylim(5e3, 5e5)

plt.show()
fig.savefig(os.path.join(image_dir, "T1s_vs_|d01|_qp.png"))
plt.close(fig)
```

```python
x_qp = 1 / float(np.mean(fit_Qqps) + 20*np.std(fit_Qqps))

# x_qp = 1.5e-4
log_product = np.log(mask_T1s * mask_qp_dipoles)
up_Q = np.exp(np.mean(log_product) + 2.0 * np.std(log_product))
down_Q = np.exp(np.mean(log_product) - 2.0 * np.std(log_product))

fig, _ = zt1.plot_t1_with_sample(
    *plot_args,
    name="x_qp",
    noise_name="t1_quasiparticle_tunneling",
    noise_values=[1/down_Q, x_qp, 1/up_Q],
    Temp=Temp,
)

plt.show()
fig.savefig(os.path.join(image_dir, "T1s_fit_xqp.png"))
plt.close(fig)
```

## Q_ind

```python
s_phi_spectrum_data, s_phi_elements = zf.calculate_phi_oper_vs_flux(
    params, s_flxs, spectrum_data=s_phi_spectrum_data
)

mask_range = (None, 4)
masks = np.ones_like(s_omegas, dtype=bool)
if omega_lb := mask_range[0]:
    masks = np.logical_and(masks, s_omegas >= omega_lb)
if omega_ub := mask_range[1]:
    masks = np.logical_and(masks, s_omegas <= omega_ub)

mask_omegas = s_omegas[masks]
mask_T1s = s_T1s[masks]
mask_T1errs = s_T1errs[masks]
mask_phi_elements = s_phi_elements[masks]

Temp_Qind = zt1.find_proper_Temp(
    Temp,
    lambda T: zt1.calc_Qind_vs_omega(
        params, mask_omegas, mask_T1s, mask_phi_elements, T1errs=mask_T1errs, Temp=T
    )[0],
)
print(f"Temp_Qind = {Temp_Qind * 1e3:.2f} mK")
```

```python
Temp_Qind = Temp
Qind_array, Qind_array_err = zt1.calc_Qind_vs_omega(
    params, s_omegas, s_T1s, s_phi_elements, T1errs=s_T1errs, Temp=Temp_Qind
)

fig, ax = zt1.plot_Q_vs_omega(s_omegas, Qind_array, Qind_array_err, Qname=r"$Q_{ind}$")
ax.set_title(f"Temp = {Temp_Qind * 1e3:.2f} mK")

_, fit_Qinds = zt1.add_Q_fit(ax, s_omegas, Qind_array, omega_range=mask_range, fit_constant=True)

plt.show()
fig.savefig(os.path.join(image_dir, "Qind_vs_omega.png"))
plt.close(fig)
```

```python
mask_ind_dipoles = zt1.calc_ind_dipole(params, mask_phi_elements, mask_omegas, Temp_Qind)

fig, ax = zt1.plot_t1_vs_elements(
    mask_ind_dipoles,
    mask_T1s,
    mask_T1errs,
    Q_name=r"$Q_{ind}$"
)
ax.set_title(f"Temp = {Temp_Qind * 1e3:.2f} mK")
# ax.set_ylim(5e3, 5e5)

plt.show()
fig.savefig(os.path.join(image_dir, "T1s_vs_|d01|_ind.png"))
plt.close(fig)
```

```python
Q_ind = float(np.mean(fit_Qinds))

# Q_ind = 7e7
log_product = np.log(mask_T1s * mask_ind_dipoles)
up_Q = np.exp(np.mean(log_product) + 2.0 * np.std(log_product))
down_Q = np.exp(np.mean(log_product) - 2.0 * np.std(log_product))

fig, ax = zt1.plot_t1_with_sample(
    *plot_args,
    name="Q_ind",
    noise_name="t1_inductive",
    noise_values=[down_Q, Q_ind, up_Q],
    Temp=Temp_Qind,
)
# ax.set_xlim(-5, -4)

plt.show()
fig.savefig(os.path.join(image_dir, "T1s_fit_Q_ind.png"))
plt.close(fig)
```

## All in once

```python
fit_init = zt1.T1FitParams(
    Temp=Temp,
    Q_cap=Q_cap,
    x_qp=x_qp,
    # Q_ind=Q_ind,
)

fit_bounds = {"Temp": (10e-3, 300e-3)}
if fit_init.Q_cap is not None:
    fit_bounds["Q_cap"] = (fit_init.Q_cap / 100, fit_init.Q_cap * 100)
if fit_init.x_qp is not None:
    fit_bounds["x_qp"] = (fit_init.x_qp / 100, fit_init.x_qp * 100)
if fit_init.Q_ind is not None:
    fit_bounds["Q_ind"] = (fit_init.Q_ind / 100, fit_init.Q_ind * 100)

fit_result = zt1.fit_t1_noise_params(
    s_flxs,
    s_T1s,
    params,
    init=fit_init,
    bounds=fit_bounds,
    # fixed=("Temp",),
    T1errs=s_T1errs,
    residual_mode="log",
    loss="soft_l1",
    max_nfev=200,
    progress=True,
)

fit_Q_cap = fit_result.params.Q_cap
fit_x_qp = fit_result.params.x_qp
fit_Q_ind = fit_result.params.Q_ind
fit_Temp = fit_result.params.Temp

fit_noise = []
if fit_Q_cap is not None:
    fit_noise.append(("t1_capacitive", dict(Q_cap=fit_Q_cap)))
if fit_x_qp is not None:
    fit_noise.append(("t1_quasiparticle_tunneling", dict(x_qp=fit_x_qp)))
if fit_Q_ind is not None:
    fit_noise.append(("t1_inductive", dict(Q_ind=fit_Q_ind)))

print("fit success =", fit_result.success)
print("fit message =", fit_result.message)
if fit_Q_cap is not None:
    print(f"Q_cap = {fit_Q_cap:.3e} +/- {fit_result.stderr.Q_cap:.1e}")
if fit_x_qp is not None:
    print(f"x_qp = {fit_x_qp:.3e} +/- {fit_result.stderr.x_qp:.1e}")
if fit_Q_ind is not None:
    print(f"Q_ind = {fit_Q_ind:.3e} +/- {fit_result.stderr.Q_ind:.1e}")
print(f"Temp = {fit_Temp * 1e3:.2f} +/- {fit_result.stderr.Temp * 1e3:.2f} mK")
print("fixed =", fit_result.fixed)
print("free =", fit_result.free)
print(f"reduced chi2 = {fit_result.reduced_chi2:.3g}")
```

```python
fit_t1_effs = zf.calculate_eff_t1_vs_flux_fast(params, t_flxs, fit_noise, fit_Temp)
fit_image_path = os.path.join(image_dir, "T1s_fit_all_in_once.png")

QubitParams(os.path.join(result_dir, "params.json")).set_t1_curve_fit(
    T1CurveFit(
        params=T1CurveFitParams(
            Temp=fit_Temp,
            Q_cap=fit_Q_cap,
            x_qp=fit_x_qp,
            Q_ind=fit_Q_ind,
        ),
        stderr=T1CurveFitUncertainty(
            Q_cap=fit_result.stderr.Q_cap,
            x_qp=fit_result.stderr.x_qp,
            Q_ind=fit_result.stderr.Q_ind,
            Temp=fit_result.stderr.Temp,
        ),
        fixed=fit_result.fixed,
        free=fit_result.free,
        cost=fit_result.cost,
        reduced_chi2=fit_result.reduced_chi2,
        success=fit_result.success,
        message=fit_result.message,
        residual_mode="log",
        loss="soft_l1",
        max_nfev=200,
        init=T1CurveFitParams(
            Temp=fit_init.Temp,
            Q_cap=fit_init.Q_cap,
            x_qp=fit_init.x_qp,
            Q_ind=fit_init.Q_ind,
        ),
        bounds=fit_bounds,
    )
)

%matplotlib inline
fig, ax = zt1.plot_eff_t1_with_sample(
    s_mAs,
    s_T1s,
    s_T1errs,
    fit_t1_effs,
    flx_half,
    flx_period,
    t_flxs,
    label="all-in-one fit",
    title=f"Temperature = {fit_Temp * 1e3:.2f} mK",
)

fit_annotation = "\n".join(
    line
    for line in [
        None if fit_Q_cap is None else f"Q_cap = {fit_Q_cap:.3e}",
        None if fit_x_qp is None else f"x_qp = {fit_x_qp:.3e}",
        None if fit_Q_ind is None else f"Q_ind = {fit_Q_ind:.3e}",
        f"Temp = {fit_Temp * 1e3:.2f} mK",
        f"reduced chi2 = {fit_result.reduced_chi2:.3g}",
    ]
    if line is not None
)
ax.text(
    1.02,
    0.98,
    fit_annotation,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="0.5",
        alpha=0.85,
    ),
    clip_on=False,
)

plt.show()
fig.savefig(fit_image_path, bbox_inches="tight")
plt.close(fig)
print("t1_curve_fit written to params.json")
print(f"all-in-one fit image saved to {fit_image_path}")
```

# Advance

```python
Temp = Temp_Qcap
```

```python
# noise_channels = fit_noise
noise_channels = [
    ("t1_capacitive", dict(Q_cap=Q_cap)),
    ("t1_quasiparticle_tunneling", dict(x_qp=x_qp)),
    ("t1_inductive", dict(Q_ind=Q_ind)),
]

noise_label = "\n".join(
    [
        f"{name:<5} = " + f"{name}(w)" if callable(value) else f"{value:.1e}"
        for ch in noise_channels
        for name, value in ch[1].items()
    ]
)
```

```python
t1_effs = zf.calculate_eff_t1_vs_flux_fast(params, t_flxs, noise_channels, Temp)
```

## Percell Effect

```python
percell_t1s = zf.calculate_percell_t1_vs_flux(
    t_flxs, bare_rf=bare_rf, kappa=rf_w, g=g, Temp=Temp, params=params
)
```

```python
%matplotlib inline
fig, ax = zt1.plot_eff_t1_with_sample(
    s_mAs,
    s_T1s,
    s_T1errs,
    1 / (1 / t1_effs + 1 / percell_t1s),
    flx_half,
    flx_period,
    t_flxs,
    label=noise_label,
    title=f"Temperature = {Temp * 1e3:.2f} mK",
)

plt.show()
fig.savefig(os.path.join(image_dir, "T1s_fit_eff_with_percell.png"))
plt.close(fig)
```

## Plot eff

```python
%matplotlib inline
fig, ax = zt1.plot_eff_t1_with_sample(
    s_mAs,
    s_T1s,
    s_T1errs,
    t1_effs,
    flx_half,
    flx_period,
    t_flxs,
    label=noise_label,
    title=f"Temperature = {Temp * 1e3:.2f} mK",
)

plt.show()
fig.savefig(os.path.join(image_dir, "T1s_fit_eff.png"))
plt.close(fig)
```

```python
t1 = zf.calculate_eff_t1_fast(
    flux=0.5,
    params=params,
    noise_channels=noise_channels,
    Temp=Temp,
    # Temp=20e-3,
)

print(f"T1 = {1e-3 * t1:.2f} us")
```

```python

```
