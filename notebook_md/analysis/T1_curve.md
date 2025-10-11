---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
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
    version: 3.13.2
---

```python
%load_ext autoreload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scqubits as scq

%autoreload 2
from zcu_tools.notebook.persistance import load_result
from zcu_tools.notebook.analysis.t1_curve import (
    plot_t1_with_sample,
    plot_sample_t1,
    plot_eff_t1_with_sample,
    calculate_eff_t1_vs_flx_with,
    plot_Q_vs_omega,
    calc_Qcap_vs_omega,
    add_Q_fit,
    calc_Qind_vs_omega,
    freq2omega,
    find_proper_Temp,
)
from zcu_tools.simulate import flx2mA, mA2flx
from zcu_tools.simulate.fluxonium import (
    calculate_eff_t1_with,
    calculate_percell_t1_vs_flx,
)
```

```python
qub_name = "2DQ9/Q5"
```

# Load data

## Parameters

```python
loadpath = f"../../result/{qub_name}/params.json"
_, params, mA_c, period, allows, results = load_result(loadpath)
EJ, EC, EL = params

# mA_c = 3.81

print(allows)

if "r_f" in allows:
    r_f = allows["r_f"]

if "sample_f" in allows:
    sample_f = allows["sample_f"]

if "dispersive" in allows:
    g = allows["dispersive"]["g"]
    r_f = allows["dispersive"]["r_f"]
```

## Load Sample Points

```python
# loading points
loadpath = f"../../result/{qub_name}/samples.csv"

freqs_df = pd.read_csv(loadpath)
freqs_df = freqs_df[~np.isnan(freqs_df["T1 (us)"])]
s_mAs = freqs_df["calibrated mA"].values  # mA
s_fpts = 1e-3 * freqs_df["Freq (MHz)"].values  # GHz
s_T1s = 1e3 * freqs_df["T1 (us)"].values  # ns
s_T1errs = 1e3 * freqs_df["T1err (us)"].values  # ns

# filter out bad points
# valid = np.logical_or(s_T1errs < 0.25 * s_T1s, np.isnan(s_T1errs))
# s_mAs = s_mAs[valid]
# s_fpts = s_fpts[valid]
# s_T1s = s_T1s[valid]
# s_T1errs = s_T1errs[valid]

# sort by flux
s_mAs, s_fpts, s_T1s, s_T1errs = tuple(
    np.array(a) for a in zip(*sorted(zip(s_mAs, s_fpts, s_T1s, s_T1errs)))
)
s_flxs = mA2flx(s_mAs, mA_c, period)

freqs_df.head(10)
```

```python
fig, _ = plot_sample_t1(s_mAs, s_T1s, s_T1errs, mA_c, period)
fig.savefig(f"../../result/{qub_name}/image/T1s.png")
plt.show()
```

# Simulation

```python
t_flxs = np.linspace(0.0, 1.0, 1000)
t_mAs = flx2mA(t_flxs, mA_c, period)
```

```python
fluxonium = scq.Fluxonium(*params, flux=0.5, cutoff=40, truncated_dim=6)

t_spectrum_data = fluxonium.get_spectrum_vs_paramvals(
    "flux", t_flxs, evals_count=20, get_eigenstates=True
)
s_spectrum_data = fluxonium.get_spectrum_vs_paramvals(
    "flux", s_flxs, evals_count=20, get_eigenstates=True
)
```

# T1 curve

```python
Temp = 20e-3

plot_args = (s_mAs, s_T1s, s_T1errs, mA_c, period, fluxonium, t_spectrum_data, t_flxs)
```

## Q_cap

```python
s_n_elements = fluxonium.get_matelements_vs_paramvals(
    "n_operator", "flux", s_flxs, evals_count=20
).matrixelem_table[:, 0, 1]


Temp_Qcap = find_proper_Temp(
    Temp,
    lambda T: calc_Qcap_vs_omega(
        params, s_fpts, s_T1s, s_n_elements, T1errs=s_T1errs, guess_Temp=T
    )[0],
)
print(f"Temp_Qcap = {Temp_Qcap * 1e3:.2f} mK")
```

```python
# Temp_Qcap = Temp
Qcaps, Qcaps_err = calc_Qcap_vs_omega(
    params, s_fpts, s_T1s, s_n_elements, T1errs=s_T1errs, guess_Temp=Temp_Qcap
)

fig, ax = plot_Q_vs_omega(s_fpts, Qcaps, Qcaps_err, Qname=r"$Q_{cap}$")
ax.set_title(f"Temp = {Temp_Qcap * 1e3:.2f} mK")
ax.set_ylim(1e3, 1e7)

fit_Qcaps = []
# fit_Qcaps.append(add_Q_fit(ax, s_fpts, Qcaps, fit_constant=True))
fit_Qcaps.append(add_Q_fit(ax, s_fpts, Qcaps, w_range=(None, 12)))
fit_Qcaps.append(add_Q_fit(ax, s_fpts, Qcaps, w_range=(15, None)))


fit_Qcaps = list(map(np.array, fit_Qcaps))
fit_Qcaps = np.concatenate(fit_Qcaps, axis=1)
fit_Qcaps = fit_Qcaps[:, np.argsort(fit_Qcaps[0])]


def fitted_Qcap(w: np.ndarray, T: float) -> np.ndarray:
    return np.interp(w, fit_Qcaps[0], fit_Qcaps[1])


fig.savefig(f"../../result/{qub_name}/image/Qcap_vs_omega.png")
plt.show()
```

```python
# Q_cap = 4e5

Q_cap_array = fitted_Qcap(freq2omega(s_fpts), Temp_Qcap)
Q_cap_max, Q_cap_min = np.max(Q_cap_array), np.min(Q_cap_array)

fig, _ = plot_t1_with_sample(
    *plot_args,
    name="Q_cap",
    noise_name="t1_capacitive",
    # values=[Q_cap / 2, Q_cap, Q_cap * 2],
    values=[Q_cap_min, fitted_Qcap, Q_cap_max],
    Temp=Temp_Qcap,
)

fig.savefig(f"../../result/{qub_name}/image/T1s_fit_Qcap.png")
plt.show()
```

## Q_qp

```python
x_qp = 1.5e-6

fig, _ = plot_t1_with_sample(
    *plot_args,
    name="x_qp",
    noise_name="t1_quasiparticle_tunneling",
    values=[x_qp / 2, x_qp, x_qp * 2],
    Temp=Temp,
)


fig.savefig(f"../../result/{qub_name}/image/T1s_fit_xqp.png")
plt.show()
```

## Q_ind

```python
s_phi_elements = fluxonium.get_matelements_vs_paramvals(
    "phi_operator", "flux", s_flxs, evals_count=20
).matrixelem_table[:, 0, 1]

Temp_Qind = find_proper_Temp(
    Temp,
    lambda T: calc_Qind_vs_omega(
        params, s_fpts, s_T1s, s_phi_elements, T1errs=s_T1errs, guess_Temp=T
    )[0],
)
print(f"Temp_Qind = {Temp_Qind * 1e3:.2f} mK")
```

```python
Temp_Qind = Temp
Qind_array, Qind_array_err = calc_Qind_vs_omega(
    params, s_fpts, s_T1s, s_phi_elements, T1errs=s_T1errs, guess_Temp=Temp_Qind
)

fig, ax = plot_Q_vs_omega(s_fpts, Qind_array, Qind_array_err, Qname=r"$Q_{ind}$")
ax.set_title(f"Temp = {Temp_Qind * 1e3:.2f} mK")

Qind_params = []
Qind_params.append(add_Q_fit(ax, s_fpts, Qind_array, fit_constant=True))
# Qind_params.append(add_Q_fit(ax, s_fpts, Qind_array, w_range=(None, 12)))
# Qind_params.append(add_Q_fit(ax, s_fpts, Qind_array, w_range=(15, None)))


Qind_params = list(map(np.array, Qind_params))
Qind_params = np.concatenate(Qind_params, axis=1)
Qind_params = Qind_params[:, np.argsort(Qind_params[0])]


def fitted_Qind(w: np.ndarray, T: float) -> np.ndarray:
    return np.interp(w, Qind_params[0], Qind_params[1])


fig.savefig(f"../../result/{qub_name}/image/Qind_vs_omega.png")
plt.show()
```

```python
# Q_ind = 7e7

Q_ind_array = fitted_Qind(freq2omega(s_fpts), Temp_Qind)
Q_ind_max, Q_ind_min = np.max(Q_ind_array), np.min(Q_ind_array)

fig, ax = plot_t1_with_sample(
    *plot_args,
    name="Q_ind",
    noise_name="t1_inductive",
    # values=[Q_ind / 2, Q_ind, Q_ind * 2],
    values=[Q_ind_min, fitted_Qind, Q_ind_max],
    Temp=Temp_Qind,
)
# ax.set_xlim(-5, -4)

fig.savefig(f"../../result/{qub_name}/image/T1s_fit_Q_ind.png")
plt.show()
```

# Advance

```python
Temp = Temp_Qcap
```

```python
# noise_channels = fit_noise
noise_channels = [
    # ("t1_capacitive", dict(Q_cap=Q_cap)),
    # ("t1_capacitive", dict(Q_cap=fitted_Qcap)),
    ("t1_capacitive", dict(Q_cap=Q_cap_max)),
    # ("t1_quasiparticle_tunneling", dict(x_qp=x_qp)),
    # ("t1_inductive", dict(Q_ind=Q_ind)),
    # ("t1_inductive", dict(Q_ind=fitted_Qind)),
    # ("t1_inductive", dict(Q_ind=Q_ind_max)),
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
t1_effs = calculate_eff_t1_vs_flx_with(
    t_flxs, noise_channels, Temp, fluxonium=fluxonium, spectrum_data=t_spectrum_data
)
```

## Percell Effect

```python
rf_w = 5e-3  # GHz
g = 0.1

percell_t1s = calculate_percell_t1_vs_flx(
    t_flxs, r_f=r_f, kappa=rf_w, g=g, Temp=Temp, params=params
)
```

```python
%matplotlib inline
fig, ax = plot_eff_t1_with_sample(
    s_mAs,
    s_T1s,
    s_T1errs,
    1 / (1 / t1_effs + 1 / percell_t1s),
    mA_c,
    period,
    t_flxs,
    label=noise_label,
    title=f"Temperature = {Temp * 1e3:.2f} mK",
)

fig.savefig(f"../../result/{qub_name}/image/T1s_fit_eff_with_percell.png")
plt.show()
```

## Plot eff

```python
%matplotlib inline
fig, ax = plot_eff_t1_with_sample(
    s_mAs,
    s_T1s,
    s_T1errs,
    t1_effs,
    mA_c,
    period,
    t_flxs,
    label=noise_label,
    title=f"Temperature = {Temp * 1e3:.2f} mK",
)

fig.savefig(f"../../result/{qub_name}/image/T1s_fit_eff.png")
plt.show()
```

```python
t1 = calculate_eff_t1_with(
    flx=0.5,
    noise_channels=noise_channels,
    Temp=Temp,
    # Temp=20e-3,
    fluxonium=fluxonium,
)

print(f"T1 = {1e-3 * t1:.2f} us")
```

```python

```
