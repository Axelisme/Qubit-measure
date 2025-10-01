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
    fit_noise_and_temp,
    calculate_eff_t1_vs_flx_with,
)
from zcu_tools.simulate import flx2mA, mA2flx
from zcu_tools.simulate.fluxonium import (
    calculate_eff_t1_with,
    calculate_percell_t1_vs_flx,
)
```

```python
qub_name = "Q3_2D/Q2"
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
```

## Load Sample Points

```python
# loading points
loadpath = f"../../result/{qub_name}/samples.csv"

freqs_df = pd.read_csv(loadpath)
freqs_df = freqs_df[~np.isnan(freqs_df["T1 (us)"])]
s_mAs = freqs_df["calibrated mA"].values  # mA
s_fpts = freqs_df["Freq (MHz)"].values
s_T1s = freqs_df["T1 (us)"].values
s_T1errs = freqs_df["T1err (us)"].values

# filter out bad points
valid = s_T1errs < 0.2 * s_T1s
s_mAs = s_mAs[valid]
s_fpts = s_fpts[valid]
s_T1s = s_T1s[valid]
s_T1errs = s_T1errs[valid]

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
t_flxs = np.linspace(0.0, 0.6, 1000)
t_mAs = flx2mA(t_flxs, mA_c, period)
```

```python
fluxonium = scq.Fluxonium(*params, flux=0.5, cutoff=40, truncated_dim=6)
spectrum_data = fluxonium.get_spectrum_vs_paramvals(
    "flux", t_flxs, evals_count=20, get_eigenstates=True
)
t_fpts = (spectrum_data.energy_table[:, 1] - spectrum_data.energy_table[:, 0]) * 1e3
```

```python
s_n_elements = np.abs(
    fluxonium.get_matelements_vs_paramvals(
        "n_operator", "flux", s_flxs, evals_count=20
    ).matrixelem_table[:, 0, 1]
)
s_n_spectral = 1 / (s_T1s * s_n_elements**2)
s_n_spectral_err = s_T1errs / (s_T1s * s_n_elements) ** 2
s_phi_elements = np.abs(
    fluxonium.get_matelements_vs_paramvals(
        "phi_operator", "flux", s_flxs, evals_count=20
    ).matrixelem_table[:, 0, 1]
)
s_phi_spectral = 1 / (s_T1s * s_phi_elements**2)
s_phi_spectral_err = s_T1errs / (s_T1s * s_phi_elements) ** 2
```

```python
plt.title(r"$S(w)/\hbar^2$")
plt.errorbar(s_fpts, s_n_spectral, yerr=s_n_spectral_err, fmt=".", label="n")
plt.errorbar(s_fpts, s_phi_spectral, yerr=s_phi_spectral_err, fmt=".", label="phi")
plt.legend()
plt.grid()
plt.show()
```

# T1 curve

```python
Temp = 113e-3
# Temp = 50e-3
# Temp = fit_temp

plot_args = (s_mAs, s_T1s, s_T1errs, mA_c, period, fluxonium, spectrum_data, t_flxs)
```

## Q_cap

```python
# Q_cap = fit_noise[0][1]["Q_cap"]
Q_cap = 7.0e5

fig, _ = plot_t1_with_sample(
    *plot_args,
    name="Q_cap",
    noise_name="t1_capacitive",
    values=[Q_cap / 2, Q_cap, Q_cap * 2],
    Temp=Temp,
)

# fig.savefig(f"../../result/{qub_name}/image/T1s_fit_Qcap.png")
plt.show()
```

## Q_qp

```python
# x_qp = fit_noise[1][1]["x_qp"]
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
# Q_ind = fit_noise[2][1]["Q_ind"]
Q_ind = 7e7

fig, ax = plot_t1_with_sample(
    *plot_args,
    name="Q_ind",
    noise_name="t1_inductive",
    values=[Q_ind / 2, Q_ind, Q_ind * 2],
    Temp=Temp,
)
# ax.set_xlim(-5, -4)

fig.savefig(f"../../result/{qub_name}/image/T1s_fit_Q_ind.png")
plt.show()
```

## Fitting eff

```python
fit_noise, fit_temp = fit_noise_and_temp(
    s_flxs,
    1e3 * s_T1s,
    fluxonium,
    init_guess_noise=[
        ("t1_capacitive", dict(Q_cap=Q_cap)),
        # ("t1_quasiparticle_tunneling", dict(x_qp=x_qp)),
        ("t1_inductive", dict(Q_ind=Q_ind)),
    ],
    bounds_noise=[
        ("t1_capacitive", dict(Q_cap=(0.5 * Q_cap, 10 * Q_cap))),
        # ("t1_quasiparticle_tunneling", dict(x_qp=(0.1 * x_qp, 2 * x_qp))),
        ("t1_inductive", dict(Q_ind=(0.5 * Q_ind, 10 * Q_ind))),
    ],
    init_guess_temp=Temp,
    bounds_temp=(10e-3, 500e-3),
    asym_loss_weight=2.0,
)
print("Noise channels:")
for ch_name, noise_params in fit_noise:
    print("\t" + ch_name + ":")
    for name, value in noise_params.items():
        print(f"\t\t{name}: {value:.1e}")
print(f"Temperature: {1e3 * fit_temp:.2f} mK")


fit_t1_effs = calculate_eff_t1_vs_flx_with(
    t_flxs, fit_noise, Temp, fluxonium=fluxonium, spectrum_data=spectrum_data
)

fig, ax = plot_eff_t1_with_sample(
    s_mAs,
    s_T1s,
    s_T1errs,
    fit_t1_effs,
    mA_c,
    period,
    t_flxs,
    title=f"Temperature = {fit_temp * 1e3:.2f} mK",
)

fig.savefig(f"../../result/{qub_name}/image/T1s_fit_eff.png")
plt.show()
```

```python
Q_cap = fit_noise[0][1]["Q_cap"]
x_qp = fit_noise[1][1]["x_qp"]
Q_ind = fit_noise[2][1]["Q_ind"]
Temp = fit_temp
```

# Advance

```python
# noise_channels = fit_noise
noise_channels = [
    ("t1_capacitive", dict(Q_cap=Q_cap)),
    # ("t1_quasiparticle_tunneling", dict(x_qp=x_qp)),
    ("t1_inductive", dict(Q_ind=Q_ind)),
]

noise_label = "\n".join(
    [
        f"{name:<5} = {value:.1e}"
        for ch in noise_channels
        for name, value in ch[1].items()
    ]
)
```

```python
t1_effs = calculate_eff_t1_vs_flx_with(
    t_flxs, noise_channels, Temp, fluxonium=fluxonium, spectrum_data=spectrum_data
)
```

## Percell Effect

```python
rf_w = 2e-3  # GHz
g = results["dispersive"]["g"]
r_f = results["dispersive"]["r_f"]

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
