---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: .venv
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
    version: 3.9.23
---

```python
%load_ext autoreload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import os

%autoreload 2
from zcu_tools.notebook.persistance import load_result
import zcu_tools.notebook.analysis.t1_curve as zt1
from zcu_tools.simulate import value2flx
import zcu_tools.simulate.fluxonium as zf
```

```python
chip_name = "Si001"
qub_name = ""

result_dir = os.path.join("..", "..", "result", chip_name, qub_name)
image_dir = os.path.join(result_dir, "image", "t1_curve")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
```

# Load data

## Parameters

```python
result_dict = load_result(os.path.join(result_dir, "params.json"))
fluxdepfit_dict = result_dict.get("fluxdep_fit")
assert fluxdepfit_dict is not None, "fluxdep_fit not found in result_dict"

EJ = fluxdepfit_dict["params"]["EJ"]
EC = fluxdepfit_dict["params"]["EC"]
EL = fluxdepfit_dict["params"]["EL"]

params = (EJ, EC, EL)
flx_half = fluxdepfit_dict["flx_half"]
flx_int = fluxdepfit_dict["flx_int"]
flx_period = fluxdepfit_dict["flx_period"]

print("params = ", params, " GHz")
print("flx_half = ", flx_half)
print("flx_int = ", flx_int)
print("flx_period = ", flx_period)

sample_f = 9.58464

g = 0.1  # GHz
rf_w = 4.2e-3  # GHz
if dispersive_dict := result_dict.get("dispersive"):
    bare_rf = dispersive_dict["bare_rf"]
    g = dispersive_dict["g"]
elif "r_f" in fluxdepfit_dict["plot_transitions"]:
    bare_rf = fluxdepfit_dict["plot_transitions"]["r_f"]
else:
    bare_rf = 5.0  # GHz
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
s_T1errs: NDArray[np.float64] = 1e3 * freqs_df["T1err (us)"].values  # type: ignore

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
s_flxs = value2flx(s_mAs, flx_half, flx_period)
s_omegas = zt1.freq2omega(s_fpts)

freqs_df.head(10)
```

```python
fig, _ = zt1.plot_sample_t1(s_mAs, s_T1s, s_T1errs, flx_half, flx_period)
plt.show()
fig.savefig(os.path.join(image_dir, "T1s.png"))
plt.close(fig)
```

# Simulation

```python
from scqubits.core.fluxonium import Fluxonium

t_flxs = np.linspace(0.0, 1.0, 100)

fluxonium = Fluxonium(*params, flux=0.5, cutoff=40, truncated_dim=6)

t_spectrum_data = fluxonium.get_spectrum_vs_paramvals(
    "flux", t_flxs, evals_count=20, get_eigenstates=True
)
s_n_spectrum_data = None
s_phi_spectrum_data = None
```

# T1 curve

```python
Temp = 60e-3

plot_args = (
    s_mAs,
    s_T1s,
    s_T1errs,
    flx_half,
    flx_period,
    fluxonium,
    t_spectrum_data,
    t_flxs,
)
```

## Q_cap

```python
s_n_spectrum_data, s_n_elements = zf.calculate_n_oper_vs_flx(
    params, s_flxs, spectrum_data=s_n_spectrum_data
)

Temp_Qcap = zt1.find_proper_Temp(
    Temp,
    lambda T: zt1.calc_Qcap_vs_omega(
        params, s_omegas, s_T1s, s_n_elements, T1errs=s_T1errs, Temp=T
    )[0],
)
print(f"Temp_Qcap = {Temp_Qcap * 1e3:.2f} mK")
```

```python
# Temp_Qcap = Temp
s_Qcaps, s_Qcaps_err = zt1.calc_Qcap_vs_omega(
    params, s_omegas, s_T1s, s_n_elements, T1errs=s_T1errs, Temp=Temp_Qcap
)

fig, ax = zt1.plot_Q_vs_omega(s_omegas, s_Qcaps, s_Qcaps_err, Qname=r"$Q_{cap}$")
ax.set_title(f"Temp = {Temp_Qcap * 1e3:.2f} mK")
# ax.set_ylim(1e3, 1e7)

fit_Qcaps = []
fit_Qcaps.append(zt1.add_Q_fit(ax, s_omegas, s_Qcaps, fit_constant=True))


fit_Qcaps = list(map(np.array, fit_Qcaps))
fit_Qcaps = np.concatenate(fit_Qcaps, axis=1)
fit_Qcaps = fit_Qcaps[:, np.argsort(fit_Qcaps[0])]


def fitted_Qcap(w: np.ndarray, T: float) -> np.ndarray:
    return np.interp(w, fit_Qcaps[0], fit_Qcaps[1])


plt.show()
fig.savefig(os.path.join(image_dir, "Qcap_vs_omega.png"))
plt.close(fig)
```

```python
# Temp_Qcap = Temp
cap_mask = np.logical_and(
    s_omegas > np.min(fit_Qcaps[0]), s_omegas < np.max(fit_Qcaps[0])
)

s_cap_dipoles = zt1.calc_cap_dipole(params, s_n_elements, s_omegas, Temp_Qcap)

fig, ax = zt1.plot_t1_vs_elements(
    s_cap_dipoles[cap_mask], s_T1s[cap_mask], s_T1errs[cap_mask], Q_name=r"$Q_{cap}$"
)
ax.set_title(f"Temp = {Temp_Qcap * 1e3:.2f} mK")
# ax.set_ylim(5e3, 5e5)

plt.show()
fig.savefig(os.path.join(image_dir, r"T1s_vs_|d01|_cap.png"))
plt.close(fig)
```

```python
Q_cap_array = fitted_Qcap(s_omegas, Temp_Qcap)
Q_cap = float(np.mean(Q_cap_array) + 2 * np.std(Q_cap_array))

# Q_cap = 4e5

fig, _ = zt1.plot_t1_with_sample(
    *plot_args,
    name="Q_cap",
    noise_name="t1_capacitive",
    noise_values=[Q_cap / 2, Q_cap, Q_cap * 2],
    # values=[Q_cap_min, fitted_Qcap, Q_cap_max],
    Temp=Temp_Qcap,
)

plt.show()
fig.savefig(os.path.join(image_dir, "T1s_fit_Qcap.png"))
plt.close(fig)

```

## Q_qp

```python
s_sin2_elements = np.asarray([zt1.calc_qp_oper(params, flx) for flx in s_flxs])

Temp_Xqp = zt1.find_proper_Temp(
    Temp,
    lambda T: zt1.calc_Qqp_vs_omega(
        params, s_omegas, s_T1s, s_sin2_elements, T1errs=s_T1errs, Temp=T
    )[0],
)
print(f"Temp_Xqp = {Temp_Xqp * 1e3:.2f} mK")
```

```python
# Temp_Xqp = Temp

s_Qqps, s_Qqps_err = zt1.calc_Qqp_vs_omega(
    params, s_omegas, s_T1s, s_sin2_elements, T1errs=s_T1errs, Temp=Temp_Xqp
)

fig, ax = zt1.plot_Q_vs_omega(s_omegas, s_Qqps, s_Qqps_err, Qname=r"$Q_{qp}$")
ax.set_title(f"Temp = {Temp_Xqp * 1e3:.2f} mK")
# ax.set_ylim(1e3, 1e7)

fit_Qqps = []
fit_Qqps.append(zt1.add_Q_fit(ax, s_omegas, s_Qqps, fit_constant=True))


fit_Qqps = list(map(np.array, fit_Qqps))
fit_Qqps = np.concatenate(fit_Qqps, axis=1)
fit_Qqps = fit_Qqps[:, np.argsort(fit_Qqps[0])]


def fitted_Qqp(w: np.ndarray, T: float) -> np.ndarray:
    return np.interp(w, fit_Qqps[0], fit_Qqps[1])


plt.show()
fig.savefig(os.path.join(image_dir, "Qqp_vs_omega.png"))
plt.close(fig)
```

```python
qp_mask = np.logical_and(s_omegas > np.min(fit_Qqps[0]), s_omegas < np.max(fit_Qqps[0]))

s_qp_dipoles = zt1.calc_qp_dipole(params, s_sin2_elements, s_omegas, Temp_Xqp)

fig, ax = zt1.plot_t1_vs_elements(
    s_qp_dipoles[qp_mask],
    s_T1s[qp_mask],
    s_T1errs[qp_mask],
    Q_name=r"$x_{qp}$",
    product2val=lambda x: 1 / x,
)
ax.set_title(f"Temp = {Temp_Xqp * 1e3:.2f} mK")
# ax.set_ylim(5e3, 5e5)

plt.show()
fig.savefig(os.path.join(image_dir, "T1s_vs_|d01|_qp.png"))
plt.close(fig)
```

```python
Qqp_array = fitted_Qqp(s_omegas, Temp_Xqp)
x_qp = 1 / float(np.mean(Qqp_array) + 2 * np.std(Qqp_array))

# x_qp = 1.5e-4

fig, _ = zt1.plot_t1_with_sample(
    *plot_args,
    name="x_qp",
    noise_name="t1_quasiparticle_tunneling",
    noise_values=[x_qp / 2, x_qp, x_qp * 2],
    Temp=Temp,
)

plt.show()
fig.savefig(os.path.join(image_dir, "T1s_fit_xqp.png"))
plt.close(fig)
```

## Q_ind

```python
s_phi_spectrum_data, s_phi_elements = zf.calculate_phi_oper_vs_flx(
    params, s_flxs, spectrum_data=s_phi_spectrum_data
)

Temp_Qind = zt1.find_proper_Temp(
    Temp,
    lambda T: zt1.calc_Qind_vs_omega(
        params, s_omegas, s_T1s, s_phi_elements, T1errs=s_T1errs, Temp=T
    )[0],
)
print(f"Temp_Qind = {Temp_Qind * 1e3:.2f} mK")
```

```python
# Temp_Qind = Temp
Qind_array, Qind_array_err = zt1.calc_Qind_vs_omega(
    params, s_omegas, s_T1s, s_phi_elements, T1errs=s_T1errs, Temp=Temp_Qind
)

fig, ax = zt1.plot_Q_vs_omega(s_omegas, Qind_array, Qind_array_err, Qname=r"$Q_{ind}$")
ax.set_title(f"Temp = {Temp_Qind * 1e3:.2f} mK")

fit_Qinds = []
fit_Qinds.append(zt1.add_Q_fit(ax, s_omegas, Qind_array, fit_constant=True))

fit_Qinds = list(map(np.array, fit_Qinds))
fit_Qinds = np.concatenate(fit_Qinds, axis=1)
fit_Qinds = fit_Qinds[:, np.argsort(fit_Qinds[0])]


def fitted_Qind(w: np.ndarray, T: float) -> np.ndarray:
    return np.interp(w, fit_Qinds[0], fit_Qinds[1])


plt.show()
fig.savefig(os.path.join(image_dir, "Qind_vs_omega.png"))
plt.close(fig)
```

```python
ind_mask = np.logical_and(
    s_omegas > np.min(fit_Qinds[0]), s_omegas < np.max(fit_Qinds[0])
)

s_ind_dipoles = zt1.calc_qp_dipole(params, s_phi_elements, s_omegas, Temp_Qind)

fig, ax = zt1.plot_t1_vs_elements(
    s_ind_dipoles[ind_mask],
    s_T1s[ind_mask],
    s_T1errs[ind_mask],
    Q_name=r"$Q_{ind}$",
    product2val=lambda x: 1 / x,
)
ax.set_title(f"Temp = {Temp_Qind * 1e3:.2f} mK")
# ax.set_ylim(5e3, 5e5)

plt.show()
fig.savefig(os.path.join(image_dir, "T1s_vs_|d01|_ind.png"))
plt.close(fig)
```

```python
Q_ind_array = fitted_Qind(s_omegas, Temp_Qind)
Q_ind = float(np.mean(Q_ind_array) + 2 * np.std(Q_ind_array))

# Q_ind = 7e7

fig, ax = zt1.plot_t1_with_sample(
    *plot_args,
    name="Q_ind",
    noise_name="t1_inductive",
    noise_values=[Q_ind / 2, Q_ind, Q_ind * 2],
    Temp=Temp_Qind,
)
# ax.set_xlim(-5, -4)

plt.show()
fig.savefig(os.path.join(image_dir, "T1s_fit_Q_ind.png"))
plt.close(fig)
```

# Advance

```python
Temp = Temp_Qcap
```

```python
# noise_channels = fit_noise
noise_channels = [
    ("t1_capacitive", dict(Q_cap=Q_cap)),
    # ("t1_quasiparticle_tunneling", dict(x_qp=x_qp)),
    # ("t1_inductive", dict(Q_ind=Q_ind)),
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
t1_effs = zt1.calculate_eff_t1_vs_flx_with(
    t_flxs, noise_channels, Temp, fluxonium=fluxonium, spectrum_data=t_spectrum_data
)
```

## Percell Effect

```python
percell_t1s = zf.calculate_percell_t1_vs_flx(
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
t1 = zf.calculate_eff_t1_with(
    flux=0.5,
    noise_channels=noise_channels,
    Temp=Temp,
    # Temp=20e-3,
    fluxonium=fluxonium,
)

print(f"T1 = {1e-3 * t1:.2f} us")
```

```python

```
