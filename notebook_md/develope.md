---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: zcu-tools
    language: python
    name: python3
---

```python
%load_ext autoreload
import os

import numpy as np
import matplotlib.pyplot as plt

%autoreload 2
from zcu_tools.table import MetaDict
from zcu_tools.simulate import mA2flx
from zcu_tools.notebook.persistance import load_result
from zcu_tools.simulate.fluxonium import (
    calculate_n_oper_vs_flx,
    calculate_energy_vs_flx,
    calculate_chi_vs_flx,
)
```

```python
qub_name = "Q12_2D[5]/Q1"

result_dir = f"../result/{qub_name}"

os.makedirs(f"{result_dir}/image/branch_floquet", exist_ok=True)
os.makedirs(f"{result_dir}/web/branch_floquet", exist_ok=True)
os.makedirs(f"{result_dir}/data/branch_floquet", exist_ok=True)
```

```python
loadpath = f"{result_dir}/params.json"
_, params, mA_c, period, allows, data_dict = load_result(loadpath)

print(f"EJ: {params[0]:.3f} GHz, EC: {params[1]:.3f} GHz, EL: {params[2]:.3f} GHz")

if dispersive_cfg := data_dict.get("dispersive"):
    g = dispersive_cfg["g"]
    r_f = dispersive_cfg["r_f"]
    print(f"g: {g} GHz, r_f: {r_f} GHz")
elif "r_f" in allows:
    r_f = allows["r_f"]
    print(f"r_f: {r_f} GHz")

# g = 100e-3  # GHz
# rf_w = 6.1e-3  # GHz
```

```python
md = MetaDict(json_path=f"{result_dir}/meta_info.json", read_only=True)
```

# Flux dependence

```python
flxs = np.linspace(-0.05, 0.55, 100)
max_level = 20

spectrum, elements = calculate_n_oper_vs_flx(params, flxs, return_dim=max_level + 1)
_, energies = calculate_energy_vs_flx(
    params, flxs, evals_count=max_level + 1, spectrum_data=spectrum
)
```

```python
qubit_levels = [0, 1]

l_critical = np.zeros((len(flxs), len(qubit_levels)), dtype=int)
n_criticals = np.zeros((len(flxs), len(qubit_levels)))

for i, flx in enumerate(flxs):
    for k, ql in enumerate(qubit_levels):
        n_candidates = []
        for l in range(max_level + 1):
            w_kl = energies[i, k] - energies[i, l]
            n_kl = elements[i, k, l]
            if k == l or n_kl == 0:
                continue
            n_candidate = np.abs((np.abs(w_kl) - r_f) / (2 * g * n_kl)) ** 2
            n_candidates.append(n_candidate)
        l_critical[i, k] = np.argmin(n_candidates)
        n_criticals[i, k] = n_candidates[l_critical[i, k]]
```

```python
flx = mA2flx(md.cur_A, md.mA_c, 2 * abs(md.mA_e - md.mA_c), fold=True)
flx
```

```python
flx_idx = np.argmin(np.abs(flxs - flx))
n_criticals[flx_idx]
```

```python
readout_gain = 0.0693
interested_gains = [0.025, 0.112, 0.17]
interested_ns = md.ac_stark_coeff * np.array(interested_gains) ** 2

n_readout = md.ac_stark_coeff * readout_gain**2
n_max = md.ac_stark_coeff * 0.175**2


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

colors = ["blue", "red", "green"]

for k in range(len(qubit_levels)):
    color = colors[k] if k < len(colors) else "black"
    (line,) = ax1.plot(
        flxs, n_criticals[:, k], label=f"qubit level {qubit_levels[k]}", color=color
    )
    ax1.fill_between(flxs, n_criticals[:, k], n_max, color=line.get_color(), alpha=0.1)
ax1.set_ylim(0.0, n_max)
ax1.set_ylabel("n_critical")
ax1.axvline(flx, color="black", linestyle="--", label="current flux")

for n in interested_ns:
    ax1.axhline(n, color="black", linestyle="-", label=f"n={n:.0f}")
ax1.scatter(
    [flx],
    [n_readout],
    color="red",
    marker="*",
    s=100,
    label=f"n={n_readout:.0f}",
    zorder=10,
)

ax1.grid(True)
ax1.legend()

for k in range(len(qubit_levels)):
    color = colors[k] if k < len(colors) else "black"
    (line,) = ax2.plot(
        flxs, l_critical[:, k], label=f"qubit level {qubit_levels[k]}", color=color
    )
ax2.axvline(flx, color="black", linestyle="--", label="current flux")
ax2.set_ylabel("l_critical")
ax2.grid(True)
ax2.legend()

ax2.set_xlabel("flux [$\Phi_0$]")  # type: ignore
ax2.set_xlim(flxs[0], flxs[-1])

plt.show()
```

```python
chis = calculate_chi_vs_flx(params, flxs, r_f, g, res_dim=20)
```

```python
fig, ax1 = plt.subplots(figsize=(12, 4))

ax1.plot(flxs, chis[:, -5:-1])
ax1.axhline(0, color="black", linestyle="-", label="ground state")
ax1.axvline(flx, color="black", linestyle="--", label="current flux")
ax1.set_xlabel("flux [$\Phi_0$]")
ax1.set_ylabel("chi")
ax1.grid(True)
ax1.set_ylim(-0.05, 0.05)
plt.show()

```

# Mist Simulation

```python
flx = mA2flx(md.cur_A, md.mA_c, 2 * abs(md.mA_e - md.mA_c), fold=True)
flx
```

```python
import qutip as qt
from scqubits.core.fluxonium import Fluxonium

qub_dim = 20

fluxonium = Fluxonium(*params, flux=flx, cutoff=qub_dim + 10, truncated_dim=qub_dim)
evals = fluxonium.eigenvals()


# r_f = 5.3  # GHz
# g = 0.06
avg_photon = 100


T = 2 * np.pi / r_f

psi0 = qt.basis(qub_dim, 1)
times = np.arange(0, 0.01e3, step=0.01 * T)  # us


gamma1 = 0.5


def noise_spectrum(omega):
    return (omega > 0) * 0.5 * gamma1 * (omega / (2 * np.pi))


esys = fluxonium.eigensys(evals_count=qub_dim)
H = qt.Qobj(fluxonium.hamiltonian(energy_esys=esys))
n_op = qt.Qobj(fluxonium.n_operator(energy_esys=esys))
phi_op = qt.Qobj(fluxonium.phi_operator(energy_esys=esys))
H_with_drive = [H, [n_op, lambda t, amp: amp * np.cos(r_f * t)]]


args = dict(amp=2 * g * np.sqrt(avg_photon))
```

```python
output = qt.fmmesolve(
    H_with_drive,
    psi0,
    times,
    c_ops=[n_op, phi_op],
    spectra_cb=[noise_spectrum, noise_spectrum],
    T=T,  # type: ignore
    args=args,
    options=dict(progress_bar="tqdm", store_final_state=True, store_states=True),
)
p_g = np.abs(
    [
        qt.expect(qt.projection(qub_dim, 0, 0), output.states[i].unit())
        for i in range(times.shape[0])
    ]
)
p_e = np.abs(
    [
        qt.expect(qt.projection(qub_dim, 1, 1), output.states[i].unit())
        for i in range(times.shape[0])
    ]
)
```

```python
plt.plot(times, p_g, label="ground state")
plt.plot(times, p_e, label="excited state")
plt.plot(times, 1 - p_g - p_e, label="Other state")
plt.grid(True)
plt.ylim(0, 1.1 * max(1, np.max(p_g), np.max(p_e)))
plt.legend()
plt.show()
```

```python
photons = np.linspace(0, 400, 1000)

p_gs = np.zeros_like(photons, dtype=np.float64)
p_es = np.zeros_like(photons, dtype=np.float64)

from tqdm.auto import tqdm

for i, photon in enumerate(tqdm(photons)):
    args["amp"] = 2 * g * np.sqrt(photon)

    output = qt.fmmesolve(
        H_with_drive,
        psi0,
        times,
        c_ops=[n_op, phi_op],
        spectra_cb=[noise_spectrum, noise_spectrum],
        T=T,  # type: ignore
        args=args,
        options=dict(progress_bar="", store_states=True),
    )
    p_gs[i] = np.abs(qt.expect(qt.projection(qub_dim, 0, 0), output.final_state.unit()))
    p_es[i] = np.abs(qt.expect(qt.projection(qub_dim, 1, 1), output.final_state.unit()))
```

```python
plt.plot(photons, p_gs, label="ground state")
plt.plot(photons, p_es, label="excited state")
plt.plot(photons, 1 - p_gs - p_es, label="Other state")
plt.grid(True)
plt.legend()
plt.show()
```

```python

```
