---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: .venv
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
md = MetaDict(f"{result_dir}/meta_info.json", read_only=True)
```

# Mist Simulation

```python
flx = mA2flx(md.cur_A, md.mA_c, 2 * abs(md.mA_e - md.mA_c), fold=True).item()
flx
```

## Floquet

```python
import qutip as qt
from scqubits.core.fluxonium import Fluxonium

qub_dim = 20

fluxonium = Fluxonium(*params, flux=flx, cutoff=qub_dim + 10, truncated_dim=qub_dim)
evals = fluxonium.eigenvals()


# r_f = 5.3  # GHz
# g = 0.06
avg_photon = 20


T = 1 / r_f  # ns

psi0 = qt.basis(qub_dim, 1)
times = np.arange(0, 0.1e3, step=0.1 * T)  # ns


gamma1 = 1 / 3.3e2


def noise_spectrum(omega):
    return (omega > 0) * 0.5 * gamma1 * (omega / (2 * np.pi))


esys = fluxonium.eigensys(evals_count=qub_dim)
H = qt.Qobj(fluxonium.hamiltonian(energy_esys=esys))
n_op = qt.Qobj(fluxonium.n_operator(energy_esys=esys))
phi_op = qt.Qobj(fluxonium.phi_operator(energy_esys=esys))
H_with_drive = [H, [n_op, lambda t, amp: amp * np.cos(r_f * t)]]
```

### Single Trace

```python
def simulate_populations(avg_photon, *, progress_bar="tqdm"):
    args = dict(amp=2 * g * np.sqrt(avg_photon))

    output = qt.fmmesolve(
        H_with_drive,
        psi0,
        2 * np.pi * times,
        c_ops=[n_op, phi_op],
        spectra_cb=[noise_spectrum, noise_spectrum],
        T=2 * np.pi * T,  # type: ignore
        args=args,
        options=dict(
            progress_bar=progress_bar, store_final_state=True, store_states=True
        ),
    )

    final_state = output.states[-1]
    p_g = float(np.real(qt.expect(qt.projection(qub_dim, 0, 0), final_state)))
    p_e = float(np.real(qt.expect(qt.projection(qub_dim, 1, 1), final_state)))
    p_other = max(0.0, 1.0 - p_g - p_e)

    return {
        "p_g": p_g,
        "p_e": p_e,
        "p_other": p_other,
        "output": output,
    }
```

```python
result = simulate_populations(avg_photon)
output = result["output"]

print(
    "steady state populations:",
    f"p_g={result['p_g']:.4f}, p_e={result['p_e']:.4f}, p_other={result['p_other']:.4f}",
)

p_g_trace = np.abs(
    [
        qt.expect(qt.projection(qub_dim, 0, 0), output.states[i].unit())
        for i in range(times.shape[0])
    ]
)
p_e_trace = np.abs(
    [
        qt.expect(qt.projection(qub_dim, 1, 1), output.states[i].unit())
        for i in range(times.shape[0])
    ]
)
```

```python
plt.plot(times, p_g_trace, label="ground state")
plt.plot(times, p_e_trace, label="excited state")
plt.plot(times, 1 - p_g_trace - p_e_trace, label="Other state")
plt.grid(True)
plt.ylim(0, 1)
plt.legend()
plt.show()
```

### Over photons

```python
from tqdm.auto import tqdm

photon_grid = np.arange(0, 301)
p_g_map = np.zeros((photon_grid.size, times.size), dtype=float)
p_e_map = np.zeros((photon_grid.size, times.size), dtype=float)
p_other_map = np.zeros((photon_grid.size, times.size), dtype=float)

for idx, n_bar in enumerate(tqdm(photon_grid)):
    result = simulate_populations(n_bar, progress_bar="")
    output = result["output"]

    p_g_map[idx] = np.abs(
        [
            qt.expect(qt.projection(qub_dim, 0, 0), output.states[i].unit())
            for i in range(times.shape[0])
        ]
    )
    p_e_map[idx] = np.abs(
        [
            qt.expect(qt.projection(qub_dim, 1, 1), output.states[i].unit())
            for i in range(times.shape[0])
        ]
    )
    p_other_map[idx] = np.clip(1 - p_g_map[idx] - p_e_map[idx], 0, 1)

```

```python
%matplotlib inline
extent = [times[0], times[-1], photon_grid[0], photon_grid[-1]]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

im0 = axes[0].imshow(p_g_map, aspect="auto", origin="lower", extent=extent)
axes[0].set_title("ground")
axes[0].set_xlabel("Time (us)")
axes[0].set_ylabel("Average photon number")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(p_e_map, aspect="auto", origin="lower", extent=extent)
axes[1].set_title("excited")
axes[1].set_xlabel("Time (us)")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(p_other_map, aspect="auto", origin="lower", extent=extent)
axes[2].set_title("other")
axes[2].set_xlabel("Time (us)")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()
```

## With Sample rate use Liouvillian

```python
import qutip as qt
from scqubits.core.fluxonium import Fluxonium

qub_dim = 20

fluxonium = Fluxonium(*params, flux=flx, cutoff=qub_dim + 10, truncated_dim=qub_dim)
evals = fluxonium.eigenvals()


# r_f = 5.3  # GHz
# g = 0.06


f_sample = 9.82  # GHz


T = 1 / r_f

psi0 = qt.basis(qub_dim, 1)
times = np.arange(0, 0.1e3, step=0.1 * f_sample)  # ns


dt_sample = 1 / f_sample  # ns

t_sample = np.arange(0, times[-1] + dt_sample, dt_sample)
dac_samples = np.cos(2 * np.pi * r_f * t_sample)


def dac_zoh(t, amp):
    # idx = np.clip(np.floor(t / dt_sample), 0, dac_samples.size - 1).astype(np.int32)
    # return amp * dac_samples[idx] + 0.1 * np.cos(2 * np.pi * f_sample * t)
    return amp * np.cos(2 * np.pi * r_f * t)


gamma1 = 1 / 1.3e1


esys = fluxonium.eigensys(evals_count=qub_dim)
H = qt.Qobj(fluxonium.hamiltonian(energy_esys=esys))
n_op = qt.Qobj(fluxonium.n_operator(energy_esys=esys))
phi_op = qt.Qobj(fluxonium.phi_operator(energy_esys=esys))
```

```python
oversample = 10

dt_signal = 1 / f_sample / oversample
t_signals = np.arange(0, times[-1] + dt_signal, dt_signal)

dac_signal = dac_zoh(t_signals, 1)

spectrum = np.fft.rfft(dac_signal)
freq = np.fft.rfftfreq(dac_signal.size, d=dt_signal)

plt.plot(freq, np.abs(spectrum))
plt.xlim(0, 1.6 * f_sample)
plt.axvline(r_f, color="r", linestyle="--", label="r_f")
plt.axvline(f_sample, color="k", linestyle=":", label="f_sample")
plt.grid(True)
plt.xlabel("Angular frequency")
plt.ylabel("|FFT|")
plt.legend()
plt.show()
```

### Single trace

```python
def simulate_populations(avg_photon, *, progress_bar="tqdm"):
    amp = 2 * g * np.sqrt(avg_photon)

    H_with_dac = [H, [n_op, dac_zoh]]
    rho0 = qt.ket2dm(psi0)

    output = qt.mesolve(
        H_with_dac,
        rho0,
        times,
        c_ops=[np.sqrt(gamma1) * n_op, np.sqrt(gamma1) * phi_op],
        c_ops=[np.sqrt(gamma1) * qt.destroy(qub_dim)],
        args=dict(amp=amp),
        options=dict(progress_bar=progress_bar, store_states=True),
    )

    p_g = np.array(
        [qt.expect(qt.projection(qub_dim, 0, 0), state) for state in output.states]
    )
    p_e = np.array(
        [qt.expect(qt.projection(qub_dim, 1, 1), state) for state in output.states]
    )
    p_other = np.clip(1 - p_g - p_e, 0, 1)

    return {
        "p_g": p_g,
        "p_e": p_e,
        "p_other": p_other,
        "output": output,
    }
```

```python
avg_photon = 0
result = simulate_populations(avg_photon)
```

```python
%matplotlib inline
plt.plot(times, result["p_g"], label="ground state")
plt.plot(times, result["p_e"], label="excited state")
plt.plot(times, result["p_other"], label="other state")
plt.grid(True)
plt.ylim(0, 1.1 * max(1, np.max(result["p_g"]), np.max(result["p_e"])))
plt.xlabel("Time (us)")
plt.ylabel("Population")
plt.legend()
plt.show()
```

### Over photons

```python
from tqdm.auto import tqdm

photon_grid = np.arange(0, 101)
p_g_map = np.zeros((photon_grid.size, times.size), dtype=float)
p_e_map = np.zeros((photon_grid.size, times.size), dtype=float)
p_other_map = np.zeros((photon_grid.size, times.size), dtype=float)

for idx, n_bar in enumerate(tqdm(photon_grid)):
    result = simulate_populations(n_bar, progress_bar="")
    p_g_map[idx] = result["p_g"]
    p_e_map[idx] = result["p_e"]
    p_other_map[idx] = result["p_other"]
```

```python
extent = [times[0], times[-1], photon_grid[0], photon_grid[-1]]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

im0 = axes[0].imshow(p_g_map, aspect="auto", origin="lower", extent=extent)
axes[0].set_title("ground")
axes[0].set_xlabel("Time (us)")
axes[0].set_ylabel("Average photon number")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(p_e_map, aspect="auto", origin="lower", extent=extent)
axes[1].set_title("excited")
axes[1].set_xlabel("Time (us)")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(p_other_map, aspect="auto", origin="lower", extent=extent)
axes[2].set_title("other")
axes[2].set_xlabel("Time (us)")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()
```
