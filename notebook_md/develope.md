---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

```python
%load_ext autoreload
from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scqubits.core.fluxonium import Fluxonium
from scqubits.core.hilbert_space import HilbertSpace

%autoreload 2
import zcu_tools.notebook.persistance as zp
from zcu_tools.meta_tool import ExperimentManager
from zcu_tools.simulate import value2flux
from zcu_tools.notebook.analysis.t1_curve import charge_spectral_density
```

```python
chip_name = "Q12_2D[6]"
qub_name = "Q1"

result_dir = Path("..", "result", chip_name, qub_name)
param_path = result_dir / "params.json"

image_dir = result_dir / "image" / "simulation"
image_dir.mkdir(exist_ok=True, parents=True)
```

```python
result_dict = zp.load_result(f"{result_dir}/params.json")
fluxdepfit_dict = result_dict.get("fluxdep_fit")
assert fluxdepfit_dict is not None, "fluxdep_fit not found in result_dict"

params = (
    fluxdepfit_dict["params"]["EJ"],
    fluxdepfit_dict["params"]["EC"],
    fluxdepfit_dict["params"]["EL"],
)
flux_half = fluxdepfit_dict["flux_half"]
flux_int = fluxdepfit_dict["flux_int"]
flux_period = fluxdepfit_dict["flux_period"]

print("params = ", params, " GHz")
print("flux_half = ", flux_half)
print("flux_int = ", flux_int)
print("flux_period = ", flux_period)

sample_f = 9.58464
fwhm = 5.247  # MHz

g = 0.1  # GHz
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

```python
em = ExperimentManager(result_dir / "exps")
ml, md = em.use_flux(label="1.800mA", readonly=True)

flux = value2flux(1.8e-3, flux_half, flux_period)

image_flx_dir = image_dir / f"{em.label}_flux{flux:.2f}"
image_flx_dir
```

```python
qub_cutoff = 40
qub_dim = 15

fluxonium = Fluxonium(*params, flux=flux, cutoff=qub_cutoff, truncated_dim=qub_dim)
hilbertspace = HilbertSpace([fluxonium])
hilbertspace.generate_lookup()

evals = hilbertspace.eigenvals()
f01 = evals[1] - evals[0]
print(f"f01 = {f01:.2f} GHz")
H_q = 2 * np.pi * hilbertspace.hamiltonian()
F_n = (
    2
    * np.pi
    * hilbertspace.op_in_dressed_eigenbasis(fluxonium.n_operator, truncated_dim=qub_dim)
)
F_phi = (
    2
    * np.pi
    * hilbertspace.op_in_dressed_eigenbasis(
        fluxonium.phi_operator, truncated_dim=qub_dim
    )
)

Q_cap = 4e5
Temp = 63e-3  # K
A_phi = 1e-3 * 2 * np.pi
omega_ir = 2 * np.pi * 1e-3
EC = params[1]
kappa = 2 * np.pi * fwhm * 1e-3


def charge_noise_sp(w: np.ndarray) -> np.ndarray:
    # 4e4 is the Qality facotor of dielectric loss, under 63mK
    mask = w > 0
    result = np.zeros_like(w)
    result[mask] = charge_spectral_density(w[mask], Temp, EC) / Q_cap
    return result


def flux_noise_sp(w: np.ndarray) -> np.ndarray:
    mask = w > 0
    result = np.zeros_like(w)
    result[mask] = A_phi**2 / np.maximum(np.abs(w[mask]), omega_ir)
    return result


H_q
```

## Single

```python
qf = f01

amp = np.sqrt(0) * kappa / (2 * np.pi)
print(f"amp: {1e3 * amp:.2f} MHz")


def round_freq(
    t1: float, t2: float, tol: float = 1
) -> tuple[float, float, float, float]:
    """
    find the largest common divisor of two periods
    let t1 ~= a*dt, t2 ~= b*dt, return a*dt, b*dt, dt, lcm(t1, t2)
    """
    a, b = (t1, t2) if t1 > t2 else (t2, t1)
    while b > tol:
        a, b = b, a % b
    dt = a
    n1 = round(t1 / dt)
    n2 = round(t2 / dt)
    return n1 * dt, n2 * dt, dt, math.lcm(n1, n2) * dt


r_qf_t = 1 / qf
r_rf_t = 1 / bare_rf
r_dt = 0.1 / bare_rf
r_T = 1 / bare_rf
# r_qf_t, r_rf_t, r_dt, r_T = round_freq(1 / qf, 1 / bare_rf, 0.8e-3)

r_qf = 1 / r_qf_t
r_rf = 1 / r_rf_t
times = np.arange(0.0, 50.0, r_dt)
print(
    f"qf_err: {1e3 * (qf - r_qf): .2f} MHz",
    f"\nrf_err: {1e3 * (bare_rf - r_rf): .2f} MHz",
    f"\ndt = {r_dt:.5f} ns, T = {r_T:.5f} ns",
    f"\nnumber of points = {len(times)}",
)


n_eps = 2 * np.pi * amp
w_rd = 2 * np.pi * r_rf
w_qd = 2 * np.pi * r_qf

avg_n = (n_eps / kappa) ** 2
print(f"avg_n = {avg_n:.2f}")

n_drive = -2 * n_eps / kappa * g
q_drive = 2 * np.pi * 0.005  # GHz

H = qt.QobjEvo(  # type: ignore
    # [H_q, [F_n, "n_drive * sin(w_rd * t) + q_drive * cos(w_qd * t)"]],
    [H_q, [F_n, "n_drive * sin(w_rd * t)"]],
    # [H_q, [F_n, "q_drive * cos(w_qd * t)"]],
    args={"n_drive": n_drive, "w_rd": w_rd, "q_drive": q_drive, "w_qd": w_qd},
)
```

```python
psi0 = qt.basis(qub_dim, 0)
psi1 = qt.basis(qub_dim, 1)
psi9 = qt.basis(qub_dim, 9)


result = qt.fmmesolve(
    H,
    psi1,
    times,
    # c_ops=[F_n, F_phi],
    c_ops=[F_phi],
    e_ops=[psi0.proj(), psi1.proj(), psi9.proj()],
    # spectra_cb=[charge_noise_sp, flux_noise_sp],
    spectra_cb=[flux_noise_sp],
    T=r_T,  # type: ignore
    options=dict(progress_bar="tqdm"),
)
```

```python
fig, ax1 = plt.subplots()
ax1.plot(times, result.expect[0], label="g")
ax1.plot(times, result.expect[1], label="e")
ax1.plot(times, result.expect[2], label="4")
ax1.plot(times, 1 - (np.sum([*result.expect], axis=0)), label="o")
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Probability")
ax1.set_ylim(-0.01, 1.01)
ax1.legend()
plt.show()

```

# Over photon

```python
H = [H_q, [F_n, "n_drive * sin(w_rd * t)"]]


def simulate_population(
    times: np.ndarray, photon: float, track_state: list[int]
) -> np.ndarray:
    n_eps = 2 * np.pi * np.sqrt(photon) * kappa / (2 * np.pi)

    w_rd = 2 * np.pi * bare_rf
    n_drive = -2 * n_eps / kappa * g

    psi_0 = qt.basis(qub_dim, 1)
    e_ops = [qt.basis(qub_dim, i).proj() for i in track_state]

    result = qt.fmmesolve(
        H,
        psi_0,
        times,
        c_ops=[F_phi],
        e_ops=e_ops,
        spectra_cb=[flux_noise_sp],
        T=1 / bare_rf,  # type: ignore
        args={"n_drive": n_drive, "w_rd": w_rd},
        options=dict(progress_bar=""),
    )

    return np.stack([result.expect[i] for i in range(len(e_ops))])

```

```python
from joblib import Parallel, delayed
from tqdm.auto import tqdm

times = np.arange(0.0, 50.0, 0.1 / bare_rf)
photons = np.linspace(0, 400, 400)
track_states = [0, 1, 9]

_result = Parallel(n_jobs=4)(
    delayed(simulate_population)(times, photon, track_states)
    for photon in tqdm(photons, desc="Simulation")
)
result = np.array(_result)  # (photons, track_states, times)
```

```python
import matplotlib.pyplot as plt

mean_result = np.mean(result[..., -int(0.1 * len(times)) :], axis=-1)
# mean_result = result[..., -1]  # (photons, times)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.imshow(
    result[:, 1, :].T,
    aspect="auto",
    cmap="RdBu_r",
    origin="lower",
    interpolation="none",
    extent=[photons[0], photons[-1], times[0], times[-1]],
)
ax1.set_ylabel("Time (ns)")

for i, state in enumerate(track_states):
    # if np.max(mean_result[:, i]) < 0.05:
    #     continue
    ax2.plot(photons, mean_result[:, i], label=f"state {state}")
other_pop = 1 - np.sum(mean_result, axis=1)
ax2.plot(photons, other_pop, label="other")
ax2.set_xlabel("Average Photon number")
ax2.set_ylabel("Population")
ax2.legend()
plt.show()
fig.savefig(f"{image_dir}/population_over_photons2.png", dpi=300)

```

```python

```
