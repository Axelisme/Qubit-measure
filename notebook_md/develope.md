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
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, cast
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import qutip as qt
from scqubits.core.fluxonium import Fluxonium
from scqubits.core.oscillator import Oscillator
from scqubits.core.hilbert_space import HilbertSpace
```

# Mist Simulation

```python
params = (5.0, 1.0, 1.0)

flx = 0.5
qub_dim = 40
qub_cutoff = 50
max_photon = 500
kappa = 5.2  # MHz

r_f = 5.0  # GHz

g = 50e-3  # GHz
```

# Single Coupling

```python
from zcu_tools.simulate.fluxonium.branch.floquet import FloquetBranchAnalysis

amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), 1e-3 * kappa)
photons = (amps / (2 * g)) ** 2


def calc_energies(branchs: List[int]) -> Dict[int, np.ndarray]:
    avg_times = np.linspace(0.0, 2 * np.pi / r_f, 100)

    fb_analysis = FloquetBranchAnalysis(
        params, r_f, g, flx=flx, qub_dim=qub_dim, qub_cutoff=qub_cutoff
    )

    fbasis_n = Parallel(n_jobs=-1)(
        delayed(fb_analysis.make_floquet_basis)(photon, precompute=avg_times)
        for photon in tqdm(photons, desc="Computing Floquet basis")
    )
    fbasis_n = cast(List[qt.FloquetBasis], fbasis_n)

    branch_infos = fb_analysis.calc_branch_infos(fbasis_n, branchs)
    branch_energies = fb_analysis.calc_branch_energies(fbasis_n, branch_infos)

    return {k: np.asarray(v) for k, v in branch_energies.items()}


branchs = list(range(25))
branch_energies = calc_energies(branchs)


resonator = Oscillator(r_f, truncated_dim=30)
fluxonium = Fluxonium(*params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim)
hilbertspace = HilbertSpace([fluxonium, resonator])
hilbertspace.add_interaction(
    g=g,
    op1=(fluxonium.n_operator(), fluxonium),
    op2=(resonator.annihilation_operator() + resonator.creation_operator(), resonator),
    add_hc=False,
)
E_n0 = hilbertspace.eigenvals(evals_count=qub_dim * 15)

hilbertspace.generate_lookup(ordering="LX")


def calc_E_n0_ij(i: int, j: int) -> np.ndarray:
    i_dressed = hilbertspace.dressed_index((i, 0))
    j_dressed = hilbertspace.dressed_index((j, 0))

    return E_n0[j_dressed] - E_n0[i_dressed]

```

```python
%matplotlib widget
from zcu_tools.notebook.analysis.mist.branch import round_to_nearest

fig, ax = plt.subplots()

transitions = {}
E_01 = branch_energies[1] - branch_energies[0]
E_01 += calc_E_n0_ij(0, 1) - E_01[0]
for i in (0, 1):
    for j in range(i + 1, np.max(branchs)):
        E_ij = branch_energies[j] - branch_energies[i]
        E_ij += calc_E_n0_ij(i, j) - E_ij[0]

        E_ij_mod = round_to_nearest(E_01, E_ij, r_f)

        transitions[f"{i} → {j}"] = E_ij_mod


ax.set_ylim(0.0, 1e3 * E_01[0] + 250)
# ax.set_ylim(0.0, 1e3 * r_f + 250)
# ax.axhline(1e3 * r_f, color="black", linestyle="--")


def calc_default_xy(photons: np.ndarray, E_ij: np.ndarray) -> tuple[float, float]:
    y = np.median(E_ij).item()
    x = photons[np.argmin(np.abs(E_ij - y))]
    return x, y


ylim = ax.get_ylim()
for name, E_ij in transitions.items():
    E_ij = 1e3 * E_ij

    mask = np.bitwise_and(E_ij > ylim[0], E_ij < ylim[1])
    if np.any(mask):
        (line,) = ax.plot(photons, E_ij)
        ax.annotate(
            name,
            xy=calc_default_xy(photons[mask], E_ij[mask]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=line.get_color(),
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                alpha=0.9,
                ec="none",
            ),
        )


plt.show(fig)
# savefig(fig, f"{image_dir}/ac_stark_populations.png", dpi=300)
# plt.close(fig)
```

```python
idx = np.argmin(np.abs(photons - 1))
chi = (
    branch_energies[1][idx]
    - branch_energies[1][0]
    - branch_energies[0][idx]
    + branch_energies[0][0]
)
print(f"chi = {1e3 * chi: .2f} MHz")
```

# Dual Coupling

```python
g1 = 0.4 * g  # GHz
g2 = -0.4 * g  # GHz
```

```python
from zcu_tools.simulate.fluxonium.branch.floquet import (
    FloquetDualCouplingBranchAnalysis,
)

amps = np.arange(0.0, (abs(g1) + abs(g2)) * np.sqrt(max_photon), 1e-3 * kappa)
photons = (amps / (abs(g1) + abs(g2))) ** 2


def calc_energies(branchs: List[int]) -> Dict[int, np.ndarray]:
    avg_times = np.linspace(0.0, 2 * np.pi / r_f, 100)

    fb_analysis = FloquetDualCouplingBranchAnalysis(
        params, r_f, g1, g2, flx=flx, qub_dim=qub_dim, qub_cutoff=qub_cutoff
    )

    fbasis_n = Parallel(n_jobs=-1)(
        delayed(fb_analysis.make_floquet_basis)(photon, precompute=avg_times)
        for photon in tqdm(photons, desc="Computing Floquet basis")
    )
    fbasis_n = cast(List[qt.FloquetBasis], fbasis_n)

    branch_infos = fb_analysis.calc_branch_infos(fbasis_n, branchs)
    branch_energies = fb_analysis.calc_branch_energies(fbasis_n, branch_infos)

    return {k: np.asarray(v) for k, v in branch_energies.items()}


branchs = list(range(25))
branch_energies = calc_energies(branchs)


resonator = Oscillator(r_f, truncated_dim=10)
fluxonium = Fluxonium(*params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim)
hilbertspace = HilbertSpace([fluxonium, resonator])
hilbertspace.add_interaction(
    g=g1,
    op1=(fluxonium.n_operator(), fluxonium),
    op2=(resonator.annihilation_operator() + resonator.creation_operator(), resonator),
    add_hc=False,
)
hilbertspace.add_interaction(
    g=g2,
    op1=(1j * fluxonium.phi_operator(), fluxonium),
    op2=(resonator.annihilation_operator() - resonator.creation_operator(), resonator),
    add_hc=False,
)
E_n0 = hilbertspace.eigenvals(evals_count=qub_dim * 10)

hilbertspace.generate_lookup(ordering="LX")


def calc_E_n0_ij(i: int, j: int) -> np.ndarray:
    i_dressed = hilbertspace.dressed_index((i, 0))
    j_dressed = hilbertspace.dressed_index((j, 0))

    return E_n0[j_dressed] - E_n0[i_dressed]

```

```python
%matplotlib widget
from zcu_tools.notebook.analysis.mist.branch import round_to_nearest

fig, ax = plt.subplots()

transitions = {}
E_01 = branch_energies[1] - branch_energies[0]
E_01 += calc_E_n0_ij(0, 1) - E_01[0]
for i in (0, 1):
    for j in range(i + 1, np.max(branchs)):
        E_ij = branch_energies[j] - branch_energies[i]
        E_ij += calc_E_n0_ij(i, j) - E_ij[0]

        E_ij_mod = round_to_nearest(E_01, E_ij, r_f)

        transitions[f"{i} → {j}"] = E_ij_mod


ax.set_ylim(1e3 * E_01[0] - 250, 1e3 * E_01[0] + 250)


def calc_default_xy(photons: np.ndarray, E_ij: np.ndarray) -> tuple[float, float]:
    y = np.median(E_ij).item()
    x = photons[np.argmin(np.abs(E_ij - y))]
    return x, y


ylim = ax.get_ylim()
for name, E_ij in transitions.items():
    E_ij = 1e3 * E_ij

    mask = np.bitwise_and(E_ij > ylim[0], E_ij < ylim[1])
    if np.any(mask):
        (line,) = ax.plot(photons, E_ij)
        ax.annotate(
            name,
            xy=calc_default_xy(photons[mask], E_ij[mask]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=line.get_color(),
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                alpha=0.9,
                ec="none",
            ),
        )


plt.show(fig)
# savefig(fig, f"{image_dir}/ac_stark_populations.png", dpi=300)
```

```python
idx = np.argmin(np.abs(photons - 1))
chi = (
    branch_energies[1][idx]
    - branch_energies[1][0]
    - branch_energies[0][idx]
    + branch_energies[0][0]
)
print(f"chi = {1e3 * chi: .2f} MHz")
```

```python

```
