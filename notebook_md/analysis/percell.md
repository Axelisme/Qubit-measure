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
import scqubits as scq
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

from zcu_tools.notebook.persistance import load_result
```

```python
qub_name = "Q12_2D[3]/Q1"

server_ip = "005-writeboard"
port = 4999
```

```python
_, params, *_, results = load_result(f"../../result/{qub_name}/params.json")


r_f = results["dispersive"]["r_f"]
g = results["dispersive"]["g"]
```

```python
Nf = 20
Nr = 20

flxs = np.linspace(0.0, 0.51, 1001)

fluxonium = scq.Fluxonium(*params, flux=0.0, cutoff=40, truncated_dim=Nf)
resonator = scq.Oscillator(E_osc=r_f, truncated_dim=Nr)
hilbertspace = scq.HilbertSpace([fluxonium, resonator])
hilbertspace.add_interaction(
    g=g, op1=fluxonium.n_operator, op2=resonator.creation_operator, add_hc=True
)


def update_hilbertspace(flx):
    fluxonium.flux = flx


sweep = scq.ParameterSweep(
    hilbertspace=hilbertspace,
    paramvals_by_name={"flux": flxs},
    update_hilbertspace=update_hilbertspace,
    evals_count=Nf * Nr,
    subsys_update_info={"flux": [fluxonium]},
)
```

$$
P_{res} (n) = (1 - exp(-\beta \hbar \omega_r)) exp(-n \beta \hbar \omega_r) \\
n_{th}(\omega_j) = \frac{1}{exp(\beta \hbar \omega_j) - 1} \\
\Gamma^{up}_{l->l'} = \sum_{n,n'} P_{res}(n)\kappa n_{th} (\omega_{l',n'} - \omega_{l,n})\left|\langle l',n'\left|a^\dagger\right|l,n \rangle\right|^2 \\
\Gamma^{down}_{l->l'} = \sum_{n,n'} P_{res}(n)\kappa (n_{th} (-\omega_{l',n'} + \omega_{l,n}) + 1)\left|\langle l',n'\left|a\right|l,n \rangle\right|^2 \\
$$

```python
T = 200e-3  # K
kappa = 7e-3  # GHz

beta_hbar = sc.hbar / (sc.k * T) * 1e9


def P_res(n):
    return (1 - np.exp(-beta_hbar * r_f)) * np.exp(-n * beta_hbar * r_f)


def n_th(w_j):
    return 1 / (np.exp(beta_hbar * w_j) - 1)


Nmax = 10  # truncate the resonator space
ns = np.arange(0, Nmax)
P_res_ns = P_res(ns)


def percell(paramsweep: scq.ParameterSweep, paramindex_tuple: tuple, **kwargs):
    global kappa, ns, P_res_ns

    fluxonium: scq.Fluxonium = paramsweep.get_subsys(0)
    resonator: scq.Oscillator = paramsweep.get_subsys(1)
    evals = paramsweep["evals"][paramindex_tuple]
    evecs = paramsweep["evecs"][paramindex_tuple]
    bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]

    def get_esys(state: int):
        idxs = [paramsweep.dressed_index((state, n), paramindex_tuple) for n in ns]
        mask = np.array([idx is None for idx in idxs])

        # mask the None index
        idxs = np.array([idx if idx is not None else 0 for idx in idxs])

        En = evals[idxs]
        Vec_n = evecs[idxs]

        # fill the None index with nan
        En[mask] = np.nan

        return En, Vec_n

    # calculate the transition rate of 0-1 caused by percell effect
    Egs, Vgs = get_esys(0)
    Ees, Ves = get_esys(1)

    E_1n0n = Egs[:, None] - Ees[None, :]  # (ns, ns), from |1, n> to |0, n>

    # calculate the transition rate of 0-1 caused by up percell effect

    up_mask = E_1n0n > 0
    E_1n0n_up = E_1n0n.copy()
    E_1n0n_up[~up_mask] = np.inf
    n_ths = n_th(E_1n0n_up)  # (ns, ns)
    # calculate <0, n'|a^dag|1, n>
    ad_op = scq.identity_wrap(
        resonator.creation_operator, resonator, [fluxonium, resonator], evecs=bare_evecs
    )
    ad_1n0n = np.zeros((len(ns), len(ns)), dtype=complex)
    for ng in ns:
        for ne in ns:
            ad_1n0n[ng, ne] = Vgs[ng].dag() * ad_op * Ves[ne]

    Percell_up = np.sum(P_res_ns[None, :] * kappa * n_ths * np.abs(ad_1n0n) ** 2)

    # calculate the transition rate of 0-1 caused by down percell effect

    down_mask = E_1n0n < 0
    E_1n0n_down = -E_1n0n.copy()
    E_1n0n_down[~down_mask] = np.inf
    n_ths = n_th(E_1n0n_down)  # (ns, ns)
    # calculate <0, n'|a|1, n>
    a_op = scq.identity_wrap(
        resonator.annihilation_operator,
        resonator,
        [fluxonium, resonator],
        evecs=bare_evecs,
    )
    a_1n0n = np.zeros((len(ns), len(ns)), dtype=complex)
    for ng in ns:
        for ne in ns:
            a_1n0n[ng, ne] = Vgs[ng].dag() * a_op * Ves[ne]

    Percell_down = np.sum(P_res_ns[None, :] * kappa * n_ths * np.abs(a_1n0n) ** 2)

    return 1 / (Percell_up + Percell_down)
```

```python
sweep.add_sweep(percell, "percell")
```

```python
plt.plot(flxs, sweep["percell"])
plt.yscale("log")
plt.ylim(1e3, 1e7)
plt.xlabel("flux")
plt.ylabel("T1 (ns)")
plt.grid()
plt.show()
```

```python
np.savez("../../result/Q12_2D[3]/Q1/percell", flxs=flxs, percell=sweep["percell"])
```

```python

```
