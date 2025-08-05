---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import scqubits as scq
import numpy as np

import matplotlib.pyplot as plt
from scqubits import Fluxonium, HilbertSpace, Oscillator
from joblib import Parallel, delayed
```

```python
r_f = 5.9
g = 0.1
params = (7.0, 1.1, 1.4)
flx = 0.0
```

```python
res_dim = 210
qub_cutoff = 40
qub_dim = 15


resonator = Oscillator(r_f, truncated_dim=res_dim)
fluxonium = Fluxonium(*params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim)
hilbertspace = HilbertSpace([fluxonium, resonator])
hilbertspace.add_interaction(
    g=g, op1=fluxonium.n_operator, op2=resonator.creation_operator, add_hc=True
)
hilbertspace.generate_lookup(ordering="LX")
```

```python
bare_esys_dag_array = np.array(
    [
        np.sqrt(j) * hilbertspace.bare_productstate((j, m)).dag().full()
        for j in range(qub_dim)
        for m in range(res_dim)
    ]
)
```

```python
def calc_population(evec: np.ndarray) -> float:
    return np.sum(np.abs(np.dot(bare_esys_dag_array, evec)) ** 2)


def branch_population(
    hilbertspace: HilbertSpace, branchs: np.ndarray, upto: int = -1
) -> np.ndarray:
    r"""
    Calculate the average population of the states in branchs upto provided photon number

    Using equation: P(i, n) = \sum_{j, m} j*|<j,m|\dash{i,n}>|^2
    """
    fluxonium, resonator = hilbertspace.subsystem_list
    qub_dim = fluxonium.truncated_dim
    res_dim = resonator.truncated_dim

    _, evecs = hilbertspace.eigensys(evals_count=qub_dim * res_dim)

    def _calc_population(i, n) -> float:
        evec_in = evecs[hilbertspace.dressed_index((i, n))].full()
        return calc_population(evec_in)

    populations = np.array(
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(_calc_population)(b, n) for b in branchs for n in range(upto)
        )
    ).reshape(len(branchs), upto)

    return populations

```

```python
branchs = list(range(15))
populations = branch_population(hilbertspace, branchs, upto=100)
```

```python
for b in branchs:
    pop_b = populations[b]
    if np.ptp(pop_b) > 1.0:
        color = None
        label = f"Branch {b}"
    else:
        color = "lightgrey"
        label = None
    plt.plot(populations[b], label=label, color=color)
plt.legend()
plt.grid()
plt.savefig("../../result/DesignR59/image/int_branch_analysis.png")
plt.show()
```

```python
flxs = np.linspace(0, 0.5, 101)


def update_hilbertspace(flx: float) -> None:
    fluxonium.flux = flx


sweep = scq.ParameterSweep(
    hilbertspace=hilbertspace,
    paramvals_by_name={"flux": flxs},
    update_hilbertspace=update_hilbertspace,
    evals_count=qub_dim * res_dim,
    subsys_update_info={"flux": [fluxonium]},
    labeling_scheme="LX",
)
```

```python
upto = 200
branchs = [0, 1]


def get_branch_populations(
    paramsweep: scq.ParameterSweep, paramindex_tuple: tuple, **kwargs
) -> np.ndarray:
    # (qub_dim * res_dim, (qub_dim * res_dim, 1))
    evecs = paramsweep["evecs"][paramindex_tuple]

    def _calc_population(b, n) -> float:
        dressed_idx = paramsweep.dressed_index((b, n), paramindex_tuple)
        return calc_population(evecs[dressed_idx].full())

    populations = np.array(
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(_calc_population)(b, n) for b in branchs for n in range(upto)
        )
    ).reshape(len(branchs), upto)

    return populations

```

```python
sweep.add_sweep(get_branch_populations, sweep_name="branch_populations")
```

```python
populations = sweep["branch_populations"]
populations.shape
```

```python
ground_populations = populations[:, 0, :]
excited_populations = populations[:, 1, :]


# fig, ax = plt.subplots(figsize=(10, 5))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

ax1.imshow(
    ground_populations.T,
    interpolation="none",
    aspect="auto",
    origin="lower",
    extent=[flxs.min(), flxs.max(), 0, upto],
)
ax1.set_title("Ground state")
ax2.imshow(
    excited_populations.T,
    interpolation="none",
    aspect="auto",
    origin="lower",
    extent=[flxs.min(), flxs.max(), 0, upto],
)
ax2.set_title("Excited state")

plt.savefig("../../result/DesignR59/image/branch_analysis.png")
plt.show()
```

```python
# calculate the critical photon number

ground_cn = np.argmax(ground_populations >= 2, axis=1)
ground_cn[ground_cn == 0] = ground_populations.shape[1] - 1
excited_cn = np.argmax(excited_populations >= 3, axis=1)
excited_cn[excited_cn == 0] = excited_populations.shape[1] - 1

# plot the critical photon number as a function of flux
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(flxs, ground_cn, label="ground")
ax.plot(flxs, excited_cn, label="excited")
ax.legend()
plt.savefig("../../result/DesignR59/image/branch_analysis_cn.png")
plt.show()
```

```python

```
