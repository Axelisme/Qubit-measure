---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: axelenv13
    language: python
    name: python3
---

```python
from zcu_tools.notebook.persistance import load_result
from zcu_tools.simulate.fluxonium import FluxoniumPredictor

qub_name = "Si001"

loadpath = f"../../../result/{qub_name}/params.json"
_, params, mA_c, period, allows, data_dict = load_result(loadpath)

print(f"EJ: {params[0]:.3f} GHz, EC: {params[1]:.3f} GHz, EL: {params[2]:.3f} GHz")

if dispersive_cfg := data_dict.get("dispersive"):
    g = dispersive_cfg["g"]
    r_f = dispersive_cfg["r_f"]
    print(f"g: {g} GHz, r_f: {r_f} GHz")
elif "r_f" in allows:
    r_f = allows["r_f"]
    print(f"r_f: {r_f} GHz")

predictor = FluxoniumPredictor(loadpath)
```

```python
import scqubits as scq
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np

g = 0.1

fluxonium = scq.Fluxonium(
    *params,
    # EJ=4.0,
    # EC=1.0,
    # EL=1.0,
    flux=0.5,
    cutoff=40,
    truncated_dim=20,
)
evals, evecs = fluxonium.eigensys(evals_count=50)
```

# Collision Find

```python
import scqubits as scq
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np

g = 0.1

fluxonium = scq.Fluxonium(
    *params,
    # EJ=4.0,
    # EC=1.0,
    # EL=1.0,
    flux=0.5,
    cutoff=40,
    truncated_dim=10,
)
resonator = scq.Oscillator(
    E_osc=r_f,
    truncated_dim=50,
)
H = scq.HilbertSpace([fluxonium, resonator])
H.add_interaction(g=g, op1=fluxonium.n_operator, op2=resonator.creation_operator, add_hc=True)

evals, evecs = H.eigensys(evals_count=100)

H.generate_lookup()
bare_idxs = [H.bare_index(i) for i in range(evals.shape[0])]
```

```python
# rfs = np.linspace(5.9, 6.0, 100)
rfs = np.linspace(r_f - 0.2, r_f + 0.2, 1000)
thresholds = 0.001

mod_evals = np.mod(evals, rfs[:, np.newaxis])

# sort mod_evals
sort_idxs = np.argsort(mod_evals, axis=1)
sorted_mod_evals = np.take_along_axis(mod_evals, sort_idxs, axis=1)


# find two mod_eval that are closest to each other
eval_diff = np.diff(
    sorted_mod_evals,
    axis=1,
    append=(sorted_mod_evals[:, 0] + rfs)[:, np.newaxis],
)
colli_idx = np.where(eval_diff < thresholds)
colli_idx1 = sort_idxs[colli_idx[0], colli_idx[1]]
colli_idx2 = sort_idxs[colli_idx[0], colli_idx[1] + 1]
```

```python
max_element = 0.6

# plt.plot(rfs, mod_evals, ".", markersize=1)
drive_op = H.op_in_dressed_eigenbasis(resonator.creation_operator)
for i in range(len(colli_idx[0])):
    ci, cj1, cj2 = colli_idx[0][i], colli_idx1[i], colli_idx2[i]
    label1 = bare_idxs[cj1]
    label2 = bare_idxs[cj2]

    if label1[0] == label2[0]:
        continue

    if label1[0] not in [0, 1] and label2[0] not in [0, 1]:
        continue

    state1 = qt.Qobj(evecs[cj1].data)
    state2 = qt.Qobj(evecs[cj2].data)

    element = np.abs(drive_op.matrix_element(state1, state2))

    if element < 0.1 * max_element:
        continue

    plt.scatter(rfs[ci], mod_evals[ci, cj1], c="r", s=5)
    plt.text(
        rfs[ci],
        mod_evals[ci, cj1],
        str(label1) + "-" + str(label2),
        alpha=min(1.0, element / max_element),
    )
plt.show()
```

```python

```
