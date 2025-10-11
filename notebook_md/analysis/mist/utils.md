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

# Collision Find

```python
import scqubits as scq
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np

fluxonium = scq.Fluxonium(
    EJ=4.0,
    EC=1.0,
    EL=1.0,
    flux=0.5,
    cutoff=40,
    truncated_dim=20,
)
coupler = scq.Transmon(
    EJ=25.0,
    EC=0.2,
    ng=0.0,
    ncut=40,
    truncated_dim=10,
)
H = scq.HilbertSpace([fluxonium, coupler])
H.add_interaction(g=0.5, op1=fluxonium.n_operator, op2=coupler.n_operator, add_hc=False)

evals, evecs = H.eigensys(evals_count=30)

H.generate_lookup()
bare_idxs = [H.bare_index(i) for i in range(evals.shape[0])]
```

```python
rfs = np.linspace(5.0, 7.0, 1000)
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
max_element = 0.05

# plt.plot(rfs, mod_evals, ".", markersize=1)
drive_op = H.op_in_dressed_eigenbasis(fluxonium.n_operator)
for i in range(len(colli_idx[0])):
    ci, cj1, cj2 = colli_idx[0][i], colli_idx1[i], colli_idx2[i]
    label1 = bare_idxs[cj1]
    label2 = bare_idxs[cj2]

    state1 = qt.Qobj(evecs[cj1].data)
    state2 = qt.Qobj(evecs[cj2].data)

    element = np.abs(drive_op.matrix_element(state1, state2))

    if label1[0] in [0, 1] or label2[0] in [0, 1]:
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
