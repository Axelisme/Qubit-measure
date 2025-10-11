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

# Two qubit gate

```python
import scqubits as scq
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np

N = 30

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
bare_evals, bare_evecs = H.eigensys(evals_count=N)
H.generate_lookup()
bare_idxs = [[H.dressed_index((i, j)) for j in range(2)] for i in range(2)]

H = scq.HilbertSpace([fluxonium, coupler])
H.add_interaction(g=0.5, op1=fluxonium.n_operator, op2=coupler.n_operator, add_hc=False)
dressed_evals, dressed_evecs = H.eigensys(evals_count=N)
H.generate_lookup()
dressed_idxs = [[H.dressed_index((i, j)) for j in range(2)] for i in range(2)]
```

|000>, |0(+)>, |0(-)>, |011>

```python

```
