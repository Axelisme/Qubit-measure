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

r_f = 5.9
g = 0.1
detune = 1e-4

qub_dim = 10
res1_dim = 10
res2_dim = 10

fluxonium = scq.Fluxonium(
    *(4.0, 1.0, 1.0), flux=0.5, cutoff=qub_dim + 10, truncated_dim=qub_dim
)
resonator1 = scq.Oscillator(r_f + detune, truncated_dim=res1_dim)
resonator2 = scq.Oscillator(r_f - detune, truncated_dim=res2_dim)
hilbertspace = scq.HilbertSpace([fluxonium, resonator1, resonator2])
hilbertspace.add_interaction(
    g=g, op1=fluxonium.n_operator, op2=resonator1.creation_operator, add_hc=True
)
hilbertspace.add_interaction(
    g=-g, op1=fluxonium.n_operator, op2=resonator2.creation_operator, add_hc=True
)
hilbertspace.generate_lookup(ordering="LX")
```

```python
evals = hilbertspace.eigenvals(evals_count=qub_dim * res1_dim * res2_dim)
E_000 = evals[hilbertspace.dressed_index((0, 0, 0))]
E_100 = evals[hilbertspace.dressed_index((1, 0, 0))]
E_001 = evals[hilbertspace.dressed_index((0, 0, 1))]
E_010 = evals[hilbertspace.dressed_index((0, 1, 0))]
E_101 = evals[hilbertspace.dressed_index((1, 0, 1))]
E_110 = evals[hilbertspace.dressed_index((1, 1, 0))]
```

```python
rf1_0 = E_010 - E_000
rf1_1 = E_110 - E_100
rf2_0 = E_001 - E_000
rf2_1 = E_101 - E_100
d1 = rf1_1 - rf1_0
d2 = rf2_1 - rf2_0
1e3 * d1, 1e3 * d2
```

```python

```
