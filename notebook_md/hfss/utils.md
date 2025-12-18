---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Find proper coupling length

```python
import numpy as np
import scipy as sp
```

```python
def kappa_func(coeff, rf, lc, lt) -> float:
    return coeff * rf * np.sin(np.pi / 2 * lc / lt) ** 2
```

```python
fpts = np.array([7.447])  # GHz
lcs = np.array([0.500])  # um
lts = np.array([4.000])  # um
kappas = np.array([2.42])  # MHz

coeff = np.mean(kappas / kappa_func(1, fpts, lcs, lts))
```

```python
want_rf = 5.9294  # GHz
lt = lts[0] * (fpts[0] / want_rf)  # um
print(lt)
```

```python
want_kappa = 1.0  # MHz

result = sp.optimize.root(
    lambda lc: kappa_func(coeff, want_rf, lc, lt) - want_kappa, 0.005
)
lc = result.x[0]
print(lc)
```

```python

```

```python

```
