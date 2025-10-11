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

# Find proper coupling length

```python
import numpy as np
import scipy as sp
```

```python
def kappa_func(coeff, rf, lc, lt) -> float:
    return coeff * rf * np.sin(np.pi / 2 * lc / lt)
```

```python
fpts = np.array([7.447])
lcs = np.array([0.500])
lts = np.array([4.000])
kappas = np.array([2.4861])

coeff = np.mean(kappas / kappa_func(1, fpts, lcs, lts))
```

```python
want_kappa = 1.0
# lt = 5.344
lt = 5.394
rf = 5.9294 * (5.344 / lt)

result = sp.optimize.root(lambda lc: kappa_func(coeff, rf, lc, lt) - want_kappa, 0.005)
want_lc = result.x[0]
print(want_lc)
```
