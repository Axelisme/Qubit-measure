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
%load_ext autoreload
import numpy as np
import matplotlib.pyplot as plt

%autoreload 2
from zcu_tools.notebook.analysis.branch import (
    branch_population,
    branch_population_over_flux,
    plot_branch_population,
    make_hilbertspace,
)
```

```python
params = (7.0, 1.1, 1.4)
r_f = 5.9
g = 0.1

res_dim = 160
qub_cutoff = 40
qub_dim = 15
```

```python
flx = 0.0
branchs = list(range(10))

hilbertspace = make_hilbertspace(params, r_f, qub_dim, qub_cutoff, res_dim, g)
populations = branch_population(hilbertspace, branchs, upto=res_dim - 10)
```

```python
fig, ax = plot_branch_population(branchs, populations)
fig.savefig("../../result/DesignR59/image/int_branch_analysis.png")
```

```python
flxs = np.linspace(0, 0.5, 101)
branchs = [0, 1]
```

```python
populations = branch_population_over_flux(
    flxs,
    params,
    r_f,
    qub_dim,
    qub_cutoff,
    res_dim,
    g,
    upto=res_dim - 10,
    branchs=branchs,
)
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
    extent=[flxs.min(), flxs.max(), 0, populations.shape[2]],
)
ax1.set_title("Ground state")
ax2.imshow(
    excited_populations.T,
    interpolation="none",
    aspect="auto",
    origin="lower",
    extent=[flxs.min(), flxs.max(), 0, populations.shape[2]],
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
