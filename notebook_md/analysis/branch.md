```python
%load_ext autoreload
import os

import numpy as np
import matplotlib.pyplot as plt

%autoreload 2
from zcu_tools.notebook.persistance import load_result
from zcu_tools.notebook.analysis.branch import (
    branch_population,
    branch_population_over_flux,
    make_hilbertspace,
    plot_branch_population,
)
```

```python
qub_name = "SF010"

os.makedirs(f"../../result/{qub_name}/", exist_ok=True)
os.makedirs(f"../../result/{qub_name}/image", exist_ok=True)
```

```python
loadpath = f"../../result/{qub_name}/params.json"
_, params, mA_c, period, allows, data_dict = load_result(loadpath)
EJ, EC, EL = params

print(params)

if dispersive_cfg := data_dict.get("dispersive"):
    g = dispersive_cfg["g"]
    r_f = dispersive_cfg["r_f"]
elif "r_f" in allows:
    r_f = allows["r_f"]
```

```python
# params = (7.0, 1.1, 1.4)
# r_f = 5.9
# g = 0.1


qub_dim = 15
qub_cutoff = 40
res_dim = 210

branchs = list(range(15))
```

```python
flx = 0.5
hilbertspace = make_hilbertspace(params, r_f, qub_dim, qub_cutoff, res_dim, g, flx=flx)
populations = branch_population(hilbertspace, branchs, upto=100)
```

```python
fig, ax = plot_branch_population(branchs, populations)
fig.savefig(f"../../result/{qub_name}/image/branch_analysis_at_phi{flx:.1f}.png")
```

```python
flxs = np.linspace(0, 0.5, 1001)
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
    batch_size=30,
)
```

```python
ground_populations = populations[:, 0, :]
excited_populations = populations[:, 1, :]

# calculate the critical photon number
ground_cn = np.argmax(ground_populations >= 2, axis=1)
ground_cn[ground_cn == 0] = ground_populations.shape[1] - 1
excited_cn = np.argmax(excited_populations >= 3, axis=1)
excited_cn[excited_cn == 0] = excited_populations.shape[1] - 1
```

```python
# plot the critical photon number as a function of flux
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

df = flxs[1] - flxs[0]
ax1.imshow(
    ground_populations.T,
    interpolation="none",
    aspect="auto",
    origin="lower",
    extent=[flxs[0] - 0.5 * df, flxs[-1] + 0.5 * df, 0, populations.shape[2]],
)
ax1.plot(flxs, ground_cn, label="ground", marker=".", color="r")
ax1.set_title("Ground state")
ax2.imshow(
    excited_populations.T,
    interpolation="none",
    aspect="auto",
    origin="lower",
    extent=[flxs[0] - 0.5 * df, flxs[-1] + 0.5 * df, 0, populations.shape[2]],
)
ax2.plot(flxs, excited_cn, label="excited", marker=".", color="r")
ax2.set_title("Excited state")

ax1.legend()
ax2.legend()
fig.savefig(f"../../result/DesignR59/{qub_name}/image/branch_analysis.png")
```

```python
np.savez_compressed(
    f"../../result/DesignR59/{qub_name}/branch_analysis.npz",
    flxs=flxs,
    branchs=branchs,
    populations=populations,
)
```

```python

```
