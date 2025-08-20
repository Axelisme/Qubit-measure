```python
%load_ext autoreload
import os

import numpy as np
import matplotlib.pyplot as plt

%autoreload 2
from zcu_tools.notebook.persistance import load_result
from zcu_tools.notebook.analysis.branch import (
    plot_cn_over_flx,
    plot_populations_over_photon,
)
from zcu_tools.notebook.analysis.branch.full_quantum import (
    calc_branch_population,
    calc_branch_population_over_flux,
    make_hilbertspace,
)
```

```python
qub_name = "SF010"

os.makedirs(f"../../result/{qub_name}/image/branch", exist_ok=True)
os.makedirs(f"../../result/{qub_name}/web/branch", exist_ok=True)
```

```python
loadpath = f"../../result/{qub_name}/params.json"
_, params, mA_c, period, allows, data_dict = load_result(loadpath)

print(f"EJ: {params[0]:.3f} GHz, EC: {params[1]:.3f} GHz, EL: {params[2]:.3f} GHz")

if dispersive_cfg := data_dict.get("dispersive"):
    g = dispersive_cfg["g"]
    r_f = dispersive_cfg["r_f"]
    print(f"g: {g} GHz, r_f: {r_f} GHz")
elif "r_f" in allows:
    r_f = allows["r_f"]
    print(f"r_f: {r_f} GHz")
```

```python
# params = (7.0, 1.1, 1.4)
# r_f = 5.9
# g = 0.1


qub_dim = 15
qub_cutoff = 40
res_dim = 210

photons = np.arange(0, res_dim - 10)
```

```python
flx = 0.5
branchs = list(range(15))

hilbertspace = make_hilbertspace(params, r_f, qub_dim, qub_cutoff, res_dim, g, flx=flx)
populations_over_flx = calc_branch_population(hilbertspace, branchs, upto=photons[-1])
```

```python
fig = plot_populations_over_photon(branchs, photons, populations_over_flx)

fig.write_html(f"../../result/{qub_name}/web/branch/populations_at_phi{flx:.1f}.html")
fig.write_image(f"../../result/{qub_name}/image/branch/populations_at_phi{flx:.1f}.png")
fig.show()
```

```python
flxs = np.linspace(0, 0.5, 5)
branchs = [0, 1]
```

```python
populations_over_flx = calc_branch_population_over_flux(
    flxs,
    params,
    r_f,
    qub_dim,
    qub_cutoff,
    res_dim,
    g,
    upto=photons[-1],
    branchs=branchs,
    batch_size=30,
)
```

```python
np.savez_compressed(
    f"../../result/{qub_name}/branch_populations.npz",
    flxs=flxs,
    branchs=branchs,
    photons=photons,
    populations_over_flx=populations_over_flx,
)
```

```python
data = np.load(f"../../result/{qub_name}/branch_populations.npz")
flxs = data["flxs"]
photons = data["photons"]
populations_over_flx = data["populations_over_flx"]
```

```python
fig = plot_cn_over_flx(
    flxs, photons, populations_over_flx, critical_levels={0: 2, 1: 3}
)

fig.write_html(f"../../result/{qub_name}/web/branch/cn_over_flx.html")
fig.write_image(f"../../result/{qub_name}/image/branch/cn_over_flx.png")
fig.show()
```

```python

```
