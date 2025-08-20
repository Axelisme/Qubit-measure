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
import os
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from joblib import Parallel, delayed
from tqdm.auto import tqdm

%autoreload 2
from zcu_tools.notebook.persistance import load_result
from zcu_tools.notebook.analysis.branch import (
    plot_cn_over_flx,
    plot_populations_over_photon,
)
from zcu_tools.simulate.fluxonium.branch.floquet import (
    encode_floquet_hamiltonian,
    calc_branch_infos,
    calc_branch_populations,
)
```

```python
qub_name = "Q12_2D[3]/Q4"

os.makedirs(f"../../result/{qub_name}/image/branch_floquet", exist_ok=True)
os.makedirs(f"../../result/{qub_name}/web/branch_floquet", exist_ok=True)
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
r_f = 5.7945
rf_w = 0.006
g = 0.11
flx = 0.5

qub_dim = 60
qub_cutoff = 120
max_photon = 150


def calc_populations_at_flx(
    branchs: List[int], flx: float, progress: bool = True
) -> Tuple[np.ndarray, Dict[int, List[float]]]:
    amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), rf_w)
    photons = (amps / (2 * g)) ** 2

    avg_times = np.linspace(0.0, 2 * np.pi / r_f, 100)

    basis_maker = encode_floquet_hamiltonian(
        params, r_f, g, flx=flx, qub_dim=qub_dim, qub_cutoff=qub_cutoff
    )
    fbasis_n = Parallel(n_jobs=-1)(
        delayed(basis_maker)(photon, precompute=avg_times)
        for photon in tqdm(
            photons, desc="Computing Floquet basis", disable=not progress
        )
    )

    branch_infos = calc_branch_infos(fbasis_n, branchs, progress=progress)
    branch_populations = calc_branch_populations(
        fbasis_n, branch_infos, avg_times[::10], progress=progress
    )

    return photons, branch_populations
```

```python
branchs = list(range(15))

photons, branch_populations = calc_populations_at_flx(branchs, flx)
```

```python
fig = plot_populations_over_photon(branchs, photons, branch_populations)

fig.write_html(
    f"../../result/{qub_name}/web/branch_floquet/populations_phi{flx:0.2f}.html"
)
fig.write_image(
    f"../../result/{qub_name}/image/branch_floquet/populations_phi{flx:0.2f}.png"
)
fig.show()
```

```python
flxs = np.linspace(0.0, 0.5, 5)
branchs = [0, 1]

populations_over_flx = []
for flx in tqdm(flxs, desc="Computing branch populations"):
    photons, branch_populations = calc_populations_at_flx(
        branchs, flx=flx, progress=False
    )

    populations_over_flx.append(list(branch_populations.values()))
populations_over_flx = np.array(populations_over_flx)
```

```python
np.savez_compressed(
    f"../../result/{qub_name}/branch_floquet_populations.npz",
    flxs=flxs,
    branchs=branchs,
    photons=photons,
    populations_over_flx=populations_over_flx,
)
```

```python
data = np.load(f"../../result/{qub_name}/branch_floquet_populations.npz")
flxs = data["flxs"]
photons = data["photons"]
populations_over_flx = data["populations_over_flx"]
```

```python
fig = plot_cn_over_flx(flxs, photons, populations_over_flx, {0: 2, 1: 3})
# fig.update_layout(height=600, width=800)

# Save the figure
fig.write_html(f"../../result/{qub_name}/image/branch_floquet/cn_over_flx.html")
fig.write_image(f"../../result/{qub_name}/image/branch_floquet/cn_over_flx.png")

fig.show()
```

```python

```
