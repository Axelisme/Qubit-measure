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
    FloquetBranchAnalysis,
    FloquetWithTLSBranchAnalysis,
)
```

# Load Parameters

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
flx = 0.3

qub_dim = 40
qub_cutoff = 80
max_photon = 150


def calc_populations_at_flx(
    branchs: List[int], flx: float, progress: bool = True
) -> Tuple[np.ndarray, Dict[int, List[float]]]:
    amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), rf_w)
    photons = (amps / (2 * g)) ** 2

    avg_times = np.linspace(0.0, 2 * np.pi / r_f, 100)

    fb_analysis = FloquetBranchAnalysis(
        params, r_f, g, flx=flx, qub_dim=qub_dim, qub_cutoff=qub_cutoff
    )

    fbasis_n = Parallel(n_jobs=-1)(
        delayed(fb_analysis.make_floquet_basis)(photon, precompute=avg_times)
        for photon in tqdm(
            photons, desc="Computing Floquet basis", disable=not progress
        )
    )

    branch_infos = fb_analysis.calc_branch_infos(fbasis_n, branchs, progress=progress)
    branch_populations = fb_analysis.calc_branch_populations(
        fbasis_n, branch_infos, avg_times, progress=progress
    )

    return photons, branch_populations
```

# Single Point

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

## With TLS

```python
r_f = 5.7945
rf_w = 0.006
g = 0.11
flx = 0.395

qub_dim = 40
qub_cutoff = 80
max_photon = 150

amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), rf_w)


def calc_populations_at_flx_with_tls(
    branchs: List[int], flx: float, E_tls: float, g_tls: float, progress: bool = True
) -> Tuple[np.ndarray, Dict[int, List[float]]]:
    amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), rf_w)
    photons = (amps / (2 * g)) ** 2

    avg_times = np.linspace(0.0, 2 * np.pi / r_f, 100)

    fb_analysis = FloquetWithTLSBranchAnalysis(
        params, r_f, g, E_tls, g_tls, flx=flx, qub_dim=qub_dim, qub_cutoff=qub_cutoff
    )

    fbasis_n = Parallel(n_jobs=-1)(
        delayed(fb_analysis.make_floquet_basis)(photon, precompute=avg_times)
        for photon in tqdm(
            photons, desc="Computing Floquet basis", disable=not progress
        )
    )

    branch_infos = fb_analysis.calc_branch_infos(fbasis_n, branchs, progress=progress)
    branch_populations = fb_analysis.calc_branch_populations(
        fbasis_n, branch_infos, avg_times, progress=progress
    )

    return photons, branch_populations
```

```python
%matplotlib inline

sweep_E_tls = np.linspace(0.1, 4.0, 10)
g_tls = 0.001

branchs = [0]

import matplotlib.pyplot as plt
from IPython.display import display
from zcu_tools.notebook.analysis.branch import calc_critical_photons

# cn_over_freq = np.full_like(sweep_E_tls, np.nan)
pop_over_freq = np.full((len(sweep_E_tls), len(amps)), np.nan)

fig, ax = plt.subplots()
# line = ax.plot(sweep_E_tls, np.full_like(sweep_E_tls, np.nan))[0]
im = ax.imshow(pop_over_freq, aspect="auto", origin="lower", interpolation="none")
dh = display(fig, display_id=True)
for i, E_tls in enumerate(tqdm(sweep_E_tls, desc="Sweeping E_tls")):
    photons, branch_populations = calc_populations_at_flx_with_tls(
        branchs, flx, E_tls, g_tls, progress=False
    )
    # cn = calc_critical_photons(photons, np.array(branch_populations[0]), 1.0)
    # cn_over_freq[i] = cn
    pop_over_freq[i] = branch_populations[0]

    # line.set_ydata(cn_over_freq)
    im.set_data(pop_over_freq)
    ax.relim(visible_only=True)
    ax.autoscale_view()
    dh.update(fig)
plt.close(fig)
```

```python
E_tls = 3.10
g_tls = 0.001

branchs = list(range(15))

photons, branch_populations = calc_populations_at_flx_with_tls(
    branchs, flx, E_tls, g_tls
)
```

```python
fig = plot_populations_over_photon(branchs, photons, branch_populations)
fig.show()
```

# Over flux

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
