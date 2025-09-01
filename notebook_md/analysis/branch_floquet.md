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
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import display

%autoreload 2
from zcu_tools.notebook.persistance import load_result
from zcu_tools.notebook.analysis.branch import (
    plot_cn_over_flx,
    plot_populations_over_photon,
    plot_chi_and_snr_over_photon,
)
from zcu_tools.simulate.fluxonium.branch.floquet import (
    FloquetBranchAnalysis,
    FloquetWithTLSBranchAnalysis,
)
from zcu_tools.notebook.analysis.design import calc_snr
```

# Load Parameters

```python
qub_name = "DesignR59"

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

# SNR power dependence

```python
r_f = 5.927
rf_w = 0.006
g = 0.1
flx = 0.5

qub_dim = 20
qub_cutoff = 60
max_photon = 70

photons, chi_over_n, snrs = calc_snr(
    params, r_f, g, flx, qub_dim, qub_cutoff, max_photon, rf_w
)
```

```python
fig, _ = plot_chi_and_snr_over_photon(photons, chi_over_n, snrs, qub_name, flx)

fig.savefig(f"../../result/{qub_name}/image/branch_floquet/snr_over_n.png")
plt.show()
```

# Single

```python
# r_f = 5.25
rf_w = 0.006
# g = 0.11
flx = 0.5

qub_dim = 40
qub_cutoff = 80
max_photon = 150

amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), rf_w)
photons = (amps / (2 * g)) ** 2


def calc_populations(
    branchs: List[int], progress: bool = True
) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
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

    return branch_populations
```

```python
branchs = list(range(15))

branch_populations = calc_populations(branchs)
```

```python
fig = plot_populations_over_photon(branchs, photons, branch_populations)

prefix = f"../../result/{qub_name}/web/branch_floquet"
fig.write_html(os.path.join(prefix, "populations_phi{flx:0.2f}.html"))
fig.write_image(os.path.join(prefix, "populations_phi{flx:0.2f}.png"))
fig.show()
```

# Sweep TLS freq

```python
r_f = 5.7945
rf_w = 0.006
g = 0.11
flx = 0.395

qub_dim = 40
qub_cutoff = 80
max_photon = 150

amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), rf_w)
photons = (amps / (2 * g)) ** 2


def calc_populations_with_tls(
    branchs: List[int], E_tls: float, g_tls: float, progress: bool = True
) -> Tuple[np.ndarray, Dict[int, List[float]]]:
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

E_tls_list = np.linspace(0.1, 4.0, 10)
g_tls = 0.001

branchs = [0]

pop_over_tls = np.full((len(E_tls_list), len(amps)), np.nan)

fig, ax = plt.subplots()
im = ax.imshow(
    pop_over_tls,
    aspect="auto",
    origin="lower",
    interpolation="none",
    vmin=0.0,
    vmax=2.0,
)
dh = display(fig, display_id=True)
for i, E_tls in enumerate(tqdm(E_tls_list, desc="Sweeping E_tls")):
    photons, branch_populations = calc_populations_with_tls(
        branchs, E_tls, g_tls, progress=False
    )
    pop_over_tls[i] = branch_populations[0]

    im.set_data(pop_over_tls)
    im.autoscale()
    dh.update(fig)
plt.close(fig)
```

```python
np.savez_compressed(
    f"../../result/{qub_name}/branch_floquet_populations_over_tls.npz",
    pop_over_tls=pop_over_tls,
    tls_freqs=E_tls_list,
    photons=photons,
    branchs=branchs,
)
```

# Sweep flux

```python
r_f = 5.7945
rf_w = 0.006
g = 0.11
flx = 0.3

qub_dim = 40
qub_cutoff = 80
max_photon = 150

amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), rf_w)
photons = (amps / (2 * g)) ** 2


def calc_populations_with_flx(
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

```python
flxs = np.linspace(0.0, 0.5, 5)
branchs = [0, 1]

pop_over_flx = []
for flx in tqdm(flxs, desc="Computing branch populations"):
    photons, branch_populations = calc_populations_with_flx(
        branchs, flx=flx, progress=False
    )

    pop_over_flx.append(list(branch_populations.values()))
pop_over_flx = np.array(pop_over_flx)
```

```python
np.savez_compressed(
    f"../../result/{qub_name}/branch_floquet_populations_over_flx.npz",
    flxs=flxs,
    branchs=branchs,
    photons=photons,
    populations_over_flx=pop_over_flx,
)
```

```python
data = np.load(f"../../result/{qub_name}/branch_floquet_populations.npz")
flxs = data["flxs"]
photons = data["photons"]
pop_over_flx = data["populations_over_flx"]
```

```python
fig = plot_cn_over_flx(flxs, photons, pop_over_flx, {0: 2, 1: 3})
# fig.update_layout(height=600, width=800)

# Save the figure
fig.write_html(f"../../result/{qub_name}/image/branch_floquet/cn_over_flx.html")
fig.write_image(f"../../result/{qub_name}/image/branch_floquet/cn_over_flx.png")

fig.show()
```

# Sweep r_f

```python
rf_w = 0.006
g = 0.11
flx = 0.5

qub_dim = 40
qub_cutoff = 80
max_photon = 150

amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), rf_w)
photons = (amps / (2 * g)) ** 2


def calc_populations_with_rf(
    branchs: List[int], r_f: float, progress: bool = True
) -> Tuple[np.ndarray, Dict[int, List[float]]]:
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

```python
%matplotlib inline
rfs = np.linspace(4.0, 7.0, 100)

branchs = [0, 1]


import matplotlib.pyplot as plt
from IPython.display import display

pop_over_rf = np.full((len(rfs), len(photons), 2), np.nan)

fig, (ax_g, ax_e) = plt.subplots(2, 1, sharex=True)
im_g = ax_g.imshow(
    pop_over_rf[..., 0].T,
    aspect="auto",
    origin="lower",
    interpolation="none",
    extent=[rfs[0], rfs[-1], photons[0], photons[-1]],
    vmin=0.0,
    vmax=2.0,
)
im_e = ax_e.imshow(
    pop_over_rf[..., 1].T,
    aspect="auto",
    origin="lower",
    interpolation="none",
    extent=[rfs[0], rfs[-1], photons[0], photons[-1]],
    vmin=0.0,
    vmax=3.0,
)
dh = display(fig, display_id=True)
for i, r_f in enumerate(tqdm(rfs, desc="Sweeping r_f")):
    if not np.any(np.isnan(pop_over_rf[i])):
        continue

    photons, branch_populations = calc_populations_with_rf(branchs, r_f, progress=False)
    pop_over_rf[i, :, 0] = branch_populations[0]
    pop_over_rf[i, :, 1] = branch_populations[1]

    im_g.set_data(pop_over_rf[..., 0].T)
    im_e.set_data(pop_over_rf[..., 1].T)
    im_g.autoscale()
    im_e.autoscale()

    dh.update(fig)
fig.savefig(
    f"../../result/{qub_name}/image/branch_floquet/populations_phi{flx:0.2f}_over_rf.png"
)
plt.close(fig)
```

```python
np.savez_compressed(
    f"../../result/{qub_name}/branch_floquet_populations_over_rf.npz",
    pop_over_rf=pop_over_rf,
    rfs=rfs,
    photons=photons,
    branchs=branchs,
)
```

```python

```
