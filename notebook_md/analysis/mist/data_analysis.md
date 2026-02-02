---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: zcu-tools
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.25
---

```python
%load_ext autoreload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


%autoreload 2
import zcu_tools.experiment.v2 as ze
from zcu_tools.notebook.persistance import load_result
from zcu_tools.simulate import mA2flx, flx2mA
```

```python
qub_name = "Q12_2D[6]/Q1"

result_dir = Path(f"../../../result/{qub_name}")
image_dir = result_dir / "image" / "mist_data_analysis" / "-4.000mA"
image_dir.mkdir(parents=True, exist_ok=True)
```

```python
_, params, mA_c, period, allows, _ = load_result(f"{result_dir}/params.json")
EJ, EC, EL = params

print(allows)

if "r_f" in allows:
    r_f = allows["r_f"]

if "sample_f" in allows:
    sample_f = allows["sample_f"]
```

# Dispersive shift

```python
filepath = (
    r"../../../Database/Q12_2D[5]/Q1/Q1_dispersive_shift_gain0.050@-2.600mA_3.hdf5"
)
exp = ze.twotone.dispersive.DispersiveExp()
exp.load(filepath)

chi, kappa, fig = exp.analyze()
plt.show(fig)
fig.savefig(image_dir / "dispersive_shift.png")
plt.close(fig)
```

# CKP

```python
filepath = [
    r"../../../Database/Q12_2D[6]/Q1/2026/01/Data_0131/Q1_ckp@1.800mA_ground_2.hdf5",
    r"../../../Database/Q12_2D[6]/Q1/2026/01/Data_0131/Q1_ckp@1.800mA_excited_2.hdf5",
]
exp = ze.twotone.ckp.CKP_Exp()
exp.load(filepath)

chi, kappa, readout_f, fig = exp.analyze()
plt.show(fig)
fig.savefig(image_dir / "dispersive_shift.png")
plt.close(fig)
```

# AC stark shift

```python
filepath = (
    r"../../../Database/Q12_2D[6]/Q1/2026/01/Data_0131/Q1_ac_stark@1.800mA_1.hdf5"
)

exp = ze.twotone.ac_stark.AcStarkExp()
exp.load(filepath)
ac_coeff, fig = exp.analyze(chi=chi, kappa=kappa, cutoff=0.1)

plt.show(fig)
fig.savefig(image_dir / "ac_stark.png")
plt.close(fig)
```

# Power dep

```python
%matplotlib inline
filepath = (
    "../../../Database/Q12_2D[5]/Q4/Q4_mist_g_singleshot_short@-4.000mA_2.hdf5"
    # "../../../Database/Q12_2D[5]/Q4/Q4_mist_e_singleshot_short@-4.000mA_3.hdf5"
    # "../../../Database/Q12_2D[5]/Q4/Q4_mist_g_singleshot_short@-0.650mA_1.hdf5"
    # "../../../Database/Q12_2D[5]/Q4/Q4_mist_e_singleshot_short@-0.650mA_2.hdf5"
)

exp = ze.singleshot.mist.PowerDepExp()
exp.load(filepath)
fig = exp.analyze(
    ac_coeff=ac_coeff,
    log_scale=True,
)


plt.show(fig)
fig.savefig(image_dir / (filepath.split("/")[-1].split("@")[0] + ".png"))
plt.close(fig)
```

# Power dep over flux

```python
sim_filepath = f"{result_dir}/data/branch_floquet/populations_over_flx.npz"
# sim_filepath = r"../../result/Q12_2D[3]/Q4/branch_populations.npz"


with np.load(sim_filepath) as data:
    sim_flxs = data["flxs"]
    sim_photons = data["photons"]
    branchs = data["branchs"]
    sim_populations = data["populations_over_flx"]
```

```python
filepaths = [
    r"../../../Database/Q12_2D[5]/Q1/Q1_mist_over_flux@-7.000mA_1.hdf5",
    r"../../../Database/Q12_2D[5]/Q1/Q1_autofluxdep_onlyfreq@-2.500mA_mist_g_signals_g_2.hdf5",
    # r"../../../Database/Q12_2D[5]/Q1/Q1_mist_over_flux@-2.600mA_1.hdf5",
    # r"../../../Database/Q12_2D[5]/Q1/Q1_autofluxdep_onlyfreq@-2.500mA_mist_e_signals_e_2.hdf5",
]
e_filepaths = [
    r"../../../Database/Q12_2D[5]/Q1/Q1_autofluxdep_onlyfreq@-2.500mA_mist_e_signals_e_2.hdf5"
]

# ac_coeff = 1e4
map_flxs = 1 - sim_flxs

from zcu_tools.notebook.analysis.mist.branch import plot_cn_with_mist
from zcu_tools.notebook.analysis.fluxdep import add_secondary_xaxis
from plotly.subplots import make_subplots

exp = ze.mist.flux_dep.MistFluxDepExp()

# fig = go.Figure()
fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1)
for filepath in filepaths:
    exp.load(filepath)
    fig = exp.analyze(
        mA_c=mA_c,
        period=period,
        ac_coeff=ac_coeff,
        fig=fig,
        secondary_xaxis=False,
        auto_range=False,
        row=1,
        col=1,
    )
for filepath in e_filepaths:
    exp.load(filepath)
    fig = exp.analyze(
        mA_c=mA_c,
        period=period,
        ac_coeff=ac_coeff,
        fig=fig,
        secondary_xaxis=False,
        auto_range=False,
        row=2,
        col=1,
    )

plot_cn_with_mist(
    fig,
    flxs=sim_flxs,
    photons=sim_photons,
    populations_over_flx=sim_populations,
    critical_levels={0: 0.5, 1: 1.5},
    mist_flxs=map_flxs,
    row=1,
    col=1,
)
if e_filepaths:
    plot_cn_with_mist(
        fig,
        flxs=sim_flxs,
        photons=sim_photons,
        populations_over_flx=sim_populations,
        critical_levels={0: 0.5, 1: 1.5},
        mist_flxs=map_flxs,
        row=2,
        col=1,
    )

add_secondary_xaxis(fig, map_flxs, flx2mA(map_flxs, mA_c, period), row=2, col=1)

fig.update_layout(height=800)
fig.write_image(
    result_dir / "image" / "branch_floquet" / "mist_over_flux_with_simulation.png"
)
fig.show()
```

```python

```
