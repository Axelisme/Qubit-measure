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
    display_name: .venv
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
    version: 3.9.23
---

```python
%load_ext autoreload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


%autoreload 2
import zcu_tools.experiment.v2 as ze
from zcu_tools.notebook.persistance import load_result
from zcu_tools.simulate import mA2flx
```

```python
qub_name = "Q12_2D[5]/Q4"

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
    r"../../../Database/Q12_2D[5]/Q4/Q4_dispersive_shift_gain0.050@-4.000mA_1.hdf5"
)
exp = ze.twotone.dispersive.DispersiveExperiment()
exp.load(filepath)

chi, kappa, fig = exp.analyze()
plt.show(fig)
fig.savefig(image_dir / "dispersive_shift.png")
plt.close(fig)
```

# AC stark shift

```python
filepath = r"../../../Database/Q12_2D[5]/Q4/Q4_ac_stark@-4.000mA_1.hdf5"
# signals, pdrs, fpts = load_data(filepath)
# fpts /= 1e6

exp = ze.twotone.ac_stark.AcStarkExperiment()
exp.load(filepath)
ac_coeff, fig = exp.analyze(chi=chi, kappa=kappa, cutoff=0.01)

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

exp = ze.twotone.singleshot.mist.MISTPowerDepSingleShot()
exp.load(filepath)
fig = exp.analyze(
    ac_coeff=ac_coeff,
    log_scale=True,
)


plt.show(fig)
fig.savefig(image_dir / (filepath.split("/")[-1].split("@")[0] + ".png"))
plt.close(fig)
```

## overnight

```python
filepath = "../../../Database/Q12_2D[5]/Q4/Q4_mist_overnight@-4.000mA_6.hdf5"

exp = ze.twotone.singleshot.mist_overnight.MISTPowerDepOvernight()
exp.load(filepath)
fig = exp.analyze(
    ac_coeff=ac_coeff,
)

plt.show(fig)
fig.savefig(f"{image_dir}/mist_overnight.png")
plt.close(fig)
```

# Power dep over flux

```python
sim_filepath = (
    f"../../../result/{qub_name}/data/branch_floquet/populations_over_flx.npz"
)
# sim_filepath = r"../../result/Q12_2D[3]/Q4/branch_populations.npz"


with np.load(sim_filepath) as data:
    sim_flxs = data["flxs"]
    sim_photons = data["photons"]
    branchs = data["branchs"]
    sim_populations = data["populations_over_flx"]
```

```python
filepath = (
    r"..\..\..\Database\Q12_2D[4]\Q4\2025\11\Data_1114\Q4_mist_flux_bare@4.000mA_1.hdf5"
)


exp = ze.twotone.mist.flux_dep.MistFluxDepExperiment()
As, pdrs, signals = exp.load(filepath)
fig = exp.analyze(mA_c=mA_c, period=period, ac_coeff=ac_coeff)

from zcu_tools.notebook.analysis.mist.branch import plot_cn_with_mist

plot_cn_with_mist(
    fig,
    flxs=sim_flxs,
    photons=sim_photons,
    populations_over_flx=sim_populations,
    critical_levels={0: 2.0, 1: 3.0},
    mist_flxs=mA2flx(As, mA_c, period),
)


fig.update_layout(height=800)
fig.write_image(image_dir / "branch_floquet/mist_over_flux_with_simulation.png")
fig.show()
```

```python

```
