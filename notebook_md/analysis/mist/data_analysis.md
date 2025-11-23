---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: axelenv13
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
    version: 3.13.5
---

```python
%load_ext autoreload
import os

import matplotlib.pyplot as plt
import numpy as np

%autoreload 2
from zcu_tools.utils.datasaver import load_data
import zcu_tools.experiment.v2 as ze
from zcu_tools.notebook.persistance import load_result
from zcu_tools.simulate import mA2flx
```

```python
qub_name = "Q12_2D[4]/Q4"
```

```python
loadpath = f"../../../result/{qub_name}/params.json"
_, params, mA_c, period, allows, _ = load_result(loadpath)
EJ, EC, EL = params

print(allows)

if "r_f" in allows:
    r_f = allows["r_f"]

if "sample_f" in allows:
    sample_f = allows["sample_f"]
```

# Reset

```python
xname = r"$|1, 0\rangle -> |2, 0\rangle$"
yname = r"$|2, 0\rangle -> |0, 1\rangle$"
```

```python
filepath = (
    "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_mux_reset_freq@-0.417mA_9.hdf5"
)
signals, fpts1, fpts2 = load_data(filepath)

fpts1 /= 1e6
fpts2 /= 1e6
```

```python
ze.twotone.reset.dual_tone.FreqExperiment().analyze(
    result=(fpts1, fpts2, signals), xname=xname, yname=yname
)
```

## reset gain

```python
filepath = (
    "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_mux_reset_gain@-0.417mA_3.hdf5"
)
signals, pdrs1, pdrs2 = load_data(filepath)
```

```python
ze.twotone.reset.dual_tone.PowerExperiment().analyze(
    result=(pdrs1, pdrs2, signals), xname=xname, yname=yname
)
```

## reset time

```python
filepath = (
    "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_mux_reset_time@-0.417mA_4.hdf5"
)
signals, Ts, _ = load_data(filepath)

Ts *= 1e6
```

```python
ze.twotone.reset.dual_tone.LengthExperiment().analyze(result=(Ts, signals))
```

# Dispersive shift

```python
filepath = r"C:\Users\QEL\Desktop\MeasureScriptX\QuantumMeasurementProcedures\Members\axel\Qubit-measure\Database\Q12_2D[4]\Q4\2025\11\Data_1114\R4_dispersive@4.000mA_1.hdf5"
signals, fpts, _ = load_data(filepath)
fpts /= 1e6

chi, kappa, fig = ze.twotone.dispersive.DispersiveExperiment().analyze(
    result=(fpts, signals.T)
)
plt.show(fig)
plt.close(fig)
```

# AC stark shift

```python
filepath = r"C:\Users\QEL\Desktop\MeasureScriptX\QuantumMeasurementProcedures\Members\axel\Qubit-measure\Database\Q12_2D[4]\Q4\2025\11\Data_1114\Q4_ac_stark@4.000mA_1.hdf5"
signals, pdrs, fpts = load_data(filepath)
fpts /= 1e6

ac_coeff, fig = ze.twotone.ac_stark.AcStarkExperiment().analyze(
    result=(pdrs, fpts, signals), chi=chi, kappa=kappa, cutoff=0.04
)
plt.show(fig)
plt.close(fig)
```

# Power dep

```python
filepath = (
    # "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_abnormal_pdr@-0.417mA_1.hdf5"
    "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_abnormal_pdr@-0.417mA_2.hdf5"
)
signals, pdrs, _ = load_data(filepath)
signals = signals.T
```

```python
%matplotlib inline
ze.twotone.mist.MISTPowerDep().analyze(
    result=(pdrs, signals),
    # g0=e0,
    # e0=g0,
    ac_coeff=ac_coeff,
)
```

## overnight

```python
filepath = "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_mist_pdr_overnight@-0.417mA_2.hdf5"
g_signals, pdrs, iters = load_data(filepath)
g_signals = g_signals.T
```

```python
ze.twotone.mist.MISTPowerDepOvernight().analyze(
    result=(iters, pdrs, g_signals),
    # g0=g0,
    # e0=e0,
    ac_coeff=ac_coeff,
)
```

```python
filepath = "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_mist_pdr_overnight@-0.417mA_3.hdf5"
e_signals, pdrs, iters = load_data(filepath)
e_signals = e_signals.T
```

```python
ze.twotone.mist.MISTPowerDepOvernight().analyze(
    result=(iters, pdrs, e_signals),
    # g0=g0,
    # e0=e0,
    ac_coeff=ac_coeff,
)
```

# Power dep over flux

```python
filepath = r"C:\Users\QEL\Desktop\MeasureScriptX\QuantumMeasurementProcedures\Members\axel\Qubit-measure\Database\Q12_2D[4]\Q4\2025\11\Data_1114\Q4_mist_flux_bare@4.000mA_1.hdf5"
signals, As, pdrs = load_data(filepath)
```

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
fig = ze.twotone.flux_dep.MistExperiment().analyze(
    result=(As, pdrs, signals), mA_c=mA_c, period=period, ac_coeff=ac_coeff
)

from zcu_tools.notebook.analysis.mist.branch import plot_cn_with_mist

plot_cn_with_mist(
    fig,
    flxs=sim_flxs,
    photons=sim_photons,
    populations_over_flx=sim_populations,
    critical_levels={0: 2.0, 1: 3.0},
    mist_flxs=mA2flx(As, mA_c, period),
)

if isinstance(fig, plt.Figure):
    fig.savefig(f"../../../result/{qub_name}/image/mist_over_flux.png")
else:
    prefix = f"../../../result/{qub_name}/"
    # postfix = "branch/mist_over_flux_with_simulation.png"
    postfix = "branch_floquet/mist_over_flux_with_simulation.png"

    os.makedirs(os.path.dirname(os.path.join(prefix, "image", postfix)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(prefix, "web", postfix)), exist_ok=True)

    fig.update_layout(height=800)
    fig.write_image(os.path.join(prefix, "image", postfix))
    fig.write_html(os.path.join(prefix, "web", postfix.replace(".png", ".html")))
    fig.show()
```

```python

```
