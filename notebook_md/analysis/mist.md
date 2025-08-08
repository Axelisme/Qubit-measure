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
    display_name: Python 3
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
    version: 3.13.2
---

```python
%load_ext autoreload

%autoreload 2
from zcu_tools.utils.datasaver import load_data
import zcu_tools.experiment.v2 as ze
from zcu_tools.notebook.persistance import load_result
from zcu_tools.simulate import mA2flx
```

```python
loadpath = "../../result/Q12_2D[2]/Q4/params.json"
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
filepath = "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/R4_dispersive@-0.417mA_2.hdf5"
signals, fpts, _ = load_data(filepath)
signals = signals.T

fpts /= 1e6
```

```python
chi, kappa = ze.twotone.dispersive.DispersiveExperiment().analyze(
    result=(fpts, signals), asym=True
)
```

# AC stark shift

```python
filepath = "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_ac_stark@-0.417mA_2.hdf5"
signals, pdrs, fpts = load_data(filepath)
signals = signals.T
fpts /= 1e6
```

```python
import matplotlib.pyplot as plt
import numpy as np

plt.imshow(
    np.abs(signals.T),
    origin="lower",
    extent=[pdrs[0], pdrs[-1], fpts[0], fpts[-1]],
    aspect="auto",
)
plt.show()
```

```python
ac_coeff = ze.twotone.ac_stark.AcStarkExperiment().analyze(
    result=(pdrs, fpts, signals), chi=chi, kappa=kappa
)
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
filepath = (
    "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_mist_flx_pdr@-0.417mA_2.hdf5"
)
signals, As, pdrs = load_data(filepath)
```

```python
mA_c = 1.92
```

```python
data = np.load(r"../../result/Q12_2D[3]/Q4/branch_analysis_n0,0.npz")
sim_flxs, branchs, populations = data["flxs"], data["branchs"], data["populations"]
sim_flxs = np.concatenate([sim_flxs, 1 - sim_flxs[::-1]])
populations = np.concatenate([populations, populations[::-1, ...]], axis=0)


flxs = mA2flx(1e3 * As, mA_c, period)
ground_populations = np.array(
    [
        np.interp(np.mod(flxs, 1.0), sim_flxs, populations[:, 0, i])
        for i in range(populations.shape[2])
    ]
).T
excited_populations = np.array(
    [
        np.interp(np.mod(flxs, 1.0), sim_flxs, populations[:, 1, i])
        for i in range(populations.shape[2])
    ]
).T

# calculate the critical photon number
ground_cn = np.argmax(ground_populations >= 1.1, axis=1)
ground_cn[ground_cn == 0] = ground_populations.shape[1] - 1
excited_cn = np.argmax(excited_populations >= 2.0, axis=1)
excited_cn[excited_cn == 0] = excited_populations.shape[1] - 1
```

```python
fig, ax = ze.twotone.mist.MISTFluxPowerDep().analyze(
    result=(As, pdrs, signals),
    mA_c=mA_c,
    period=period,
    ac_coeff=ac_coeff,
)


ax.plot(flxs, ground_cn, label="ground", marker=".", color="r")
ax.plot(flxs, excited_cn, label="excited", marker=".", color="b")
# ax.set_xlim(0.4, 0.6)
fig.savefig("../../result/Q12_2D[3]/Q4/image/qubit_mist_over_flux_with_simulation.png")
```

```python

```
