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
from zcu_tools.datasaver import load_data

import zcu_tools.notebook.single_qubit as zf
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
zf.mux_reset_fpt_analyze(fpts1, fpts2, signals, xname=xname, yname=yname)
```

## reset gain

```python
filepath = (
    "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_mux_reset_gain@-0.417mA_3.hdf5"
)
signals, pdrs1, pdrs2 = load_data(filepath)
```

```python
zf.mux_reset_pdr_analyze(pdrs1, pdrs2, signals, xname=xname, yname=yname)
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
zf.mux_reset_time_analyze(Ts, signals)
```

# Dispersive shift

```python
filepath = "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/R4_dispersive@-0.417mA_2.hdf5"
signals, fpts, _ = load_data(filepath)
signals = signals.T

fpts /= 1e6
```

```python
chi, kappa = zf.analyze_dispersive(fpts, signals, asym=True)
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
ac_coeff = zf.analyze_ac_stark_shift(pdrs, fpts, signals, chi=chi, kappa=kappa)
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
zf.analyze_mist_pdr_dep(
    pdrs,
    signals[0, :],
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
g0 = zf.analyze_mist_pdr_overnight(
    pdrs,
    g_signals,
    # pi_signal=e0,
    ac_coeff=ac_coeff,
)
```

```python
filepath = "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_mist_pdr_overnight@-0.417mA_3.hdf5"
e_signals, pdrs, iters = load_data(filepath)
e_signals = e_signals.T
```

```python
e0 = zf.analyze_mist_pdr_overnight(pdrs, e_signals, pi_signal=g0, ac_coeff=ac_coeff)
```

# Abnormal

```python
filepath = (
    # "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_abnormal_pdr@-0.417mA_1.hdf5"
    # "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_abnormal_pdr@-0.417mA_2.hdf5"
    "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_abnormal_pdr_mux_reset@-0.417mA_1.hdf5"
    # "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_abnormal_pdr_mux_reset@-0.417mA_2.hdf5"
)
signals, pdrs, _ = load_data(filepath)
signals = signals.T
```

```python
zf.analyze_abnormal_pdr_dep(
    pdrs,
    signals,
    g0=g0,
    e0=e0,
    ac_coeff=ac_coeff,
)
```

# Power dep over flux

```python
filepath = (
    "../../Database/Q12_2D[2]/Q4/2025/06/Data_0609/Q4_mist_flx_pdr@-0.417mA_2.hdf5"
)
signals, mAs, pdrs = load_data(filepath)

mAs *= 1e3
flxs = mA2flx(mAs, mA_c, period)
```

```python
fig, ax = zf.analyze_mist_flx_pdr(
    flxs,
    pdrs,
    signals,
    # ac_coeff=ac_coeff,
)
# ax.set_xlim(0.4, 0.6)
plt.show()
```

```python

```
