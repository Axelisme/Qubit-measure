---
jupyter:
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scqubits as scq

%autoreload 2
from zcu_tools.notebook.persistance import load_result
from zcu_tools.notebook.analysis.t1_curve import (
    plot_t1_vs_flx,
    get_eff_t1,
    plot_sample_t1,
)
from zcu_tools.simulate import flx2mA, mA2flx
```

```python
qub_name = "Test049"
```

# Load data


## Parameters

```python
loadpath = f"../../result/{qub_name}/params.json"
_, params, mA_c, period, allows, _ = load_result(loadpath)
EJ, EC, EL = params

# mA_c = 4.46

print(allows)

if "r_f" in allows:
    r_f = allows["r_f"]

if "sample_f" in allows:
    sample_f = allows["sample_f"]


flxs = np.linspace(0.0, 1.5, 1000)
mAs = flx2mA(flxs, mA_c, period)
```

## Load Sample Points

```python
# loading points
loadpath = f"../../result/{qub_name}/sample.csv"

freqs_df = pd.read_csv(loadpath)
freqs_df = freqs_df[~np.isnan(freqs_df["T1 (us)"])]
s_mAs = freqs_df["calibrated mA"].values  # mA
s_fpts = freqs_df["Freq (MHz)"].values * 1e-3  # GHz
s_T1s = freqs_df["T1 (us)"].values
s_T1errs = freqs_df["T1err (us)"].values

# sort by flux
s_mAs, s_fpts, s_T1s, s_T1errs = tuple(
    np.array(a) for a in zip(*sorted(zip(s_mAs, s_fpts, s_T1s, s_T1errs)))
)
s_flxs = mA2flx(s_mAs, mA_c, period)

freqs_df.head(10)
```

# Simulation

```python
fluxonium = scq.Fluxonium(*params, flux=0.5, cutoff=40, truncated_dim=6)
spectrumData = fluxonium.get_matelements_vs_paramvals(
    operator="n_operator", param_name="flux", param_vals=flxs, evals_count=40
)
evals, evecs = spectrumData.energy_table, spectrumData.state_table
```

# T1 curve

```python
fig, _ = plot_sample_t1(s_mAs, s_T1s, s_T1errs, mA_c, period)
fig.savefig(f"../../result/{qub_name}/image/T1s.png")
plt.show()
```

```python
Temp = 113e-3
# Temp = 200e-3
```

```python
plot_args = (s_mAs, s_flxs, s_T1s, s_T1errs, mA_c, period, fluxonium)
plot_kwargs = dict(Temp=Temp, t_mAs=mAs, t_flxs=flxs, esys=(evals, evecs))
```

```python
Q_cap = 7e4

fig, _ = plot_t1_vs_flx(
    *plot_args,
    name="Q_cap",
    noise_name="t1_capacitive",
    values=[Q_cap / 2, Q_cap, Q_cap * 2],
    **plot_kwargs,
)

fig.savefig(f"../../result/{qub_name}/image/T1s_fit_Qcap.png")
plt.show()
```

```python
x_qp = 1.0e-5

fig, _ = plot_t1_vs_flx(
    *plot_args,
    name="x_qp",
    noise_name="t1_quasiparticle_tunneling",
    values=[x_qp / 2, x_qp, x_qp * 2],
    **plot_kwargs,
)


fig.savefig(f"../../result/{qub_name}/image/T1s_fit_xqp.png")
plt.show()
```

```python
Q_ind = 1e6

fig, ax = plot_t1_vs_flx(
    *plot_args,
    name="Q_ind",
    noise_name="t1_inductive",
    values=[Q_ind / 2, Q_ind, Q_ind * 2],
    **plot_kwargs,
)
# ax.set_xlim(-5, -4)

fig.savefig(f"../../result/{qub_name}/image/T1s_fit_Q_ind.png")
plt.show()
```

```python
fig, ax = fluxonium.plot_t1_effective_vs_paramvals(
    param_name="flux",
    param_vals=flxs,
    noise_channels=[
        ("t1_capacitive", dict(Q_cap=Q_cap)),
        # ("t1_quasiparticle_tunneling", dict(x_qp=x_qp)),
        # ("t1_inductive", dict(Q_ind=Q_ind)),
    ],
    common_noise_options=dict(i=1, j=0, T=Temp),
    spectrum_data=spectrumData,
)
ax.set_xlim(s_flxs.min() - 0.1, s_flxs.max() + 0.1)
ax.plot(s_flxs, 1e3 * s_T1s, ".-", label="T1 data")
plt.show(fig)
```

```python
1e-3 * get_eff_t1(
    0.5, fluxonium, noise_channels=[("t1_capacitive", {"Q_cap": Q_cap})], Temp=Temp
)
```

```python

```

```python

```
