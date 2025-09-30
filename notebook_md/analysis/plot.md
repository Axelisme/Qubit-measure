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
import numpy as np
import pandas as pd

%autoreload 2
import zcu_tools.notebook.analysis.plot as zp
from zcu_tools.notebook.persistance import load_result
from zcu_tools.simulate import flx2mA, mA2flx
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flx
```

```python
qub_name = "Q3_2D/Q2"

os.makedirs(f"../../result/{qub_name}/image/design", exist_ok=True)
os.makedirs(f"../../result/{qub_name}/web/design", exist_ok=True)
```

## Parameters

```python
loadpath = f"../../result/{qub_name}/params.json"
_, params, mA_c, period, allows, data_dict = load_result(loadpath)
EJ, EC, EL = params

print(allows)

if dispersive_cfg := data_dict.get("dispersive"):
    g = dispersive_cfg["g"]
    r_f = dispersive_cfg["r_f"]
    print(f"g: {g}, r_f: {r_f}")
elif "r_f" in allows:
    r_f = allows["r_f"]
    g = 1e-3
    print(f"r_f: {r_f}")

if "sample_f" in allows:
    sample_f = allows["sample_f"]


flxs = np.linspace(0.0, 1.0, 1000)
mAs = flx2mA(flxs, mA_c, period)
```

```python
spectrum_data = None
```

## Load Sample Points

```python
# loading points
loadpath = f"../../result/{qub_name}/samples.csv"

freqs_df = pd.read_csv(loadpath)
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

# Matrix elements

```python
show_idxs = [(i, j) for i in range(2) for j in range(5) if j > i]

fig, _ = zp.plot_matrix_elements(params, flxs, show_idxs, spectrum_data=spectrum_data)
fig.show()
```

```python
fig.write_html(f"../../result/{qub_name}/web/matrixelem.html", include_plotlyjs="cdn")
fig.write_image(f"../../result/{qub_name}/image/matrixelem.png", format="png")
```

# Dispersive

```python
fig = zp.plot_dispersive_shift(params, flxs, r_f=r_f, g=g, upto=3)
```

```python
fig.update_yaxes(range=[-3, 3])
fig.show()
```

# Flux dependence

```python
_, energies = calculate_energy_vs_flx(params, flxs, spectrum_data=spectrum_data)
```

```python
v_allows = {
    # **allows,
    "transitions": [(i, j) for i in (0, 1) for j in range(i + 1, 15)],
    # "transitions2": [(i, j) for i in (0, 1) for j in range(i + 1, 15)],
    "red side": [(i, j) for i in (0, 1) for j in range(i + 1, 4)],
    # "blue side": [(i, j) for i in (0, 1) for j in range(i + 1, 4)],
    # "red side": [(i, j) for i in (0, 1) for j in range(i + 1, 15)],
    # "mirror": [(i, j) for i in (0, 1, 2) for j in range(i + 1, 15)],
    # "mirror red": [(i, j) for i in (0, 1, 2, 3) for j in range(i + 1, 15)],
    "r_f": 7.520,
    "sample_f": 9.58464 / 2,
    # "r_f": r_f,
    # "sample_f": sample_f,
}

import plotly.graph_objects as go
from zcu_tools.notebook.analysis.fluxdep import energy2transition

fig = go.Figure()
freqs, names = energy2transition(energies, v_allows)
for i in range(len(names)):
    fig.add_trace(go.Scatter(x=mAs, y=freqs[:, i], mode="lines", name=f"{names[i]}"))
    # fig.add_trace(go.Scatter(x=flxs, y=freqs[:, i], mode="lines", name=f"{names[i]}"))
# for j in range(1, 10):
#     for i in range(len(names)):
#         fig.add_trace(
#             go.Scatter(x=mAs, y=freqs[:, i] / j, mode="lines", name=f"{j}_{names[i]}")
#         )

fig.add_hline(y=v_allows["r_f"], line_dash="dash", line_color="black", line_width=2)
fig.add_hline(
    y=v_allows["sample_f"], line_dash="dash", line_color="black", line_width=2
)
# fig.add_hline(
#     y=2 * v_allows["sample_f"] - v_allows["r_f"],
#     line_dash="dash",
#     line_color="black",
#     line_width=2,
# )

fig.update_layout(
    title=f"EJ/EC/EL = ({params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f})",
    title_x=0.501,
)
fig.update_yaxes(range=[0.0, 8.0])
fig.update_layout(height=1000)
fig.show()
```

# T1

```python
Temp = 120e-3
Q_cap = 7.0e5
# Q_ind = 1.0e7
# x_qp = 1.0e-8

noise_channels=[
    ("t1_capacitive", dict(Q_cap=Q_cap)),
    # ("t1_inductive", dict(Q_ind=Q_ind)),
    # ("t1_quasiparticle_tunneling", dict(x_qp=x_qp)),
]

fig, t1s = zp.plot_t1s(
    params,
    flxs,
    noise_channels=noise_channels,
    Temp=Temp,
)
title1 = f"EJ/EC/EL = {params[0]:.3f}/{params[1]:.3f}/{params[2]:.3f}"
title2 = ", ".join(
    ", ".join(f"{name} = {value:.1e}" for name, value in p_dict.items())
    for _, p_dict in noise_channels
)
fig.update_layout(
    title=title1 + "<br>" + title2,
    title_x=0.515,
    yaxis_range=[np.log10(0.5 * t1s.min()), np.log10(1.5 * t1s.max())],
)
fig.show()
```

```python

```
