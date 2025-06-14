---
jupyter:
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
import numpy as np
import pandas as pd

%autoreload 2
import zcu_tools.notebook.analysis.plot as zp
from zcu_tools.notebook.persistance import load_result
from zcu_tools.simulate import flx2mA, mA2flx
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flx
```

```python
qub_name = "Q12_2D[2]/Q4"

server_ip = "021-zcu216"
port = 4999
```

# Load data


## Parameters

```python
loadpath = f"../../result/{qub_name}/params.json"
_, params, mA_c, period, allows, _ = load_result(loadpath)
EJ, EC, EL = params

print(allows)

if "r_f" in allows:
    r_f = allows["r_f"]

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
loadpath = f"../../result/{qub_name}/freqs.csv"

freqs_df = pd.read_csv(loadpath)
s_mAs = freqs_df["Current (mA)"].values  # mA
s_fpts = freqs_df["Freq (MHz)"].values * 1e-3  # GHz
s_T1s = freqs_df["T1 (μs)"].values
s_T1errs = freqs_df["T1err (μs)"].values

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

# Flux dependence

```python
_, energies = calculate_energy_vs_flx(params, flxs, spectrum_data=spectrum_data, cutoff=50, evals_count=40)
```

```python
v_allows = {
    **allows,
    "transitions": [(i, j) for i in (0, 1, 2, 3) for j in range(i + 1, 40)],
    # "red side": [(i, j) for i in (0, 1, 2, 3) for j in range(i + 1, 15)],
    # "mirror": [(i, j) for i in (0, 1, 2) for j in range(i + 1, 15)],
    # "mirror red": [(i, j) for i in (0, 1, 2, 3) for j in range(i + 1, 15)],
    "r_f": 5.7965,
    "sample_f": 6.881280 / 2,
}

import plotly.graph_objects as go
from zcu_tools.notebook.analysis.fluxdep import energy2transition

fig = go.Figure()
freqs, names = energy2transition(energies, v_allows)
for j in range(1, 10):
    for i in range(len(names)):
        fig.add_trace(go.Scatter(x=mAs, y=freqs[:, i]/j, mode="lines", name=f"{j}_{names[i]}"))

fig.add_hline(y=v_allows["r_f"], line_dash="dash", line_color="black", line_width=2)
fig.add_hline(
    y=v_allows["sample_f"], line_dash="dash", line_color="black", line_width=2
)
fig.add_hline(
    y=2 * v_allows["sample_f"] - v_allows["r_f"],
    line_dash="dash",
    line_color="black",
    line_width=2,
)

fig.update_layout(
    title=f"EJ/EC/EL = ({params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f})",
    title_x=0.501,
)
fig.update_yaxes(range=[0.0, 8.0])
fig.update_layout(height=1000)
fig.show()
```

```python

```

```python

```
