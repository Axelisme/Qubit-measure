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
import numpy as np
import plotly.graph_objects as go

%autoreload 2
from zcu_tools.notebook.persistance import load_result
from zcu_tools.simulate import flx2mA, mA2flx  # noqa: F401
from zcu_tools.simulate.fluxonium import (
    calculate_system_n_oper_vs_flx,
    calculate_n_oper_vs_flx,
    calculate_energy_vs_flx,
)
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
_, params, mA_c, period, _, result = load_result(loadpath)

r_f = result["dispersive"]["r_f"]
g = result["dispersive"]["g"]
```

# Matrix elements

```python
sweep = None
spectrum_data = None

return_dim = 7
# flxs = np.linspace(0.0, 0.5, 200)
# mAs = flx2mA(flxs, mA_c, period)
mAs = np.linspace(-1.0, 3.0, 200)
flxs = mA2flx(mAs, mA_c, period)
```

```python
spectrum_data, energies = calculate_energy_vs_flx(
    params, flxs, spectrum_data=spectrum_data
)
```

```python
spectrum_data, elements = calculate_n_oper_vs_flx(
    params, flxs, return_dim=return_dim, spectrum_data=spectrum_data
)
```

```python
sweep, side_elements = calculate_system_n_oper_vs_flx(
    params, flxs, r_f, g=g, return_dim=return_dim, sweep=sweep
)
```

```python
bypass_levels = [None, 2, 3]
```

```python
fig = go.Figure()

for f, t in [(0, 1), (1, 0)]:
    for bypass_level in bypass_levels:
        if bypass_level is None:
            m = np.abs(side_elements[:, f, t])
            label = f"{f}->{t}"
        else:
            m1 = np.abs(elements[:, f, bypass_level])
            m2 = np.abs(side_elements[:, bypass_level, t])
            # m = np.sqrt(m1 * m2)
            m = np.minimum(m1, m2)
            label = f"{f}->{bypass_level}->{t}"
        fig.add_trace(go.Scatter(x=mAs, y=m, mode="lines", name=label))

fig.update_layout(
    yaxis_range=[0, 0.2],
    xaxis_title="mA",
    yaxis_title="Matrix Elements",
    showlegend=True,
)
fig.show()
```

```python
prefix = f"../../result/{qub_name}"
fig.write_html(f"{prefix}/web/reset_element.html", include_plotlyjs="cdn")
fig.write_image(
    f"{prefix}/image/reset_element.png", format="png", width=800, height=400
)
```

```python
fig = go.Figure()

for f, t in [(0, 1), (1, 0)]:
    for bypass_level in bypass_levels:
        if bypass_level is None:
            energy = energies[:, t] - energies[:, f]
            fig.add_trace(
                go.Scatter(
                    x=mAs, y=np.abs(r_f + energy), mode="lines", name=f"{f}->{t} red"
                )
            )
        else:
            energy1 = energies[:, bypass_level] - energies[:, f]
            energy2 = energies[:, t] - energies[:, bypass_level]
            fig.add_trace(
                go.Scatter(
                    x=mAs, y=np.abs(energy1), mode="lines", name=f"{f}->{bypass_level}"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=mAs,
                    y=np.abs(r_f + energy2),
                    mode="lines",
                    name=f"{bypass_level}->{t} red",
                )
            )

fig.update_layout(
    yaxis_range=[0, 7.0],
    xaxis_title="Flux",
    yaxis_title="Energy",
    showlegend=True,
)
fig.show()
```

```python
prefix = f"../../result/{qub_name}"
fig.write_html(f"{prefix}/web/reset_energy.html", include_plotlyjs="cdn")
```

```python

```
