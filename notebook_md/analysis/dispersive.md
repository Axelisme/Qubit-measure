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

%autoreload 2
from zcu_tools.datasaver import load_data
import zcu_tools.notebook.persistance as zp
import zcu_tools.notebook.analysis.dispersive as zd
from zcu_tools.simulate import mA2flx, flx2mA
from zcu_tools.simulate.fluxonium import calculate_dispersive_vs_flx
```

```python
qub_name = "Q12_2D[2]/Q4"
```

```python
loadpath = f"../../result/{qub_name}/params.json"
_, params, mA_c, period, allows, _ = zp.load_result(loadpath)

# mA_c = 4.46
# mA_c, _, period = (4.395142504148789, -0.3432768475307726, 9.476838703359125)

if "r_f" in allows:
    r_f = allows["r_f"]
    print(f"r_f = {r_f}")
```

```python
mA_c, period
```

# Plot with Onetone

```python
onetone_path = "../../Database/Q12_2D/Q4/2025/05/Data_0528/R4_flux_2.hdf5"

signals, sp_fpts, sp_mAs = load_data(
    onetone_path, server_ip="005-writeboard", port=4999
)
sp_mAs, sp_fpts, signals = zp.format_rawdata(sp_mAs, sp_fpts, signals)
signals = signals.T  # (sp_mAs, sp_fpts)

sp_flxs = mA2flx(sp_mAs, mA_c, period)
```

```python
r_f = 5.7945
best_g = 0.04
```

```python
best_g, best_rf = zd.auto_fit_dispersive(
    params,
    r_f,
    sp_flxs,
    sp_fpts,
    signals,
    g_bound=(0.02, 0.15),
    g_init=best_g,
    fit_rf=True,
)
if best_rf is not None:
    r_f = best_rf
best_g, r_f
```

```python
%matplotlib widget
finish_fn = zd.search_proper_g(
    params, r_f, sp_flxs, sp_fpts, signals, g_bound=(0.0, 0.2), g_init=best_g
)
```

```python
best_g = finish_fn()
best_g
```

```python
flxs = np.linspace(sp_flxs.min(), sp_flxs.max(), 501)
mAs = flx2mA(flxs, mA_c, period)
```

```python
rf_list = calculate_dispersive_vs_flx(params, flxs, r_f=r_f, g=best_g, return_dim=2)
fig = zd.plot_dispersive_with_onetone(
    r_f, best_g, mAs, flxs, rf_list, sp_mAs, sp_flxs, sp_fpts, signals
)
fig.show()
```

```python
fig.write_html(f"../../result/{qub_name}/web/dispersive.html", include_plotlyjs="cdn")
fig.write_image(
    f"../../result/{qub_name}/image/dispersive.png", format="png", width=800, height=400
)
```

# Write back g to result

```python
zp.update_result(loadpath, dict(dispersive=dict(g=best_g, r_f=r_f)))
```

```python

```
