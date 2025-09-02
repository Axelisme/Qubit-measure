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
import os
from pprint import pprint
import numpy as np

%autoreload 2
from zcu_tools.utils.datasaver import load_data

from zcu_tools.simulate.fluxonium import calculate_energy_vs_flx

import zcu_tools.notebook.analysis.fluxdep as zf
import zcu_tools.notebook.persistance as zp
from zcu_tools.simulate import mA2flx
```

```python
qub_name = "Q3_2D/Q1"

server_ip = "021-zcu216"
port = 4999

mA_c = None
mA_e = None
period = None
s_spects = {}

os.makedirs(f"../../result/{qub_name}/image", exist_ok=True)
os.makedirs(f"../../result/{qub_name}/web", exist_ok=True)
```

```python
loadpath = f"../../result/{qub_name}/params.json"
_, sp_params, mA_c, period, allows, _ = zp.load_result(loadpath)

mA_e = mA_c + 0.5 * period
pprint(allows)
```

# Load Spectrum

```python
spect_path = r"../../Database/Q3_2D/Q1/003_qubit_flux_spec_ge_Q1_1.hdf5"
# spect_path = r"../../Database/SF010/3D5,9G_flux_2.hdf5"
spectrum, _fpts, _As = load_data(spect_path, server_ip=server_ip, port=port)
mAs, fpts, spectrum = zp.format_rawdata(_As, _fpts, spectrum)
```

```python
%matplotlib widget
actLine = zf.InteractiveLines(spectrum, mAs, fpts, mA_c, mA_e)
```

```python
mA_c, mA_e = actLine.get_positions()
period = 2 * abs(mA_e - mA_c)

mA_c, mA_e, period
```

```python
%matplotlib widget
# actSel = zf.InteractiveOneTone(mAs, fpts, spectrum, threshold=0.5)
actSel = zf.InteractiveFindPoints(spectrum, mAs, fpts, threshold=6.0)
```

```python
ss_mAs, ss_fpts = actSel.get_positions()
```

```python
name = os.path.basename(spect_path)
s_spects.update(
    {
        name: {
            "mA_c": mA_c,
            "period": period,
            "spectrum": {
                "mAs": mAs,
                "fpts": fpts,
                "data": spectrum,
            },
            "points": {
                "mAs": ss_mAs,
                "fpts": ss_fpts,
            },
        }
    }
)
s_spects.keys()
```

# Save & Load

```python
processed_spect_path = f"../../result/{qub_name}/data/fluxdep/spectrums.hdf5"
zp.dump_spects(processed_spect_path, s_spects, mode="x")
```

```python
processed_spect_path = f"../../result/{qub_name}/data/fluxdep/spectrums.hdf5"
s_spects = zp.load_spects(processed_spect_path)
s_spects.keys()
```

```python
# del s_spects["s002_onetone_flux_Q2_4.hdf5"]
```

# Align half flux

```python
# for val in s_spects.values():  # swap mA_c and mA_e
#     val["mA_c"] = val["mA_c"] + 0.5 * period
```

```python
mA_c = list(s_spects.values())[-1]["mA_c"]
period = list(s_spects.values())[-1]["period"]
for spect in s_spects.values():
    shift = mA_c - spect["mA_c"]
    spect["mA_c"] += shift
    spect["spectrum"]["mAs"] += shift
    spect["points"]["mAs"] += shift
```

```python
mA_bound = (
    np.nanmin([np.nanmin(s["spectrum"]["mAs"]) for s in s_spects.values()]),
    np.nanmax([np.nanmax(s["spectrum"]["mAs"]) for s in s_spects.values()]),
)
fpt_bound = (
    np.nanmin([np.nanmin(s["points"]["fpts"]) for s in s_spects.values()]),
    np.nanmax([np.nanmax(s["points"]["fpts"]) for s in s_spects.values()]),
)
s_selected = None
t_mAs = np.linspace(mA_bound[0], mA_bound[1], 1000)
t_fpts = np.linspace(fpt_bound[0], fpt_bound[1], 1000)
t_flxs = mA2flx(t_mAs, mA_c, period)
```

# Manual Remove Points

```python
%matplotlib widget
intSel = zf.InteractiveSelector(s_spects, selected=s_selected)
```

```python
s_mAs, s_fpts, s_selected = intSel.get_positions()
s_flxs = mA2flx(s_mAs, mA_c, period)
```

# Fitting range

```python
# general
EJb = (2.0, 15.0)
ECb = (0.2, 2.0)
ELb = (0.1, 2.0)
# interger
# EJb = (3.0, 6.0)
# ECb = (0.8, 2.0)
# ELb = (0.08, 0.2)
# all
# EJb = (1.0, 20.0)
# ECb = (0.1, 4.0)
# ELb = (0.01, 3.0)
# custom
# EJb = (3.0, 8.0)
# ECb = (0.1, 1.0)
# ELb = (0.05, 1.0)
```

# Search in Database

```python
allows = {
    "transitions": [(0, 1), (0, 2), (0, 3), (1, 2)],
    # "transitions": [(0, 1), (0, 2)],
    # "red side": [(0, 1), (0, 2), (1, 2), (0, 3)],
    "mirror": [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3)],
    "mirror2": [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3)],
    # "r_f": 7.527,
    "sample_f": 9.584640 / 2,
    # "sample_f": 6.881280 / 2,
}
allows = {
    **allows,
    # "transitions": [(i, j) for i in (0, 1) for j in range(10) if i < j],
    # "red side": [(i, j) for i in (0, 1) for j in range(10) if i < j],
    # "blue side": [(i, j) for i in (0, 1, 2) for j in range(8) if i < j],
    # "mirror": [(i, j) for i in (0, 1) for j in range(10) if i < j],
    # "transitions2": [(i, j) for i in (0, 1, 2) for j in range(11) if i < j],
    # "mirror2": [(i, j) for i in (0, 1, 2) for j in range(8) if i < j],
}
```

```python
best_params, fig = zf.search_in_database(
    s_flxs, s_fpts, "../../Database/simulation/fluxonium_1.h5", allows, EJb, ECb, ELb
)
fig.savefig(f"../../result/{qub_name}/image/search_result.png")
```

```python
_, energies = calculate_energy_vs_flx(best_params, t_flxs, cutoff=40, evals_count=15)
```

```python
v_allows = {
    **allows,
    # "transitions": [(0, 1), (0, 2), (1, 2), (0, 3)],
    # "red side": [(0, 1), (0, 2), (1, 2), (0, 3)],
    # "mirror": [(0, 1), (0, 2), (1, 2), (0, 3)],
    # "transitions": [(i, j) for i in (0, 1, 2) for j in range(i + 1, 15)],
    # "red side": [(i, j) for i in [0, 1, 2] for j in range(i + 1, 15)],
    # "mirror": [(i, j) for i in (0, 1) for j in range(i + 1, 15)],
    # "mirror red": [(i, j) for i in (0, 1) for j in range(i + 1, 15)],
    "transitions2": [(0, 1), (0, 2), (1, 2), (0, 3)],
}

vs = zf.VisualizeSpet(
    s_spects, s_mAs, s_fpts, t_mAs, energies, v_allows, auto_hide=False
)
fig = vs.create_figure()
_ = fig.update_layout(
    title=f"EJ/EC/EL = ({best_params[0]:.2f}, {best_params[1]:.2f}, {best_params[2]:.2f})",
    title_x=0.501,
)
# fig.update_yaxes(range=[allows["r_f"] - 0.01, allows["r_f"] + 0.01])
fig.update_layout(height=1000)
fig.show()
```

# Scipy Optimization

```python
# fit the spectrumData
sp_params = zf.fit_spectrum(s_flxs, s_fpts, best_params, allows, (EJb, ECb, ELb))

# print the results
print("Fitted params:", *sp_params)
```

```python
_, energies = calculate_energy_vs_flx(sp_params, t_flxs, cutoff=40, evals_count=15)
```

```python
v_allows = {
    **allows,
    # "transitions": [(i, j) for i in (0, 1) for j in range(i + 1, 10)],
    # "red side": [(i, j) for i in (0, 1) for j in range(i + 1, 15)],
}

vs = zf.VisualizeSpet(
    s_spects, s_mAs, s_fpts, t_mAs, energies, v_allows, auto_hide=True
)
fig = vs.create_figure()
fig.update_layout(
    title=f"EJ/EC/EL = ({sp_params[0]:.3f}, {sp_params[1]:.3f}, {sp_params[2]:.3f})",
    title_x=0.501,
)
# fig.update_yaxes(range=[allows["r_f"] - 0.01, allows["r_f"] + 0.01])
# fig.update_layout(height=1000)
fig.show()
```

```python
fig.write_html(f"../../result/{qub_name}/web/spect_fit.html", include_plotlyjs="cdn")
fig.write_image(f"../../result/{qub_name}/image/spect_fit.png", format="png")
```

# Save Parameters

```python
# dump the data
savepath = f"../../result/{qub_name}/params.json"

zp.dump_result(savepath, qub_name, sp_params, mA_c, period, allows)
```

```python
savepath = f"../../result/{qub_name}/data/fluxdep/selected_points.npz"

np.savez(savepath, flxs=s_flxs, fpts=s_fpts, selected=s_selected)
```

```python

```
