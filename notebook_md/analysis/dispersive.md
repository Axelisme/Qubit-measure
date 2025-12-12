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

import numpy as np

%autoreload 2
from zcu_tools.experiment.v2.onetone import FluxDepExperiment
import zcu_tools.notebook.persistance as zp
import zcu_tools.notebook.analysis.dispersive as zd
from zcu_tools.simulate import mA2flx, flx2mA
from zcu_tools.simulate.fluxonium import calculate_dispersive_vs_flx
from zcu_tools.notebook.analysis.plot import plot_dispersive_shift
from zcu_tools.notebook.analysis.fluxdep import InteractiveLines
```

```python
chip_name = "Q12_2D[5]"
qub_name = "Q1"

result_dir = Path(f"../../result/{chip_name}/{qub_name}")
param_path = result_dir / "params.json"
```

```python
_, params, mA_c, period, allows, _ = zp.load_result(str(param_path))

# mA_c = 4.46
# mA_c, _, period = (4.395142504148789, -0.3432768475307726, 9.476838703359125)

mA_e = mA_c + period / 2

print("mA_c = ", mA_c)
print("period = ", period)

if "r_f" in allows:
    r_f: float = allows["r_f"]
    print(f"r_f = {r_f}")
```

# Plot with Onetone

```python
onetone_path = r"../../Database/Q12_2D[5]/Q1/s002_onetone_flux_Q0_015.hdf5"


exp = FluxDepExperiment()
sp_As, sp_fpts, signals = exp.load(onetone_path)
sp_fpts *= 1e-3  # MHz to GHz

sp_flxs = mA2flx(sp_As, mA_c, period)
```

```python
%matplotlib widget
actLine = InteractiveLines(signals, sp_As, sp_fpts, mA_c=mA_c, mA_e=mA_e)
```

```python
mA_c, mA_e = actLine.get_positions()
period = 2 * abs(mA_e - mA_c)

mA_c, mA_e, period
```

```python
from zcu_tools.utils.fitting.resonance import (
    fit_edelay,
    remove_edelay,
    fit_circle_params,
    calc_phase,
)
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d

edelays = Parallel(n_jobs=-1)(
    delayed(fit_edelay)(sp_fpts, signal) for signal in tqdm(signals)
)
edelays = np.asarray(edelays)
edelay = np.median(edelays).item()

rot_signals = remove_edelay(sp_fpts, signals, edelay)
rot_signals = gaussian_filter1d(rot_signals, len(sp_fpts) // 30, axis=1)

circle_param = np.median(
    [fit_circle_params(rot_signal.real, rot_signal.imag) for rot_signal in rot_signals],
    axis=0,
)

phases = calc_phase(rot_signals, circle_param[0], circle_param[1], axis=1)

```

```python
%matplotlib inline
norm_phases = (phases - np.min(phases, axis=1, keepdims=True)) / np.ptp(
    phases, axis=1, keepdims=True
)

norm_phases = np.diff(norm_phases, axis=1, prepend=norm_phases[0, 0])
norm_phases = np.clip(norm_phases, -5 * np.std(norm_phases), 5 * np.std(norm_phases))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
ax1.imshow(
    np.abs(signals.T),
    aspect="auto",
    extent=[sp_flxs.min(), sp_flxs.max(), sp_fpts.min(), sp_fpts.max()],
)
ax2.plot(sp_flxs, edelays)
ax2.axhline(edelay, color="k", linestyle="--")
ax2.grid()
ax3.imshow(
    np.abs(norm_phases).T,
    aspect="auto",
    extent=[sp_flxs.min(), sp_flxs.max(), sp_fpts.min(), sp_fpts.max()],
)
plt.show()
plt.close(fig)
```

```python
# r_f = 5.345
best_g = 0.1
```

```python
%matplotlib widget
finish_fn = zd.search_proper_g(
    params, r_f, sp_flxs, sp_fpts, norm_phases, g_bound=(0.0, 0.2), g_init=best_g
)
```

```python
best_g, r_f = finish_fn()
best_g, r_f
```

```python
best_g, best_rf = zd.auto_fit_dispersive(
    params,
    r_f,
    sp_flxs,
    sp_fpts,
    norm_phases,
    g_bound=(0.02, 0.15),
    g_init=best_g,
    fit_rf=True,
)
if best_rf is not None:
    r_f = best_rf
best_g, r_f
```

```python
flxs = np.linspace(sp_flxs.min(), sp_flxs.max(), 501)
mAs = flx2mA(flxs, mA_c, period)
```

```python
rf_list = calculate_dispersive_vs_flx(params, flxs, r_f=r_f, g=best_g, return_dim=3)
fig = zd.plot_dispersive_with_onetone(
    r_f, best_g, mAs, flxs, rf_list, sp_As, sp_flxs, sp_fpts, norm_phases
)
fig.show()
```

```python
figname = "dispersive"
fig.write_html(str(result_dir / "web" / f"{figname}.html"), include_plotlyjs="cdn")
fig.write_image(
    str(result_dir / "image" / f"{figname}.png"), format="png", width=800, height=400
)
```

```python
fig = plot_dispersive_shift(params, flxs, r_f=r_f, g=best_g, upto=5)
fig.show()
```

```python
figname = "dispersive_pdr_dep"
fig.write_html(str(result_dir / "web" / f"{figname}.html"), include_plotlyjs="cdn")
fig.write_image(
    str(result_dir / "image" / f"{figname}.png"), format="png", width=800, height=400
)
```

# Write back g to result

```python
zp.update_result(str(param_path), dict(dispersive=dict(g=best_g, r_f=r_f)))
```

```python

```
