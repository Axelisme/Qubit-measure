---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: zcu-tools
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
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d

%autoreload 2
import zcu_tools.notebook.persistance as zp
import zcu_tools.notebook.analysis.dispersive as zd
from zcu_tools.simulate import value2flux
from zcu_tools.simulate.fluxonium import calculate_dispersive_vs_flux
from zcu_tools.notebook.analysis.plot import plot_dispersive_shift
from zcu_tools.utils.fitting.resonance import (
    fit_edelay,
    remove_edelay,
    fit_circle_params,
    calc_phase,
)
```

```python
chip_name = "Q12_2D[5]"
qub_name = "Q1"

result_dir = Path(f"../../result/{chip_name}/{qub_name}")
param_path = result_dir / "params.json"

image_dir = result_dir / "image" / "dispersive"
web_dir = result_dir / "web" / "dispersive"
image_dir.mkdir(exist_ok=True)
web_dir.mkdir(exist_ok=True)
```

```python
result_dict = zp.load_result(str(param_path))
fluxdepfit_dict = result_dict.get("fluxdep_fit")
assert fluxdepfit_dict is not None, "fluxdep_fit not found in result_dict"

params = (
    fluxdepfit_dict["params"]["EJ"],
    fluxdepfit_dict["params"]["EC"],
    fluxdepfit_dict["params"]["EL"],
)
flux_half = fluxdepfit_dict["flux_half"]
flux_int = fluxdepfit_dict["flux_int"]
flux_period = fluxdepfit_dict["flux_period"]

print("params = ", params, " GHz")
print("flux_half = ", flux_half)
print("flux_int = ", flux_int)
print("flux_period = ", flux_period)

if dispersive_dict := result_dict.get("dispersive"):
    bare_rf = dispersive_dict["bare_rf"]
elif "r_f" in fluxdepfit_dict["plot_transitions"]:
    bare_rf = fluxdepfit_dict["plot_transitions"]["r_f"]
else:
    bare_rf = 5.0  # GHz
print(f"bare rf = {bare_rf}", "GHz")
```

# Plot with Onetone

```python
from zcu_tools.experiment.v2.onetone import FluxDepExp

onetone_path = r"../../Database/Q12_2D[5]/Q1/R1_flux_1.hdf5"

sp_dev_values, sp_freqs, sp_signals = FluxDepExp().load(onetone_path)
sp_freqs *= 1e-3  # MHz to GHz
sp_fluxs = value2flux(sp_dev_values, flux_half, flux_period)
```

```python
edelays = Parallel(n_jobs=-1)(
    delayed(fit_edelay)(sp_freqs, sp_signal)
    for sp_signal in tqdm(sp_signals, desc="Fitting edelay")
)
edelays = np.asarray(edelays)
edelay = np.median(edelays).item()

rot_signals = remove_edelay(sp_freqs, sp_signals, edelay)
rot_signals = gaussian_filter1d(rot_signals, len(sp_freqs) // 30, axis=1)
rot_signals = np.asarray(rot_signals, dtype=np.complex128)

circle_param = np.median(
    [fit_circle_params(rot_signal.real, rot_signal.imag) for rot_signal in rot_signals],
    axis=0,
)

phases = calc_phase(rot_signals, circle_param[0], circle_param[1], axis=1)
```

```python
%matplotlib inline
norm_phases = phases
norm_phases = gaussian_filter1d(norm_phases, phases.shape[1] // 10, axis=1)
norm_phases = np.diff(norm_phases, axis=1, prepend=norm_phases[:, :1])
norm_phases = np.abs(norm_phases)
norm_phases /= np.max(norm_phases, axis=1, keepdims=True)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
ax1.set_title("Signal Magnitude")
ax1.imshow(
    np.abs(sp_signals).T,
    aspect="auto",
    extent=[sp_fluxs[0], sp_fluxs[-1], sp_freqs[0], sp_freqs[-1]],
    interpolation="none",
)
ax1.axvline(1.0, color="b", linestyle="--")
ax1.axvline(0.5, color="r", linestyle="--")
ax2.set_title("Fitted edelay")
ax2.plot(sp_fluxs, edelays)
ax2.axhline(edelay, color="k", linestyle="--")
ax2.axvline(1.0, color="b", linestyle="--")
ax2.axvline(0.5, color="r", linestyle="--")
ax2.set_xlim(sp_fluxs[0], sp_fluxs[-1])
ax2.grid()
ax3.set_title("Normalized Phases")
ax3.imshow(
    norm_phases.T,
    aspect="auto",
    extent=[sp_fluxs[0], sp_fluxs[-1], sp_freqs[0], sp_freqs[-1]],
    interpolation="none",
)
ax3.axvline(1.0, color="b", linestyle="--")
ax3.axvline(0.5, color="r", linestyle="--")
plt.show()
plt.close(fig)
```

```python
best_g = 0.05  # GHz
```

```python
%matplotlib widget
finish_fn = zd.search_proper_g(
    params, bare_rf, sp_fluxs, sp_freqs, norm_phases, g_bound=(0.0, 0.2), g_init=best_g
)
```

```python
best_g, bare_rf = finish_fn()
best_g, bare_rf
```

```python
best_g, best_rf = zd.auto_fit_dispersive(
    params,
    bare_rf,
    sp_fluxs,
    sp_freqs,
    norm_phases,
    g_bound=(0.01, 0.15),
    g_init=best_g,
    fit_bare_rf=True,
)
if best_rf is not None:
    bare_rf = best_rf
best_g, bare_rf
```

```python
best_g = 0.06834219285137985
bare_rf = 5.349831026722171
```

```python
t_fluxs = np.linspace(sp_fluxs.min(), sp_fluxs.max(), 501)
```

```python
plot_rfs = calculate_dispersive_vs_flux(params, t_fluxs, bare_rf, best_g, return_dim=3)
```

```python
fig = zd.plot_dispersive_with_onetone(
    bare_rf, best_g, t_fluxs, plot_rfs, sp_fluxs, sp_freqs, norm_phases
)
fig.show()
```

```python
figname = "dispersive"
fig.write_html(f"{web_dir}/{figname}.html", include_plotlyjs="cdn")
fig.write_image(f"{image_dir}/{figname}.png", format="png", width=800, height=400)
```

```python
fig = plot_dispersive_shift(params, t_fluxs, bare_rf, best_g, upto=5)
fig.show()
```

```python
figname = "dispersive_pdr_dep"
fig.write_html(f"{web_dir}/{figname}.html", include_plotlyjs="cdn")
fig.write_image(f"{image_dir}/{figname}.png", format="png", width=800, height=400)
```

# Write back g to result

```python
zp.update_result(str(param_path), dict(dispersive=dict(g=best_g, bare_rf=bare_rf)))
```
