---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
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
    version: 3.13.11
---

**Readers:** operator, lab-author. Fit resonator responses from an entry-local predictor and export a dispersive artifact.

Set the entry name, component name and bare_rf in GHz below. The component must
have kind `qubit/fluxonium` in the entry setup. Predictor input comes from
`predictors/<component>.json`. This notebook does not choose a component from aliases
or seed bare_rf from saved artifacts. It does not read params.json.
The [dispersive GUI](../../lib/zcu_tools/gui/app/dispersive/README.md) owns the interactive workflow;
its MCP interface remains read-only.

```python
%load_ext autoreload
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d

%autoreload 2
import zcu_tools.notebook.analysis.dispersive as zd
from zcu_lab.components import register_all
from zcu_tools.resources.entry import ResultEntry, component_registry, snapshot
from zcu_tools.resources.predictor_file import load_predictor_file
from zcu_tools.resources.analysis_source import AnalysisSource
from zcu_tools.resources.dispersive_file import (
    DispersiveFile, DispersiveParams, load_dispersive_file, save_dispersive_file,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate import value2flux
from zcu_tools.simulate.fluxonium import calculate_dispersive_vs_flux
from zcu_tools.notebook.analysis.plot import plot_dispersive_shift
from zcu_tools.analysis.fitting.resonance import (
    fit_edelay,
    remove_edelay,
    fit_circle_params,
    calc_phase,
)
```

```python
project_root = Path("../..").resolve()
entry_name = "fluxonium-analysis"
component = "Q1"  # Set the exact component name from this entry, not an alias.
bare_rf = 5.0  # Set the bare resonator frequency in GHz for this analysis.
register_all(component_registry)
entry = ResultEntry.open(
    entry_name,
    result_root=project_root / "result",
    database_root=project_root / "Database",
)
component_model = snapshot(entry.setup).components[component]
if component_model.kind != "qubit/fluxonium":
    raise ValueError(f"Expected qubit/fluxonium, got {component_model.kind!r}")
result_dir = entry.result_path
dispersive_path = entry.dispersive_path / f"{component}.json"

image_dir = result_dir / "image" / "dispersive"
web_dir = result_dir / "web" / "dispersive"
image_dir.mkdir(parents=True, exist_ok=True)
web_dir.mkdir(parents=True, exist_ok=True)
```

```python
predictor_file = load_predictor_file(entry.predictors_path / f"{component}.json")
if not isinstance(predictor_file.params, FluxoniumParams):
    raise ValueError("Select a fluxonium predictor")
predictor = predictor_file.params.build_predictor()
prior = (
    load_dispersive_file(dispersive_path).params
    if dispersive_path.exists() else DispersiveParams()
)

params = predictor.params
# value2flux below uses effective device coordinates, including stored bias.
flux_half = predictor.flux_half - predictor.flux_bias
flux_period = predictor.flux_period
flux_int = flux_half + 0.5 * flux_period

print("params = ", params, " GHz")
print("flux_half = ", flux_half)
print("flux_int = ", flux_int)
print("flux_period = ", flux_period)

g = prior.g if prior.g is not None else 0.05  # GHz
print(f"bare rf = {bare_rf}", "GHz")
print(f"g = {g}", "GHz")
```

# Plot with Onetone

```python
from zcu_lab.v2.onetone.flux_dep.core import FluxDepExp

onetone_path = r"../../Database/Q12_2D[5]/Q1/R1_flux_1.hdf5"

onetone_run = FluxDepExp().load(Path(onetone_path))
spectrum = onetone_run.result
sp_dev_values = spectrum.values
sp_freqs = spectrum.freqs * 1e-3  # MHz to GHz
sp_signals = spectrum.signals
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
best_g = g
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

# Export dispersive result

Artifact and setup writes are separate commits. An I/O or schema failure preserves
any successful earlier write. Only declared g/bare_rf fields are updated.

```python
entry.dispersive_path.mkdir(parents=True, exist_ok=True)
current = (
    load_dispersive_file(dispersive_path).params
    if dispersive_path.exists() else DispersiveParams()
)
save_dispersive_file(
    dispersive_path,
    DispersiveFile(
        format="zcu.dispersive", version=1,
        params=current.model_copy(update={"g": float(best_g), "bare_rf": float(bare_rf)}),
        source=AnalysisSource(
            run_id=None, created_at=datetime.now(timezone.utc).isoformat(),
        ),
    ),
)
entry.setup.refresh()
component_model = snapshot(entry.setup).components[component]
with entry.setup.edit() as draft:
    for field, value in (("g", best_g), ("bare_rf", bare_rf)):
        if field in type(component_model).model_fields:
            draft.set(f"{component}.{field}", float(value))
```
