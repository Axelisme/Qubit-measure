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
import os
from pprint import pprint
import numpy as np

%autoreload 2
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux
import zcu_tools.notebook.analysis.fluxdep as zf
import zcu_tools.notebook.persistance as zp
from zcu_tools.notebook.utils import savefig
from zcu_tools.simulate import value2flux, flux2value
import zcu_tools.experiment.v2 as ze
```

```python
chip_name = "Q3_2D[2]"
qub_name = "Q1"

flux_half = None
flux_int = None
flux_period = None
spectrums = dict[str, zp.SpectrumResult]()

result_dir = f"../../result/{chip_name}/{qub_name}"
image_dir = f"{result_dir}/image/fluxdep_fit"
web_dir = f"{result_dir}/web/fluxdep_fit"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(web_dir, exist_ok=True)
```

```python
loadpath = f"{result_dir}/params.json"
result = zp.load_result(loadpath)
fluxdep_result = result.get("fluxdep_fit")
assert fluxdep_result is not None, "No fluxdep_fit result found in the loaded data."

flux_half = fluxdep_result["flux_half"]
flux_period = fluxdep_result["flux_period"]
flux_period = fluxdep_result["flux_period"]
```

# Load Spectrum

```python
# spect_path = r"..\..\Database\Si001\2025\10\Data_1028\Si001_flux_1.hdf5"
spect_path = r"..\..\Database\Q3_2D[2]\Q1\2026\03\Data_0318\Q1_flux_1.hdf5"
# spect_path = r"..\..\Database\Q3_2D[2]\Q1\2026\03\Data_0316\R1_flux_6.hdf5"

# exp = ze.onetone.FluxDepExp()
exp = ze.twotone.FreqFluxExp()
dev_values, freqs, signals = exp.load(spect_path)
freqs *= 1e-3  # MHz -> GHz
```

```python
%matplotlib widget
actLine = zf.InteractiveLines(signals, dev_values, freqs, flux_half, flux_int)
```

```python
flux_half, flux_int = actLine.get_positions()
flux_period = 2 * abs(flux_int - flux_half)

fluxs = value2flux(dev_values, flux_half, flux_period)

flux_half, flux_int, flux_period
```

```python
%matplotlib widget
# actSel = zf.InteractiveOneTone(signals, dev_values, freqs, threshold=2.5)
actSel = zf.InteractiveFindPoints(signals, dev_values, freqs, threshold=6.0)
```

```python
ss_dev_values, ss_freqs = actSel.get_positions()
ss_fluxs = value2flux(ss_dev_values, flux_half, flux_period)
```

```python
name = os.path.basename(spect_path)
spectrums.update(
    {
        name: {
            "flux_half": flux_half,
            "flux_int": flux_int,
            "flux_period": flux_period,
            "spectrum": {
                "dev_values": dev_values,
                "fluxs": fluxs,
                "freqs": freqs,
                "signals": signals,
            },
            "points": {
                "dev_values": ss_dev_values,
                "fluxs": ss_fluxs,
                "freqs": ss_freqs,
            },
        }
    }
)
spectrums.keys()
```

# Save & Load

```python
processed_spect_path = f"{result_dir}/data/fluxdep/spectrums.hdf5"
zp.dump_spectrums(processed_spect_path, spectrums, mode="x")
```

```python
processed_spect_path = f"{result_dir}/data/fluxdep/spectrums.hdf5"
spectrums = zp.load_spectrums(processed_spect_path)
spectrums.keys()
```

# Align half flux

```python
s_selected = None
flux_bound = zf.derive_bound(spectrums, lambda s: s["spectrum"]["fluxs"])
freq_bound = zf.derive_bound(spectrums, lambda s: s["spectrum"]["freqs"])

t_fluxs = np.linspace(flux_bound[0], flux_bound[1], 1000)
t_dev_values = flux2value(t_fluxs, flux_half, flux_period)
```

# Manual Remove Points

```python
%matplotlib widget
intSel = zf.InteractiveSelector(spectrums, selected=s_selected)
```

```python
s_fluxs, s_freqs, s_selected = intSel.get_positions()
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
transitions = zp.TransitionDict(
    {
        "transitions": [(0, 1), (0, 2), (1, 2), (1, 3)],
        # "red side": [(0, 1)],
        "mirror": [(0, 1), (0, 2), (1, 2), (1, 3)],
        "r_f": 7.4445,  # GHz
        "sample_f": 9.584640,  # GHz
        # "sample_f": 6.881280,
    }
)

transitions.update(
    {
        # "transitions": [(i, j) for i in (0, 1) for j in range(5) if i < j],
        # "red side": [(i, j) for i in (0, 1) for j in range(10) if i < j],
        # "blue side": [(i, j) for i in (0, 1, 2) for j in range(8) if i < j],
        # "mirror": [(i, j) for i in (0, 1) for j in range(5) if i < j],
        # "transitions2": [(i, j) for i in (0, 1, 2) for j in range(11) if i < j],
        # "mirror2": [(i, j) for i in (0, 1, 2) for j in range(8) if i < j],
    }
)
```

```python
%matplotlib inline
database_path = r"../../Database/simulation/fluxonium_1.h5"
best_params, fig = zf.search_in_database(
    s_fluxs, s_freqs, database_path, transitions, EJb, ECb, ELb
)
savefig(fig, f"{image_dir}/search_result.png")
```

```python
_, energies = calculate_energy_vs_flux(best_params, t_fluxs, cutoff=40, evals_count=15)
```

```python
plot_transitions = zp.TransitionDict(
    {
        **transitions,
        # "transitions": [(0, 1), (0, 2), (1, 2), (1, 3)],
        # "red side": [(0, 1)],
        # "mirror": [(0, 2), (0, 3), (1, 3)],
        # "transitions": [(i, j) for i in (0, 1, 2) for j in range(i + 1, 15)],
        # "red side": [(i, j) for i in [0, 1, 2] for j in range(i + 1, 15)],
        # "mirror": [(i, j) for i in (0, 1) for j in range(i + 1, 15)],
        # "mirror red": [(i, j) for i in (0, 1) for j in range(i + 1, 15)],
        # "transitions2": [(0, 1), (0, 2), (1, 2), (0, 3)],
    }
)

r_f = plot_transitions.get("r_f", 0.0)
sample_f = plot_transitions.get("sample_f", 0.0)

fig = (
    zf.FreqFluxDependVisualizer()
    .plot_background(spectrums)
    .plot_simulation_lines(t_fluxs, energies, plot_transitions)
    .plot_points(s_fluxs, s_freqs, marker_color="blue", opacity=0.5)
    .plot_constant_freq(r_f, "r_f")
    .plot_constant_freq(0.5 * sample_f, "half sample_f")
    .plot_constant_freq(sample_f - r_f, "mirror r_f")
    .add_dev_values_ticks(t_fluxs, t_dev_values)
    .auto_derive_limits()
    .get_figure()
)
_ = fig.update_layout(
    title=f"EJ/EC/EL = ({best_params[0]:.2f}, {best_params[1]:.2f}, {best_params[2]:.2f})",
    title_x=0.501,
)
fig.update_layout(height=1000)
fig.show()
```

# Scipy Optimization

```python
# fit the spectrumData
sp_params = zf.fit_spectrum(s_fluxs, s_freqs, best_params, transitions, (EJb, ECb, ELb))

# print the results
print("Fitted params:", *sp_params)
```

```python
_, energies = calculate_energy_vs_flux(sp_params, t_fluxs, cutoff=40, evals_count=15)
```

```python
plot_transitions = zp.TransitionDict(
    {
        **transitions,
        # "transitions": [(i, j) for i in (0, 1) for j in range(i + 1, 5)],
        # "red side": [(i, j) for i in (0, 1) for j in range(i + 1, 15)],
    }
)

r_f = plot_transitions.get("r_f", 0.0)
sample_f = plot_transitions.get("sample_f", 0.0)

fig = (
    zf.FreqFluxDependVisualizer()
    .plot_background(spectrums)
    .plot_simulation_lines(t_fluxs, energies, plot_transitions)
    .plot_points(s_fluxs, s_freqs, marker_color="blue", opacity=0.5)
    .plot_constant_freq(r_f, "r_f")
    .plot_constant_freq(0.5 * sample_f, "half sample_f")
    .plot_constant_freq(sample_f - r_f, "mirror r_f")
    .add_dev_values_ticks(t_fluxs, t_dev_values)
    .auto_derive_limits()
    .get_figure()
)
_ = fig.update_layout(
    title=f"EJ/EC/EL = ({best_params[0]:.2f}, {best_params[1]:.2f}, {best_params[2]:.2f})",
    title_x=0.501,
)
fig.update_layout(height=1000)
fig.show()
```

```python
fig.write_html(f"{web_dir}/spect_fit.html", include_plotlyjs="cdn")
fig.write_image(f"{image_dir}/spect_fit.png", format="png")
```

# Save Parameters

```python
# dump the data
savepath = f"{result_dir}/params.json"

zp.dump_result(
    savepath,
    f"{chip_name}/{qub_name}",
    fluxdep_fit={
        "params": {
            "EJ": sp_params[0],
            "EC": sp_params[1],
            "EL": sp_params[2],
        },
        "flux_half": flux_half,
        "flux_int": flux_int,
        "flux_period": flux_period,
        "plot_transitions": plot_transitions,
    },
)
```

```python
savepath = f"{result_dir}/data/fluxdep/selected_points.npz"

np.savez(savepath, fluxs=s_fluxs, freqs=s_freqs, selected=s_selected)
```

```python

```
