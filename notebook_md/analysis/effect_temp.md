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
    display_name: zcu-tools (3.13.11.final.0)
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

# Effective Temperature

This notebook estimates output-referred thermal noise from a passive attenuator chain.

```python
%load_ext autoreload

%matplotlib inline
%autoreload 2

from pathlib import Path

import numpy as np

import zcu_tools.notebook.analysis.fit_tools as zfit
```

```python
output_dir = Path("../../result/Eff_T")
output_dir.mkdir(parents=True, exist_ok=True)
```

# User Settings

```python
probe_frequency_hz = 5.79313e9
impedance_ohm = 50.0

frequency_axis_hz = np.geomspace(1e6, 100e9, 1000)
temperature_axis_hz = np.linspace(10e6, 10e9, 1000)
highlight_frequency_range_hz = (0.4e9, 4.5e9)
```

```python
input_temperature_K = 300.0

stages = (
    zfit.ThermalAttenuatorStage("50K", Temp_K=50.0, attenuation_db=13.0),
    zfit.ThermalAttenuatorStage("4K", Temp_K=4.0, attenuation_db=13.0),
    zfit.ThermalAttenuatorStage("100mK", Temp_K=0.1, attenuation_db=13.0),
    zfit.ThermalAttenuatorStage("20mK", Temp_K=0.02, attenuation_db=23.0),
)
```

# Calculate

```python
noise = zfit.calculate_thermal_chain(
    frequency_axis_hz,
    stages,
    input_temperature_K=input_temperature_K,
    impedance_ohm=impedance_ohm,
)
probe = noise.probe(probe_frequency_hz)

print(
    f"T_eff({probe.frequency_hz * 1e-9:.4g} GHz) = "
    f"{probe.effective_temperature_K * 1e3:.2f} mK"
)
print(f"n_photon({probe.frequency_hz * 1e-9:.4g} GHz) = {probe.photon_number:.4g}")
```

# PSD

```python
fig, _ = zfit.plot_thermal_chain_psd(
    noise,
    probe_frequency_hz=probe_frequency_hz,
    ylim=(-30.5, -17.5),
)
fig.savefig(
    output_dir / f"{probe_frequency_hz * 1e-9:.3f}GHz_01.png",
    bbox_inches="tight",
)
```

# Effective Temperature

```python
temperature_curve = zfit.calculate_thermal_chain(
    temperature_axis_hz,
    stages,
    input_temperature_K=input_temperature_K,
    impedance_ohm=impedance_ohm,
)
```

```python
fig, _ = zfit.plot_effective_temperature_vs_frequency(
    temperature_curve,
    probe_frequency_hz=probe_frequency_hz,
    highlight_range_hz=highlight_frequency_range_hz,
)
fig.savefig(output_dir / "Eff_T_vs_Freq.png", bbox_inches="tight")
```

```python
np.savez(
    output_dir / "Eff_T_vs_Freq.npz",
    fpts=temperature_curve.frequencies_hz,
    T_effs=temperature_curve.effective_temperature_K,
    n_photons=temperature_curve.effective_photon_number,
)
```
