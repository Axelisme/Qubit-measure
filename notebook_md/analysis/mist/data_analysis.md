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
import plotly.graph_objects as go


%autoreload 2
import zcu_tools.experiment.v2 as ze
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.experiment.v2.twotone.dispersive import DispersiveAnalyzeOptions
from zcu_tools.experiment.v2.twotone.ac_stark import AcStarkAnalyzeOptions
from zcu_tools.experiment.v2.singleshot.mist.power import PowerAnalyzeOptions
from zcu_tools.experiment.v2.mist.flux_dep import mist_signal2real
from zcu_tools.resources.qubit_params import QubitParams
from zcu_tools.simulate import mA2flx, flx2mA
```

```python
qub_name = "Q12_2D[6]/Q1"

result_dir = Path(f"../../../result/{qub_name}")
image_dir = result_dir / "image" / "mist_data_analysis" / "-4.000mA"
image_dir.mkdir(parents=True, exist_ok=True)
```

```python
params_file = QubitParams(result_dir / "params.json", readonly=True)
fit = params_file.require_fluxdep_fit()

params = fit.params
mA_c = fit.flux_half
period = fit.flux_period
allows = fit.plot_transitions
EJ, EC, EL = params

print(allows)

if "r_f" in allows:
    r_f = allows["r_f"]

if "sample_f" in allows:
    sample_f = allows["sample_f"]
```

# Dispersive shift

```python
filepath = (
    r"../../../Database/Q12_2D[5]/Q1/Q1_dispersive_shift_gain0.050@-2.600mA_3.hdf5"
)
exp = NotebookAdapter(ze.twotone.dispersive.DispersiveExp())
dispersive_run = exp.load(Path(filepath))
dispersive_analysis = exp.analyze(DispersiveAnalyzeOptions(), source=dispersive_run)
chi, kappa = dispersive_analysis.result.chi, dispersive_analysis.result.avg_fwhm
fig = dispersive_analysis.figures["fit"]
fig.savefig(image_dir / "dispersive_shift.png")
```

# CKP

Load one canonical CKP file containing the `Initial State` axis. Replace the example path with a file saved by the current CKP experiment; the old pair of ground/excited files is not this schema.

```python
filepath = result_dir / "data" / "ckp.hdf5"
exp = NotebookAdapter(ze.twotone.ckp.CKP_Exp())
ckp_run = exp.load(filepath)
ckp_analysis = exp.analyze(None, source=ckp_run)
chi, kappa, readout_f = ckp_analysis.result.chi, ckp_analysis.result.kappa, ckp_analysis.result.res_freq
fig = ckp_analysis.figures["fit"]
fig.savefig(image_dir / "dispersive_shift.png")
```

# AC stark shift

```python
filepath = (
    r"../../../Database/Q12_2D[6]/Q1/2026/01/Data_0131/Q1_ac_stark@1.800mA_1.hdf5"
)

exp = NotebookAdapter(ze.twotone.ac_stark.AcStarkExp())
ac_stark_run = exp.load(Path(filepath))
ac_stark_analysis = exp.analyze(
    AcStarkAnalyzeOptions(chi=chi, kappa=kappa, cutoff=0.1), source=ac_stark_run
)
ac_coeff = ac_stark_analysis.result.ac_coeff
fig = ac_stark_analysis.figures["fit"]
fig.savefig(image_dir / "ac_stark.png")
```

# Power dep

```python
%matplotlib inline
filepath = (
    "../../../Database/Q12_2D[5]/Q4/Q4_mist_g_singleshot_short@-4.000mA_2.hdf5"
    # "../../../Database/Q12_2D[5]/Q4/Q4_mist_e_singleshot_short@-4.000mA_3.hdf5"
    # "../../../Database/Q12_2D[5]/Q4/Q4_mist_g_singleshot_short@-0.650mA_1.hdf5"
    # "../../../Database/Q12_2D[5]/Q4/Q4_mist_e_singleshot_short@-0.650mA_2.hdf5"
)

exp = NotebookAdapter(ze.singleshot.mist.PowerExp())
power_run = exp.load(Path(filepath))
power_analysis = exp.analyze(
    PowerAnalyzeOptions(ac_coeff=ac_coeff, log_scale=True), source=power_run
)
fig = power_analysis.figures["fit"]
fig.savefig(image_dir / (filepath.split("/")[-1].split("@")[0] + ".png"))
```

# Power dep over flux

```python
sim_filepath = f"{result_dir}/data/branch_floquet/populations_over_flx.npz"
# sim_filepath = r"../../result/Q12_2D[3]/Q4/branch_populations.npz"


with np.load(sim_filepath) as data:
    sim_flxs = data["flxs"]
    sim_photons = data["photons"]
    branchs = data["branchs"]
    sim_populations = data["populations_over_flx"]
```

```python
filepaths = [
    r"../../../Database/Q12_2D[5]/Q1/Q1_mist_over_flux@-7.000mA_1.hdf5",
    r"../../../Database/Q12_2D[5]/Q1/Q1_autofluxdep_onlyfreq@-2.500mA_mist_g_signals_g_2.hdf5",
    # r"../../../Database/Q12_2D[5]/Q1/Q1_mist_over_flux@-2.600mA_1.hdf5",
    # r"../../../Database/Q12_2D[5]/Q1/Q1_autofluxdep_onlyfreq@-2.500mA_mist_e_signals_e_2.hdf5",
]
e_filepaths = [
    r"../../../Database/Q12_2D[5]/Q1/Q1_autofluxdep_onlyfreq@-2.500mA_mist_e_signals_e_2.hdf5"
]

# ac_coeff = 1e4
map_flxs = 1 - sim_flxs

from zcu_tools.notebook.analysis.mist.branch import plot_cn_with_mist
from zcu_tools.notebook.analysis.fluxdep import add_secondary_xaxis
from plotly.subplots import make_subplots

exp = ze.mist.flux_dep.FluxDepExp()

# The notebook owns this Plotly composition; the core returns numerical records.
fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1)
for row, paths in enumerate((filepaths, e_filepaths), start=1):
    for filepath in paths:
        mist_run = exp.load(Path(filepath))
        mist = mist_run.result
        fig.add_trace(
            go.Heatmap(
                x=mA2flx(mist.values, mA_c, period),
                y=ac_coeff * mist.gains**2,
                z=mist_signal2real(mist.signals).T,
                colorscale="Greys",
                showscale=False,
            ),
            row=row,
            col=1,
        )

plot_cn_with_mist(
    fig,
    flxs=sim_flxs,
    photons=sim_photons,
    populations_over_flx=sim_populations,
    critical_levels={0: 0.5, 1: 1.5},
    mist_flxs=map_flxs,
    row=1,
    col=1,
)
if e_filepaths:
    plot_cn_with_mist(
        fig,
        flxs=sim_flxs,
        photons=sim_photons,
        populations_over_flx=sim_populations,
        critical_levels={0: 0.5, 1: 1.5},
        mist_flxs=map_flxs,
        row=2,
        col=1,
    )

add_secondary_xaxis(fig, map_flxs, flx2mA(map_flxs, mA_c, period), row=2, col=1)

fig.update_layout(height=800)
fig.write_image(
    result_dir / "image" / "branch_floquet" / "mist_over_flux_with_simulation.png"
)
fig.show()
```

```python

```
