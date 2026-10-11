# Branch populations from a result entry

This is an offline analysis example. It reads an existing new-format entry, typed `params.json` fits, and one point. It does not connect to instruments or accept values into the point. Run it only on a caller-selected analysis copy of the data.

## Definitions and entry

Register component definitions once per kernel. Restart the kernel before repeating registration. Opening requires an entry with setup, points, records, and its Database directory; it does not migrate old metadata files.

```python
from pathlib import Path

import numpy as np

from zcu_lab.components import register_all
from zcu_tools.notebook.analysis.mist.branch import plot_populations_over_photon
from zcu_tools.resources.entry import ResultEntry, component_registry
from zcu_tools.resources.qubit_params import QubitParams
from zcu_tools.simulate import value2flux
from zcu_tools.simulate.fluxonium.branch.full_quantum import (
    calc_branch_population,
    make_hilbertspace,
)

register_all(component_registry)
entry = ResultEntry.open(
    "Q1",
    result_root=Path("../../../analysis_copy/results/Q12_2D[5]"),
    database_root=Path("../../../analysis_copy/Database/Q12_2D[5]"),
)
```

## Typed fits and point values

The dispersive fit is required for this example. No guessed coupling or resonator frequency replaces a missing fit. Energies, resonator frequency, and coupling from params.json are GHz.

```python
params_file = QubitParams(entry.result_path / "params.json", readonly=True)
fit = params_file.require_fluxdep_fit()
dispersive = params_file.get_dispersive_fit()
if dispersive is None:
    raise ValueError("This branch analysis requires a dispersive fit")

point_label = "1.800mA"
point = entry.use_point(point_label)
roles = point.resolve(["qubit", "resonator"], focus="Q1")
flux = value2flux(
    point.general.ext.flux_value,
    roles.qubit.flux_half,
    roles.qubit.flux_period,
)
```

The three flux inputs must use the same explicit `general.ext.flux_unit`. Point reads use a cached snapshot, and the container does not convert current or voltage. The label is only a directory name; it is not parsed as a current.

## Calculate and export one branch plot

`upto` is the number of photon states, an integer, not the last float coordinate. The resonator dimension must cover that range. These cutoffs are example analysis choices; check convergence before interpreting the result.

```python
branches = [0, 1]
n_photons = 40
photons = np.arange(n_photons, dtype=np.float64)

hilbertspace = make_hilbertspace(
    params=fit.params,
    r_f=dispersive.bare_rf,
    g=dispersive.g,
    qub_dim=8,
    qub_cutoff=40,
    res_dim=50,
    flux=flux,
)
populations = calc_branch_population(hilbertspace, branches, upto=n_photons)
figure = plot_populations_over_photon(branches, photons, populations)
figure.show()
```

```python
output_dir = entry.result_path / "points" / point_label / "analysis_exports"
output_dir.mkdir(parents=True, exist_ok=True)
figure.write_html(output_dir / "branch_populations.html")
np.savez_compressed(
    output_dir / "branch_populations.npz",
    photons=photons,
    populations=np.stack([populations[branch] for branch in branches]),
)
```

These are explicit analysis exports, not a saved acquisition or accepted calibration. No acquired, saved, or accepted event is manufactured for this calculation. The former flux sweeps and Floquet/TLS exploration cells are retired; this notebook keeps one typed-input analysis route.
