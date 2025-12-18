```python
%load_ext autoreload
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

%autoreload 2
import zcu_tools.experiment.v2 as ze
from zcu_tools.utils.datasaver import load_data
from zcu_tools.simulate import mA2flx, flx2mA
from zcu_tools.notebook.persistance import load_result
from zcu_tools.notebook.analysis.mist.branch.overlay import calc_overlay, plot_overlay
```

```python
qub_name = "Q12_2D[5]/Q1"

result_dir = Path(f"../../../result/{qub_name}")
result_dir.mkdir(parents=True, exist_ok=True)
result_dir.joinpath("data").mkdir(exist_ok=True)
result_dir.joinpath("image").mkdir(exist_ok=True)
result_dir.joinpath("web").mkdir(exist_ok=True)
```

```python
_, params, mA_c, period, allows, data_dict = load_result(
    str(result_dir / "params.json")
)

if dispersive_cfg := data_dict.get("dispersive"):
    g = dispersive_cfg["g"]
    r_f = dispersive_cfg["r_f"]
    print(f"g: {g}, r_f: {r_f}")
elif "r_f" in allows:
    r_f = allows["r_f"]
    print(f"r_f: {r_f}")

if "sample_f" in allows:
    sample_f = allows["sample_f"]

# r_f = 7.520
rf_w = 11.2e-3
# g = 0.130

sim_flxs = np.linspace(-0.05, 0.55, 200)
```

```python
1e3 * mA_c, 1e3 * (mA_c - 0.5 * period)
```

# Process data

```python
%matplotlib widget
filepath = r"..\..\..\Database\Q12_2D[4]\Q4\2025\11\Data_1127\R4_flux_1.hdf5"

exp = ze.onetone.FluxDepExperiment()
flxs, fpts, signals = exp.load(filepath)

actline = exp.analyze()
```

```python
mA_c, mA_e = actline.get_positions()
period = 2 * abs(mA_e - mA_c)
1e3 * mA_c, 1e3 * mA_e
```

```python
%matplotlib inline
filepath = (
    r"..\..\..\Database\Q12_2D[4]\Q4\2025\11\Data_1114\R4_dispersive@4.000mA_1.hdf5"
)
# signals, fpts, _ = load_data(filepath)
# fpts /= 1e6

exp = ze.twotone.dispersive.DispersiveExperiment()
fpts, signals = exp.load(filepath)

chi, kappa, fig = exp.analyze()
plt.show(fig)
plt.close(fig)
```

```python
filepath = (
    r"..\..\..\Database\Q12_2D[4]\Q4\2025\11\Data_1114\Q4_ac_stark@4.000mA_1.hdf5"
)

exp = ze.twotone.ac_stark.AcStarkExperiment()
pdrs, fpts, signals = exp.load(filepath)

ac_coeff, fig = exp.analyze(chi=chi, kappa=kappa, cutoff=0.04)

plt.show(fig)
plt.close(fig)
```

```python
filepaths = [
    # r"..\..\..\Database\Q12_2D[4]\Q4\2025\11\Data_1116\Q4_mist_flux_bare@-4.990mA_1.hdf5",
    r"../../../Database/Q12_2D[5]/Q1/Q1_mist_over_flux@-7.000mA_1.hdf5"
]

ac_coeff = 1e3

fig = go.Figure()

for filepath in filepaths:
    signals, As, pdrs = load_data(filepath)
    assert pdrs is not None

    flxs = mA2flx(As, mA_c, period)
    photons = ac_coeff * pdrs**2

    real_signals = mist_signal2real(signals)

    fig.add_trace(
        go.Heatmap(
            x=flxs,
            y=photons,
            z=real_signals.T,
            colorscale="Greys",
            showscale=False,
        )
    )
fig.show()
```

# Overlay

```python
from tqdm.auto import tqdm
from joblib import Parallel, delayed

qub_dim = 30
qub_cutoff = 40
max_photon = 100

amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), rf_w)
sim_photons = (amps / (2 * g)) ** 2

overlay_over_flx = Parallel(n_jobs=-1)(
    delayed(calc_overlay)(params, sim_photons, r_f, g, flx, qub_dim, qub_cutoff)
    for flx in tqdm(sim_flxs)
)
overlay_over_flx = np.array(overlay_over_flx)  # shape: (flxs, photons, ge)
```

```python
np.savez(
    result_dir.joinpath("data", "overlay_over_flx.npz"),
    flxs=sim_flxs,
    photons=sim_photons,
    overlay_over_flx=overlay_over_flx,
)
```

```python
data = np.load(result_dir.joinpath("data", "overlay_over_flx.npz"))
sim_flxs = data["flxs"]
sim_photons = data["photons"]
overlay_over_flx = data["overlay_over_flx"]

sim_As = flx2mA(sim_flxs, mA_c, period)
```

```python
threshold = 0.9

map_flxs = 1 - sim_flxs

fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
fig.update_layout(height=600, margin=dict(t=10, b=20, l=20))

from zcu_tools.experiment.v2.twotone.mist.flux_dep import mist_signal2real

for filepath in filepaths:
    signals, As, pdrs = load_data(filepath)
    assert pdrs is not None

    flxs = mA2flx(As, mA_c, period)
    photons = ac_coeff * pdrs**2

    real_signals = mist_signal2real(signals)
    fig.add_trace(
        go.Heatmap(
            x=flxs,
            y=photons,
            z=real_signals.T,
            colorscale="Greys",
            showscale=False,
        )
    )


plot_overlay(
    fig,
    "Ground",
    overlay_over_flx[..., 0],
    map_flxs,
    sim_photons,
    threshold,
    row=2,
    line_kwargs=dict(color="blue"),
)
plot_overlay(
    fig,
    "Excited",
    overlay_over_flx[..., 1],
    map_flxs,
    sim_photons,
    threshold,
    row=3,
    line_kwargs=dict(color="red"),
)


fig.write_image(result_dir.joinpath("image", "mist_over_flux.png"))
fig.write_html(result_dir.joinpath("web", "mist_over_flux.html"))


fig.show()
```

```python

```
