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

import numpy as np
import pandas as pd

%autoreload 2
import zcu_tools.notebook.analysis.design as zd
import zcu_tools.notebook.analysis.plot as zp
import zcu_tools.simulate.equation as zeq
from zcu_tools.notebook.persistance import dump_result
from zcu_tools.notebook.analysis.branch import plot_chi_and_snr_over_photon
```

```python
qub_name = "DesignR59"

os.makedirs(f"../../result/{qub_name}/image/design", exist_ok=True)
os.makedirs(f"../../result/{qub_name}/web/design", exist_ok=True)
os.makedirs(f"../../result/{qub_name}/data/design", exist_ok=True)
```

# Scan params

```python
EJb = (4.0, 7.5)
ECb = (1.0, 1.5)
ELb = (0.4, 1.5)

flx = 0.5
r_f = 5.927
# r_f = 7.52062
g = 0.11

Temp = 113e-3
Q_cap = 4.0e5
Q_ind = 1.7e7
x_qp = 1.5e-6

noise_channels = [
    ("t1_capacitive", dict(Q_cap=Q_cap)),
    # ("t1_inductive", dict(Q_ind=Q_ind)),
    # ("t1_quasiparticle_tunneling", dict(x_qp=x_qp)),
]

avoid_freqs = [r_f]

flxs = np.linspace(0.0, 0.55, 1000)
```

```python
params_table = zd.generate_params_table(EJb, ECb, ELb, flx, precision=0.1)

zd.calculate_esys(params_table)
zd.calculate_f01(params_table)
zd.calculate_m01(params_table)
zd.calculate_t1(params_table, noise_channels, Temp)
# zd.calculate_dipersive_shift(params_table, g=g, r_f=r_f)
zd.calculate_snr(params_table, g, r_f, rf_w=7e-3, max_photon=70)
```

```python
params_table["valid"] = True
zd.avoid_collision(params_table, avoid_freqs, threshold=0.4)
zd.avoid_low_f01(params_table, f01_threshold=0.08)
zd.avoid_low_m01(params_table, m01_threshold=0.07)
result_table = params_table.drop(["esys"], axis=1)
result_table.to_parquet(f"../../result/{qub_name}/data/design_table.parquet")
result_table
```

```python
result_table = pd.read_parquet(f"../../result/{qub_name}/data/design_table.parquet")

fig = zd.plot_scan_results(result_table)
fig.update_layout(
    title=", ".join(
        ", ".join(f"{name} = {value:.1e}" for name, value in p_dict.items())
        for _, p_dict in noise_channels
    ),
    title_x=0.51,
)

best_params = zd.annotate_best_point(fig, result_table)
zd.add_real_sample(
    fig, "Q12_2D[2]/Q4", noise_channels=noise_channels, Temp=Temp, flx=flx
)

fig.show()
```

```python
save_name = f"t1vsSNR_rf{r_f:.2f}"
fig.write_html(
    f"../../result/{qub_name}/web/design/{save_name}.html", include_plotlyjs="cdn"
)
fig.write_image(f"../../result/{qub_name}/image/design/{save_name}.png", format="png")

dump_result(
    f"../../result/{qub_name}/params.json",
    name=qub_name,
    params=best_params,
    cflx=0.5,
    period=1.0,
    allows=dict(),
)
```

```python
# best_params = 6.1, 1.3, 1.15
best_params
```

```python
show_idxs = [(i, j) for i in range(2) for j in range(10) if j > i]

fig = zp.plot_transitions(
    best_params,
    flxs,
    show_idxs,
    ref_freqs=avoid_freqs + [2 * r_f, 3 * r_f, 4 * r_f, 5 * r_f, 6 * r_f],
)

fig.update_yaxes(range=(0.0, 14.0))
fig.update_layout(
    height=1200,
)
fig.show()
```

```python
save_name = f"f01_rf{r_f:.2f}"
fig.write_html(
    f"../../result/{qub_name}/web/design/{save_name}.html", include_plotlyjs="cdn"
)
fig.write_image(f"../../result/{qub_name}/image/design/{save_name}.png", format="png")
```

```python
show_idxs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]

fig, _ = zp.plot_matrix_elements(best_params, flxs, show_idxs)
fig.show()
```

```python
save_name = f"Matrix_rf{r_f:.2f}"
fig.write_html(
    f"../../result/{qub_name}/web/design/{save_name}.html", include_plotlyjs="cdn"
)
fig.write_image(f"../../result/{qub_name}/image/design/{save_name}.png", format="png")
```

```python
fig = zp.plot_dispersive_shift(best_params, flxs, r_f=r_f, g=g, upto=5)
fig.show()
```

```python
save_name = f"Chi_rf{r_f:.2f}"
fig.write_html(
    f"../../result/{qub_name}/web/design/{save_name}.html", include_plotlyjs="cdn"
)
fig.write_image(f"../../result/{qub_name}/image/design/{save_name}.png", format="png")
```

```python
rf_w = 0.006
g = 0.1
flx = 0.5

qub_dim = 20
qub_cutoff = 60
max_photon = 70

photons, chi_over_n, snrs = zd.calc_snr(
    best_params, r_f, g, flx, qub_dim, qub_cutoff, max_photon, rf_w
)

fig, (ax1, ax2) = plot_chi_and_snr_over_photon(photons, chi_over_n, snrs, qub_name, flx)
# ax1.set_ylim(0.002, 0.006)
# ax2.set_ylim(0.0, 1.5)

os.makedirs(f"../../result/{qub_name}/image/branch_floquet", exist_ok=True)
fig.savefig(f"../../result/{qub_name}/image/branch_floquet/snr_over_n.png")
```

```python
# Temp = 60e-3
# Q_cap = 1.0e5
# Q_ind = 1.0e7
# x_qp = 1.0e-8

fig, t1s = zp.plot_t1s(
    best_params,
    flxs,
    noise_channels=[
        ("t1_capacitive", dict(Q_cap=Q_cap)),
        # ("t1_inductive", dict(Q_ind=Q_ind)),
        # ("t1_quasiparticle_tunneling", dict(x_qp=x_qp)),
    ],
    Temp=Temp,
)
title1 = f"EJ/EC/EL = {best_params[0]:.3f}/{best_params[1]:.3f}/{best_params[2]:.3f}"
title2 = ", ".join(
    ", ".join(f"{name} = {value:.1e}" for name, value in p_dict.items())
    for _, p_dict in noise_channels
)
fig.update_layout(
    title=title1 + "<br>" + title2,
    title_x=0.515,
    yaxis_range=[np.log10(0.5 * t1s.min()), np.log10(1.5 * t1s.max())],
)
fig.show()
```

```python
save_name = f"T1_rf{r_f:.2f}"
fig.write_html(
    f"../../result/{qub_name}/web/design/{save_name}.html", include_plotlyjs="cdn"
)
fig.write_image(f"../../result/{qub_name}/image/design/{save_name}.png", format="png")
```

```python
import plotly.graph_objects as go
from zcu_tools.simulate.fluxonium import calculate_percell_t1_vs_flx

rf_w = 7e-3  # GHz

percell_t1s = calculate_percell_t1_vs_flx(
    flxs, r_f=r_f, kappa=rf_w, g=g, Temp=Temp, params=best_params
)
```

```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=flxs, y=1 / (1 / percell_t1s + 1 / t1s), mode="lines"))
fig.update_layout(
    xaxis_title="flux",
    yaxis_title="T1 (ns)",
    yaxis_type="log",
    yaxis_range=[np.log10(0.5 * t1s.min()), np.log10(1.5 * t1s.max())],
)
fig.update_yaxes(exponentformat="power")
fig.show()
```

```python
save_name = f"T1_percell_rf{r_f:.2f}"
fig.write_html(
    f"../../result/{qub_name}/web/design/{save_name}.html", include_plotlyjs="cdn"
)
fig.write_image(f"../../result/{qub_name}/image/design/{save_name}.png", format="png")
```

# EC to C

```python
project_name = "FluxoniumX400"
```

```python
EC = (1.1 * 1.1 / 0.9 * 1.1) ** 0.5
# EC = best_params[1]

Cap = zeq.EC2C(EC)
Lj = zeq.Cfreq2L(Cap, 6.3652)
# Lj = zeq.Cfreq2L(Cap, c_freq)

print(f"EC: {EC:.4g} GHz")
print(f"Capacitance: {Cap:.4g} fF")
print(f"Inductance: {Lj:.5g} nH")
```

```python
# result_path = f"../../result/{qub_name}/{project_name}/X325_Y51,5.csv"
result_path = f"../../result/{qub_name}/{project_name}/X35_Y2.csv"
fig, ax, c_Lj, c_freq, width = zd.fit_hfss_anticross(result_path)

hfss_C = zeq.Lfreq2C(c_Lj, c_freq)
hfss_EC = zeq.C2EC(hfss_C)
c_EL = zeq.L2EL(c_Lj)
g = width / zeq.n_coeff(hfss_EC, c_EL)

ax.set_title(f"EC={hfss_EC:.4g} GHz, g={1e3 * g:.4g} MHz")

fig.savefig(result_path.replace(".csv", ".png"))

print(f"hfss_C = {hfss_C:.4g} fF")
print(f"hfss_EC = {hfss_EC:.4g} GHz")
print(f"g = {1e3 * g:.4g} MHz")
print(f"Frequency: {c_freq:.5g} GHz")
```

```python
sweep_path = f"../../result/{qub_name}/{project_name}/X325_Y_sweep.csv"
fig, ax, max_y = zd.analyze_1d_sweep(sweep_path, c_freq, "Pad_Y [um]")
fig.savefig(sweep_path.replace(".csv", ".png"))
```

```python
sweep_path = f"../../result/{qub_name}/{project_name}/XY_sweep.csv"
fig, ax, max_x, max_y = zd.analyze_xy_sweep(sweep_path, c_freq)
fig.savefig(sweep_path.replace(".csv", ".png"))
```

```python

```
