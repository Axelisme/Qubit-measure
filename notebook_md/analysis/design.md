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
```

```python
qub_name = "Design400"

os.makedirs(f"../../result/{qub_name}/image", exist_ok=True)
os.makedirs(f"../../result/{qub_name}/web", exist_ok=True)
```

# Scan params

```python
EJb = (4.0, 7.5)
ECb = (1.0, 1.5)
ELb = (0.4, 1.5)

flx = 0.5
r_f = 5.927
# r_f = 7.52994
g = 0.1

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
```

```python
params_table = zd.generate_params_table(EJb, ECb, ELb, flx)

zd.calculate_esys(params_table)
zd.calculate_f01(params_table)
zd.calculate_m01(params_table)
zd.calculate_t1(params_table, noise_channels, Temp)
zd.calculate_dipersive_shift(params_table, g=g, r_f=r_f)
```

```python
params_table["valid"] = True
zd.avoid_collision(params_table, avoid_freqs, threshold=0.4)
zd.avoid_low_f01(params_table, f01_threshold=0.08)
zd.avoid_low_m01(params_table, m01_threshold=0.07)
result_table = params_table.drop(["esys"], axis=1)
result_table.to_parquet(f"../../result/{qub_name}/result_table.parquet")
result_table
```

```python
result_table = pd.read_parquet(f"../../result/{qub_name}/result_table.parquet")

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
save_name = f"t1vsChi_rf{r_f:.2f}"
fig.write_html(f"../../result/{qub_name}/web/{save_name}.html", include_plotlyjs="cdn")
fig.write_image(f"../../result/{qub_name}/image/{save_name}.png", format="png")
```

```python
# best_params = 6.1, 1.3, 1.15

flxs = np.linspace(0.0, 1.0, 1000)
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
fig.write_html(f"../../result/{qub_name}/web/{save_name}.html", include_plotlyjs="cdn")
fig.write_image(f"../../result/{qub_name}/image/{save_name}.png", format="png")
```

```python
show_idxs = [(i, j) for i in range(2) for j in range(3) if j > i]

fig, _ = zp.plot_matrix_elements(best_params, flxs, show_idxs)
fig.show()
```

```python
save_name = f"Matrix_rf{r_f:.2f}"
fig.write_html(f"../../result/{qub_name}/web/{save_name}.html", include_plotlyjs="cdn")
fig.write_image(f"../../result/{qub_name}/image/{save_name}.png", format="png")
```

```python
fig = zp.plot_dispersive_shift(best_params, flxs, r_f=r_f, g=g, upto=5)
fig.show()
```

```python
save_name = f"Chi_rf{r_f:.2f}"
fig.write_html(f"../../result/{qub_name}/web/{save_name}.html", include_plotlyjs="cdn")
fig.write_image(f"../../result/{qub_name}/image/{save_name}.png", format="png")
```

```python
# Temp = 60e-3
# Q_cap = 1.0e5
# Q_ind = 1.0e7
# x_qp = 1.0e-8

fig = zp.plot_t1s(
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
)
fig.show()
```

```python
save_name = f"T1_rf{r_f:.2f}"
fig.write_html(f"../../result/{qub_name}/web/{save_name}.html", include_plotlyjs="cdn")
fig.write_image(f"../../result/{qub_name}/image/{save_name}.png", format="png")
```

# EC to C

```python
EC = 1.1
# EC = best_params[1]

Cap = zeq.EC2C(EC)
Lj = zeq.Cfreq2L(Cap, 6.4324)
# Lj = zeq.Cfreq2L(Cap, c_freq)

print(f"Capacitance: {Cap:.4g} fF")
print(f"Inductance: {Lj:.4g} nH")
```

```python
result_path = f"../../result/{qub_name}/E_X380_Y6.csv"
fig, ax, c_Lj, c_freq, width = zd.fit_hfss_anticross(result_path)
fig.savefig(result_path.replace(qub_name, f"{qub_name}/image").replace(".csv", ".png"))
print(f"Frequency: {c_freq:.5g} GHz")
```

```python
hfss_C = zeq.Lfreq2C(c_Lj, c_freq)
hfss_EC = zeq.C2EC(hfss_C)
c_EL = zeq.L2EL(c_Lj)

g = width / zeq.n_coeff(hfss_EC, c_EL)
print(f"hfss_C = {hfss_C:.4g} fF")
print(f"hfss_EC = {hfss_EC:.4g} GHz")
print(f"g = {1e3 * g:.4g} MHz")

```

```python
sweep_path = f"../../result/{qub_name}/Y_sweep.csv"
fig, ax, max_y = zd.analyze_1d_sweep(sweep_path, c_freq, "Pad_Y [um]")
fig.savefig(f"../../result/{qub_name}/image/g_over_y_sweep.png")
```

```python
sweep_path = f"../../result/{qub_name}/XY_sweep.csv"
fig, ax, max_x, max_y = zd.analyze_xy_sweep(sweep_path, c_freq)
fig.savefig(f"../../result/{qub_name}/image/g_over_xy_sweep.png")
```

```python

```
