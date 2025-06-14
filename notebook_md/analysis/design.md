---
jupyter:
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

%autoreload 2
import zcu_tools.notebook.analysis.design as zd
import zcu_tools.notebook.analysis.plot as zp
```

```python
qub_name = "Design1"

os.makedirs(f"../../result/{qub_name}/image", exist_ok=True)
os.makedirs(f"../../result/{qub_name}/web", exist_ok=True)
```

# Scan params

```python
EJb = (2.0, 7.5)
EC = 0.8
# EC = 1.4
ELb = (0.35, 1.0)

flx = 0.6
r_f = 5.9
# r_f = 7.52994
g = 0.1
```

```python
Temp = 113e-3
Q_cap = 4.0e5
Q_ind = 1.7e7
x_qp = 1.5e-6

noise_channels = [
    ("t1_capacitive", dict(Q_cap=Q_cap)),
    # ("t1_inductive", dict(Q_ind=Q_ind)),
    # ("t1_quasiparticle_tunneling", dict(x_qp=x_qp)),
]

avoid_freqs = [r_f, 2 * r_f]


params_table = zd.generate_params_table(EJb, EC, ELb, flx)
```

```python
zd.calculate_esys(params_table)
zd.calculate_f01(params_table)
zd.calculate_m01(params_table)
zd.calculate_t1(params_table, noise_channels, Temp)
zd.calculate_dipersive_shift(params_table, g=g, r_f=r_f)
```

```python
params_table["valid"] = True
zd.avoid_collision(params_table, avoid_freqs, threshold=0.5)
zd.avoid_low_f01(params_table, f01_threshold=0.1)
zd.avoid_low_m01(params_table, m01_threshold=0.05)
params_table.drop(["esys"], axis=1)
```

```python
fig = zd.plot_scan_results(params_table)
fig.update_layout(
    title=", ".join(
        ", ".join(f"{name} = {value:.1e}" for name, value in p_dict.items())
        for _, p_dict in noise_channels
    ),
    title_x=0.51,
)

best_params = zd.annotate_best_point(fig, params_table)
# zd.add_real_sample(fig, "Q12_2D/Q4", noise_channels=noise_channels, Temp=Temp, flx=flx)

fig.show()
```

```python
save_name = f"t1vsChi_EC{EC:.2f}_rf{r_f:.2f}"
fig.write_html(f"../../result/{qub_name}/web/{save_name}.html", include_plotlyjs="cdn")
fig.write_image(f"../../result/{qub_name}/image/{save_name}.png", format="png")
```

```python
best_params = 4.0, 0.8, 0.4

flxs = np.linspace(0.0, 1.0, 1000)
best_params
```

```python
show_idxs = [(i, j) for i in range(2) for j in range(10) if j > i]

fig = zp.plot_transitions(best_params, flxs, show_idxs, ref_freqs=avoid_freqs)

fig.update_yaxes(range=(0.0, 14.0))
fig.update_layout(
    height=1200,
)
fig.show()
```

```python
save_name = f"f01_EC{EC:.2f}_rf{r_f:.2f}"
fig.write_html(f"../../result/{qub_name}/web/{save_name}.html", include_plotlyjs="cdn")
fig.write_image(f"../../result/{qub_name}/image/{save_name}.png", format="png")
```

```python
show_idxs = [(i, j) for i in range(2) for j in range(3) if j > i]

fig = zp.plot_matrix_elements(best_params, flxs, show_idxs)
fig.show()
```

```python
save_name = f"Matrix_EC{EC:.2f}_rf{r_f:.2f}"
fig.write_html(f"../../result/{qub_name}/web/{save_name}.html", include_plotlyjs="cdn")
fig.write_image(f"../../result/{qub_name}/image/{save_name}.png", format="png")
```

```python
fig = zp.plot_dispersive_shift(best_params, flxs, r_f=r_f, g=g)
fig.update_yaxes(range=(r_f - 0.01, r_f + 0.01))
fig.show()
```

```python
save_name = f"Chi_EC{EC:.2f}_rf{r_f:.2f}"
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
save_name = f"T1_EC{EC:.2f}_rf{r_f:.2f}"
fig.write_html(f"../../result/{qub_name}/web/{save_name}.html", include_plotlyjs="cdn")
fig.write_image(f"../../result/{qub_name}/image/{save_name}.png", format="png")
```

```python

```
