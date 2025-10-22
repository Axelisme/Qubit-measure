---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
from zcu_tools.utils.fitting.singleshot import calc_population_pdf
import numpy as np
import matplotlib.pyplot as plt
```

```python
s = 0.2
p0 = 0.1
p_avg = 0.1

xs = np.linspace(-1.0, 2.0, 1000)
for ratio in [0.01, 0.1, 0.5, 1.0]:
    plt.plot(
        xs,
        calc_population_pdf(xs, 0, 1, s, p0, p_avg, ratio),
        label=f"ratio = {ratio}",
    )
plt.legend()
plt.show()
```

```python
s = 0.3
t1 = 1.0
p0 = 0.1
p_avg = 0.1
ratio = 1.0

true_params = [0.0, 1.0, s, p0, p_avg]

xs = np.linspace(-1.5, 2.5, 1000)
g_pdfs = calc_population_pdf(xs, 0.0, 1.0, s, p0, p_avg, ratio)
e_pdfs = calc_population_pdf(xs, 0.0, 1.0, s, 1 - p0, p_avg, ratio)
plt.plot(xs, g_pdfs, label="g")
plt.plot(xs, e_pdfs, label="e")
plt.legend()
plt.show()
```

```python
from zcu_tools.utils.datasaver import load_data
from zcu_tools.experiment.v2.twotone.singleshot import SingleShotExperiment

zdata, xdata, _ = load_data(
    "../Database/Test096/2025/04/Data_0403/single_shot_rf_q_-2.080mA_2.hdf5"
)
```

```python
single_shot_exp = SingleShotExperiment()
fid, _, _, pops, params = single_shot_exp.analyze(length_ratio=1.0, result=zdata.T)
```

```python
sg, se, s, p0, p_avg = params
print(f"Ideal snr: {abs(se - sg) / s: .2f}")
print(f"p0: {p0: .3f}")
print(f"p_avg: {p_avg: .3f}")
```

```python

```
