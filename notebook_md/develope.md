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
from zcu_tools.utils.datasaver import load_data
from zcu_tools.experiment.v2.onetone.freq import FreqExperiment
from zcu_tools.experiment.v2.twotone.singleshot import SingleShotExperiment
```

```python
# signals, fpts, _ = load_data("../Database/Test096/Res3D_freq@1.000mA_1.hdf5")
signals, fpts, _ = load_data("../Database/Q12_2D[3]/Q4/R4_freq@-0.414mA_1.hdf5")
# signals, fpts, _ = load_data("../Database/Q3_2D/Q2/R2_freq@-2.300V_1.hdf5")
# signals, fpts, _ = load_data("../Database/Q3_2D/Q2/R2_freq@1220.000mV_1.hdf5")
fpts *= 1e-6

freq_exp = FreqExperiment()
freq, kappa, param_dict, fig = freq_exp.analyze(result=(fpts, signals))
```

```python
signals, *_ = load_data(
    "../Database/Test096/2025/04/Data_0403/single_shot_rf_q_-2.080mA_2.hdf5"
)
signals = signals.T

single_shot_exp = SingleShotExperiment()
_ = single_shot_exp.analyze(result=(signals,), init_p0=None, length_ratio=None)
```

```python

```
