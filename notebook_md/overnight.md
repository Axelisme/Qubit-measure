---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: zcu-tools
    language: python
    name: python3
---

```python
%load_ext autoreload
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

%autoreload 2
import zcu_tools.experiment.v2.overnight as zeo
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.library import ModuleLibrary
from zcu_tools.table import MetaDict
from zcu_tools.utils.datasaver import create_datafolder
from zcu_tools.notebook.utils import make_sweep
```

```python
chip_name = "Q12_2D[6]"
res_name = "R1"
qub_name = "Q1_fs6881"

result_dir = f"../result/{chip_name}/{qub_name}"

database_path = create_datafolder(
    os.path.join(os.getcwd(), ".."), prefix=os.path.join(chip_name, qub_name)
)
ml = ModuleLibrary(cfg_path=f"{result_dir}/module_cfg.yaml")
md = MetaDict(f"{result_dir}/meta_info.json")
```

# Connect ZCU216

```python
from zcu_tools.remote import make_proxy

soc, soccfg = make_proxy("192.168.10.179", 8887)
print(soccfg)
```

# Predefine Parameters

```python
res_ch = 0
qub_0_1_ch = 11
qub_1_4_ch = 2
# qub_4_5_ch = 5
# qub_5_6_ch = 5
# lo_flux_ch = 14

ro_ch = 0
```

# Start Measurement

```python
import gc

# del executor
plt.close("all")
gc.collect()
```

```python
%matplotlib widget
filename = f"{qub_name}_overnight"


executor = (
    zeo.OvernightExecutor(num_times=300, interval=120)
    .add_measurement(
        "mist_g",
        zeo.singleshot.MistTask(
            ml.make_cfg(
                {
                    # "reset": "reset_10",
                    "probe_pulse": {
                        "waveform": ml.get_waveform(
                            "mist_waveform",
                            {"length": 5.0 / (2 * np.pi * md.rf_w) + 0.3},
                        ),
                        "ch": res_ch,
                        "nqz": 2,
                        "freq": md.readout_f,
                        "post_delay": 10 / (2 * np.pi * md.rf_w),
                    },
                    "readout": "readout_dpm",
                    "sweep": {
                        "gain": make_sweep(0.0, 0.22, 101),
                    },
                    "relax_delay": 50.5,  # us
                },
                reps=3000,
                rounds=1,
            ),
            md.g_center,
            md.e_center,
            md.ge_radius,
        ),
    )
    .add_measurement(
        "mist_e",
        zeo.singleshot.MistTask(
            ml.make_cfg(
                {
                    # "reset": "reset_10",
                    "init_pulse": "pi_amp",
                    "probe_pulse": {
                        "waveform": ml.get_waveform(
                            "mist_waveform",
                            {"length": 5.0 / (2 * np.pi * md.rf_w) + 0.3},
                        ),
                        "ch": res_ch,
                        "nqz": 2,
                        "freq": md.readout_f,
                        "post_delay": 10 / (2 * np.pi * md.rf_w),
                    },
                    "readout": "readout_dpm",
                    "sweep": {
                        "gain": make_sweep(0.0, 0.22, 101),
                    },
                    "relax_delay": 50.5,  # us
                },
                reps=3000,
                rounds=1,
            ),
            md.g_center,
            md.e_center,
            md.ge_radius,
        ),
    )
    .add_measurement(
        "mist_steady",
        zeo.singleshot.MistTask(
            ml.make_cfg(
                {
                    "probe_pulse": {
                        "waveform": ml.get_waveform(
                            "mist_waveform",
                            {"length": 5.0 / (2 * np.pi * md.rf_w) + 50.0},
                        ),
                        "ch": res_ch,
                        "nqz": 2,
                        "freq": md.readout_f,
                        "post_delay": 10 / (2 * np.pi * md.rf_w),
                    },
                    "readout": "readout_dpm",
                    "sweep": {
                        "gain": make_sweep(0.0, 0.22, 101),
                    },
                    "relax_delay": 50.5,  # us
                },
                reps=3000,
                rounds=1,
            ),
            md.g_center,
            md.e_center,
            md.ge_radius,
        ),
    )
)
_ = executor.run(
    fail_retry=3,
    env_dict={"soccfg": soccfg, "soc": soc},
)
```

```python
executor.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    comment=datetime.now().strftime("Overnight run at %Y-%m-%d %H:%M:%S"),
)
del executor
```

```python

```
