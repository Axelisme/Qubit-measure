---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: axelenv
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
    version: 3.8.20
---

```python
%load_ext autoreload
import os

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

%autoreload 2
import zcu_tools.experiment.v2 as ze
from zcu_tools.library import ModuleLibrary
from zcu_tools.notebook.utils import make_sweep, make_comment, savefig
from zcu_tools.utils.datasaver import create_datafolder
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
```

# Create data folder

```python
chip_name = r"Si001"

database_path = create_datafolder(os.path.join(os.getcwd(), ".."), prefix=chip_name)
ml = ModuleLibrary(cfg_path=f"../result/{chip_name}/module_cfg.yaml")
```

# Connect to zcu216

```python
from zcu_tools.remote import make_proxy

soc, soccfg = make_proxy("192.168.10.82", 8887)
print(soccfg)
```

# Predefine parameters

```python
res_ch = 0
# qub_0_1_ch = 11
# qub_1_4_ch = 14
# qub_5_6_ch = 2
# lo_flux_ch = 14

ro_ch = 0
```

# Initalze device

```python
import pyvisa
from zcu_tools.device import GlobalDeviceManager

resource_manager = pyvisa.ResourceManager()
```

## YOKOGS200

```python
from zcu_tools.device.yoko import YOKOGS200


flux_yoko = YOKOGS200(
    VISAaddress="USB0::0x0B21::0x0039::91S522309::INSTR", rm=resource_manager
)
GlobalDeviceManager.register_device("flux_yoko", flux_yoko)

flux_yoko.set_mode("current", rampstep=1e-6)
# flux_yoko.set_mode("voltage", rampstep=1e-3)
```

```python
cur_A = flux_yoko.get_current()
cur_A * 1e3
# cur_V = flux_yoko.get_voltage()
# cur_V
```

```python
# # cur_A = 0.0e-3
flux_yoko.set_current(current=cur_A)
# cur_V = 0.0
# flux_yoko.set_voltage(voltage=cur_V)
```

# Predictor

```python
preditor = FluxoniumPredictor(f"../result/{chip_name}/params.json")
# preditor = FluxoniumPredictor("../result/SF008/params.json")
```

# Lookback

```python
timeFly = 0.6
```

```python
exp_cfg = {
    "readout": {
        "type": "base",
        "pulse_cfg": {
            "waveform": {
                # "style": "const",
                "style": "padding",
                "length": 0.8,
                "pre_length": 0.05*0.2,
                "pre_gain": 1.0,
                "post_length": 0.03*0.2,
                "post_gain": -1.0,
            },
            "ch": res_ch,
            "nqz": 2,
            "gain": 0.5*0.2,
            # "freq": 5934,
            "freq": r_f,
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": 1.5,  # us
            "trig_offset": timeFly - 0.1,  # us
        },
    },
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, rounds=10000)

lookback_exp = ze.LookbackExperiment()
Ts, signals = lookback_exp.run(soc, soccfg, cfg)
```

```python
predict_offset = lookback_exp.analyze(
    ratio=0.1, smooth=1.0, ro_cfg=cfg["readout"]["ro_cfg"]
)
predict_offset
```

# OneTone Frequency

```python
probe_len = 5.65
exp_cfg = {
    "readout": {
        "type": "base",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": probe_len},
            "ch": res_ch,
            "nqz": 2,
            "gain": 0.5,
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": probe_len - 0.1,  # us
            "trig_offset": timeFly + 0.05,  # us
        },
    },
    # "sweep": make_sweep(5910, 5950, 101),
    "sweep": make_sweep(r_f - 10, r_f + 10, 301),
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

res_freq_exp = ze.onetone.FreqExperiment()
fpts, signals = res_freq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
r_f, rf_w, params, fig = res_freq_exp.analyze(model_type="auto")
```

# MIST

```python
qub_name = "Si001"

mist_pulse_len = 0.5  # us
ml.register_waveform(
    mist_waveform={
        "style": "const",
        "length": mist_pulse_len,  # us
    }
)
```

## Flux Power depedence

```python
r_f = ml.get_module("readout_rf")["pulse_cfg"]["freq"]
rf_w = 4.4
```

```python
exp_cfg = {
    "probe_pulse": {
        "waveform": ml.get_waveform("mist_waveform"),
        "ch": res_ch,
        "nqz": 2,
        "freq": r_f,
        "post_delay": 10 / (2 * np.pi * rf_w),
    },
    "readout": "readout_mist_dpm",
    "dev": {
        "flux_yoko": {
            "label": "flux_dev",
            # "mode": "voltage",
            "mode": "current",
        },
    },
    "sweep": {
        "flux": make_sweep(1.13e-3, 1.21e-3, 101),
        # "flux": make_sweep(preditor.flx_to_A(0.4), preditor.flx_to_A(1.1), 201),
        "gain": make_sweep(0.0, 1.0, 101),
    },
    "relax_delay": 10.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

mist_flux_exp = ze.twotone.flux_dep.MistExperiment()
values, gains, signals = mist_flux_exp.run(soc, soccfg, cfg)
```

```python
fig = mist_flux_exp.analyze()
savefig(
    fig,
    f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_mist_over_flux.png",
)
```

```python
mist_flux_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_mist_flux@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_mist_flux@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## Single trace

```python
cur_A = 1.162e-3
flux_yoko.set_current(current=cur_A)
# cur_V = 0.0
# flux_yoko.set_voltage(voltage=cur_V)
```

```python
exp_cfg = {
    "probe_pulse": {
        "waveform": ml.get_waveform("mist_waveform"),
        "ch": res_ch,
        "nqz": 2,
        "freq": r_f,
        "post_delay": 10 / (2 * np.pi * rf_w),
    },
    "readout": "readout_dpm",
    "sweep": {
        "gain": make_sweep(0.0, 1.0, 101),
    },
    "relax_delay": 10.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

mist_exp = ze.twotone.mist.MISTPowerDep()
gains, signals = mist_exp.run(soc, soccfg, cfg)
```

## Set Probe Pulse

```python
ml.register_module(
    mist_pulse={
        **exp_cfg["probe_pulse"],
        "gain": 0.7,
    },
)
```

## Overnight

```python
exp_cfg = {
    "probe_pulse": {
        "waveform": ml.get_waveform("mist_waveform"),
        "ch": res_ch,
        "nqz": 2,
        "freq": r_f,
        "post_delay": 10 / (2 * np.pi * rf_w),
    },
    "readout": "readout_dpm",
    "sweep": {
        "gain": make_sweep(0.0, 1.0, 101),
    },
    "relax_delay": 10.0,  # us
    "interval": 300, # s
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=20)

mist_overnight_exp = ze.twotone.mist.MISTPowerDepOvernight()
iters, gains, signals = mist_overnight_exp.run(soc, soccfg, cfg, num_times=120, fail_retry=2)
```

```python
mist_overnight_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_mist_overnight@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_mist_overnight@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## T1

```python
exp_cfg = {
    # "reset": "reset_120",
    "pi_pulse": "mist_pulse",
    "readout": "readout_dpm",
    "relax_delay": 20.0,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.0, 15.0, 51),
    # "sweep": make_sweep(0.01*t1, 5 * t1, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

t1_exp = ze.twotone.time_domain.T1Experiment()
Ts, signals = t1_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
t1, t1err, fig = t1_exp.analyze(max_contrast=True, dual_exp=False)
t1
```

## Optimize Readout

```python

```

### Frequency

```python
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": "mist_pulse",
    "readout": ml.get_module(
        "readout_rf",
        override_cfg={
            "pulse_cfg": {
                "waveform": {"length": 1.5},
                "gain": 0.5,
                # "gain": pdr_max,
            },
            "ro_cfg": {
                "ro_length": 1.4,
            },
        },
    ),
    "relax_delay": 10.0,  # us
    "sweep": make_sweep(r_f - 10, r_f + 10, 101),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=50)

opt_ro_freq_exp = ze.twotone.ro_optimize.OptimizeFreqExperiment()
fpts, snrs = opt_ro_freq_exp.run(soc, soccfg, cfg)
```

```python
fpt_max = opt_ro_freq_exp.analyze(smooth=1)
```

### Power

```python
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": "mist_pulse",
    "readout": ml.get_module(
        "readout_rf",
        override_cfg={
            "pulse_cfg": {
                "waveform": {"length": 1.5},
                "freq": fpt_max,
            },
            "ro_cfg": {
                "ro_length": 1.5,
            },
        },
    ),
    "relax_delay": 10.0,  # us
    # "relax_delay": 3 * t1,  # us
    "sweep": make_sweep(0.1, 1.0, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

opt_ro_pdr_exp = ze.twotone.ro_optimize.OptimizePowerExperiment()
pdrs, snrs = opt_ro_pdr_exp.run(soc, soccfg, cfg)
```

```python
pdr_max = opt_ro_pdr_exp.analyze()
pdr_max
```

### Length

```python
# pdr_max = 0.6
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": "mist_pulse",
    "readout": ml.get_module(
        "readout_rf",
        override_cfg={
            "pulse_cfg": {
                "freq": fpt_max,
                "gain": pdr_max,
            }
        },
    ),
    "relax_delay": 10.0,  # us
    # "relax_delay": 3 * t1,  # us
    "sweep": make_sweep(0.01, 10.0, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=10000, rounds=1)

opt_ro_len_exp = ze.twotone.ro_optimize.OptimizeLengthExperiment()
ro_lens, snrs = opt_ro_len_exp.run(soc, soccfg, cfg)
```

```python
ro_max = opt_ro_len_exp.analyze(t0=2.5 * t1)
ro_max
```

```python
# ro_max = 1.5
ml.register_module(
    readout_mist_dpm=ml.get_module(
        "readout_rf",
        {
            "pulse_cfg": {
                "freq": fpt_max,
                "gain": pdr_max,
                "waveform": {
                    "length": ro_max + 0.2,
                },
            },
            "ro_cfg": {
                "ro_length": ro_max,
            },
            "desc": "Readout with largest dispersive shift on MIST",
        },
    )
)
```

## SingleShot

```python
exp_cfg = {
    # "reset": "reset_120",
    "probe_pulse": "mist_pulse",
    "readout": "readout_mist_dpm",
    "relax_delay": 20.0,  # us
}
cfg = ml.make_cfg(exp_cfg, shots=500000)

singleshot_exp = ze.twotone.SingleShotExperiment()
signals = singleshot_exp.run(soc, soccfg, cfg)
```

```python
_ = singleshot_exp.analyze(init_p0=0.0)
```

```python

```
