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
from unittest.mock import Mock

import numpy as np
import matplotlib.pyplot as plt

%autoreload 2
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.experiment.v2.autofluxdep import (
    FluxDepExecutor,
    QubitFreqMeasurementTask,
    LenRabiMeasurementTask,
    T1MeasurementTask,
    T2RamseyMeasurementTask,
    T2EchoMeasurementTask,
    MistMeasurementTask,
)
from zcu_tools.library import ModuleLibrary
from zcu_tools.notebook.utils import make_sweep
from zcu_tools.device import GlobalDeviceManager
```

```python
ml = ModuleLibrary()

predictor = FluxoniumPredictor(r"../result/Q12_2D[3]/Q4/params.json")

GlobalDeviceManager.register_device("flux_yoko", Mock())
```

```python
ml.register_module(
    qub_probe_pulse={
        "waveform": {"style": "const", "length": 1.0},
        "ch": 0,
        "nqz": 2,
        "freq": 1000,
        "gain": 0.1,
    },
    pi_pulse={
        "waveform": {"style": "const", "length": 0.5},
        "ch": 0,
        "nqz": 2,
        "freq": 1000,
        "gain": 0.2,
    },
    pi2_pulse={
        "waveform": {"style": "const", "length": 0.5},
        "ch": 0,
        "nqz": 2,
        "freq": 1000,
        "gain": 0.1,
    },
    mist_pulse={
        "waveform": {"style": "const", "length": 1.0},
        "ch": 0,
        "nqz": 2,
        "freq": 1000,
        "gain": 0.1,
    },
    readout_dpm={
        "type": "base",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": 1.0},
            "ch": 0,
            "nqz": 2,
            "freq": 1000,
            "gain": 1.0,
        },
        "ro_cfg": {
            "ro_ch": 0,
            "ro_length": 1.0,
            "triger_offset": 0.5,
        },
    },
)
```

```python
%matplotlib widget
flx_values = np.linspace(-1, 3.0, 31)

executor = (
    FluxDepExecutor(
        flx_values=flx_values,
        flux_dev_name="flux_yoko",
        flux_dev_cfg={
            "type": "YOKOGS200",
            "address": "...",
            "label": "flux_dev",
        },
        env_dict={
            "soccfg": Mock(),
            "soc": Mock(),
            "ml": ml,
            "predictor": predictor,
        },
    )
    .add_measurement(
        "qubit_freq",
        QubitFreqMeasurementTask(
            detune_sweep=make_sweep(-7, 7, 101),
            cfg_maker=lambda ctx, ml: (qub_pulse := ml.get_module("qub_probe_pulse"))
            and {
                "qub_pulse": {
                    **qub_pulse,
                    "gain": min(1.0, qub_pulse["gain"] * ctx.env_dict["gain_factor"]),
                },
                "readout": "readout_dpm",
                "relax_delay": 0.0,
                "reps": 1000,
                "rounds": 100,
            },
            earlystop_snr=20,
        ),
    )
    .add_measurement(
        "lenrabi",
        LenRabiMeasurementTask(
            length_sweep=make_sweep(0.0, 2.0, 101),
            ref_pi_product=0.2 * 0.5,
            cfg_maker=lambda ctx, ml: (pi_pulse := ml.get_module("pi_pulse"))
            and {
                "rabi_pulse": {
                    **pi_pulse,
                    "freq": ctx.env_dict["qubit_freq"],
                    "gain": min(1.0, pi_pulse["gain"] * ctx.env_dict["gain_factor"]),
                },
                "readout": "readout_dpm",
                "relax_delay": 0.0,
                "reps": 1000,
                "rounds": 100,
            },
            earlystop_snr=20,
        ),
    )
    .add_measurement(
        "t1",
        T1MeasurementTask(
            length_sweep=make_sweep(0.0, 15.0, 101),
            cfg_maker=lambda ctx, ml: (pi_pulse := ml.get_module("pi_pulse"))
            and {
                "rabi_pulse": {
                    **pi_pulse,
                    "freq": ctx.env_dict["qubit_freq"],
                    "gain": min(1.0, pi_pulse["gain"] * ctx.env_dict["gain_factor"]),
                },
                "readout": "readout_dpm",
                "relax_delay": 0.0,
                "reps": 1000,
                "rounds": 100,
            },
            earlystop_snr=20,
        ),
    )
    .add_measurement(
        "t2ramsey",
        T2RamseyMeasurementTask(
            length_sweep=make_sweep(0.0, 10.0, 101),
            activate_detune=1.0,
            cfg_maker=lambda ctx, ml: (pi2_pulse := ml.get_module("pi2_pulse"))
            and {
                "pi2_pulse": {
                    **pi2_pulse,
                    "freq": ctx.env_dict["qubit_freq"],
                    "gain": min(1.0, pi2_pulse["gain"] * ctx.env_dict["gain_factor"]),
                },
                "readout": "readout_dpm",
                "relax_delay": 0.0,
                "reps": 1000,
                "rounds": 100,
            },
            earlystop_snr=20,
        ),
    )
    .add_measurement(
        "t2echo",
        T2EchoMeasurementTask(
            length_sweep=make_sweep(0.0, 10.0, 101),
            activate_detune=1.0,
            cfg_maker=lambda ctx, ml: (pi_pulse := ml.get_module("pi_pulse"))
            and (pi2_pulse := ml.get_module("pi2_pulse"))
            and {
                "pi_pulse": {
                    **pi_pulse,
                    "freq": ctx.env_dict["qubit_freq"],
                    "gain": min(1.0, pi_pulse["gain"] * ctx.env_dict["gain_factor"]),
                },
                "pi2_pulse": {
                    **pi2_pulse,
                    "freq": ctx.env_dict["qubit_freq"],
                    "gain": min(1.0, pi2_pulse["gain"] * ctx.env_dict["gain_factor"]),
                },
                "readout": "readout_dpm",
                "relax_delay": 0.0,
                "reps": 1000,
                "rounds": 100,
            },
            earlystop_snr=20,
        ),
    )
    .add_measurement(
        "mist",
        MistMeasurementTask(
            gain_sweep=make_sweep(0.0, 1.0, 101),
            cfg_maker=lambda ctx, ml: (pi_pulse := ml.get_module("pi_pulse"))
            and {
                "mist_pulse": "mist_pulse",
                "pi_pulse": {
                    **pi_pulse,
                    "freq": ctx.env_dict["qubit_freq"],
                    "gain": min(1.0, pi_pulse["gain"] * ctx.env_dict["gain_factor"]),
                },
                "readout": "readout_dpm",
                "relax_delay": 0.0,
                "reps": 1000,
                "rounds": 100,
            },
        ),
    )
)
_, fig = executor.run()
plt.close(fig)
```

```python
# executor.save("./test")
```

```python

```
