---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

```python
%load_ext autoreload
import os
from datetime import datetime
from collections import OrderedDict

import numpy as np
from typing_extensions import cast

%autoreload 2
import zcu_tools.experiment.v2.autofluxdep as zefd
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.meta_manager import ExperimentManager
from zcu_tools.utils.datasaver import create_datafolder
from zcu_tools.notebook.utils import make_sweep
from zcu_tools.program.v2 import PulseCfg
```

```python
chip_name = "Test"
res_name = "R1"
qub_name = "Q1"

result_dir = f"../result/{chip_name}/{qub_name}"

database_path = create_datafolder(
    os.path.join(os.getcwd(), ".."), prefix=os.path.join(chip_name, qub_name)
)
em = ExperimentManager(f"{result_dir}/exps")
ml, md = em.use_flux(label="0305_1.800mA", readonly=True)
```

# Connect ZCU216

```python
from zcu_tools.remote import make_soc_proxy

soc, soccfg = make_soc_proxy("192.168.10.179", 8887)
print(soccfg)
```

```python
soc.get_sample_rates()
# soc.valid_sample_rates(tiletype='dac', tile=2)
```

# Connect Instruments

```python
import pyvisa
from zcu_tools.device import GlobalDeviceManager

resource_manager = pyvisa.ResourceManager()
```

## Flux Yoko

```python
from zcu_tools.device.yoko import YOKOGS200

flux_yoko = YOKOGS200("USB0::0x0B21::0x0039::90ZB35281::INSTR", resource_manager)
GlobalDeviceManager.register_device("flux_yoko", flux_yoko)

flux_yoko.set_mode("current", rampstep=1e-6)
# flux_yoko.set_mode("voltage", rampstep=1e-3)
```

# Initial Tools

```python
preditor = FluxoniumPredictor.from_file(f"{result_dir}/params.json")
preditor.A_c = md.mA_c
preditor.period = 2 * abs(md.mA_e - md.mA_c)
preditor.update_bias(md.bias)
```

# Start Measurement

```python
flux_yoko.set_current(1.20e-3)  # Set to initial flux bias
```

```python
%matplotlib widget
flx_values = np.linspace(1.5e-3, 1.8e-3, 151)

filename = f"{qub_name}_autofluxdep"

pi_pulse = cast(PulseCfg, ml.get_module("pi_amp"))
pi_product = pi_pulse["waveform"]["length"] * pi_pulse["gain"]
ref_pi_len = 1.2 * pi_pulse["waveform"]["length"]


executor = (
    zefd.FluxDepExecutor(flx_values=flx_values)
    .add_measurements(
        OrderedDict(
            qubit_freq=zefd.QubitFreqTask(
                detune_sweep=make_sweep(-2.5, 2.5, step=0.025),
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env["info"])
                    and (pred_qf := info["predict_freq"])
                    and (prev_pi_product := info.last.get("pi_product", pi_product))
                    and ml.make_cfg(
                        {
                            "modules": {
                                "qub_pulse": {
                                    "waveform": ml.get_waveform(
                                        "qub_flat", {"length": 6}
                                    ),
                                    "ch": md.qub_1_4_ch,
                                    "nqz": 2 if pred_qf > 2000 else 1,
                                    "gain": min(
                                        1.0,
                                        (0.003 / info["m_ratio"])
                                        * (prev_pi_product / ref_pi_len),
                                    ),
                                    "freq": pred_qf,
                                    # "mixer_freq": cur_qf,
                                },
                                "readout": "readout_dpm",
                            },
                            "relax_delay": 0.1,
                            "reps": 1000,
                            "rounds": 10,
                        }
                    )
                ),
                earlystop_snr=50,
            ),
            lenrabi=zefd.LenRabiTask(
                length_sweep=make_sweep(0.03, 1.0, 201),
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env["info"])
                    and (cur_qf := info.get("qubit_freq"))
                    and (prev_t1 := info.last.get("smooth_t1", md.t1))
                    and (prev_pi_product := info.last.get("pi_product", pi_product))
                    and ml.make_cfg(
                        {
                            "modules": {
                                "rabi_pulse": {
                                    **pi_pulse,
                                    "nqz": 2 if cur_qf > 2000 else 1,
                                    "freq": cur_qf,
                                    "gain": min(
                                        1.0,
                                        (pi_pulse["gain"] / info["m_ratio"])
                                        * (prev_pi_product / ref_pi_len),
                                    ),
                                    # "mixer_freq": cur_qf,
                                },
                                "readout": "readout_dpm",
                            },
                            "relax_delay": 3 * prev_t1,
                            "reps": 1000,
                            "rounds": 10,
                        }
                    )
                ),
                earlystop_snr=30,
            ),
            t1=zefd.T1Task(
                num_expts=151,
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env["info"])
                    and (prev_t1 := info.last.get("smooth_t1", md.t1))
                    and (cur_pi_pulse := info.get("pi_pulse"))
                    and ml.make_cfg(
                        {
                            "modules": {
                                "pi_pulse": cur_pi_pulse,
                                "readout": "readout_dpm",
                            },
                            "relax_delay": max(1.0, 3 * prev_t1),
                            "reps": 1000,
                            "rounds": 10,
                            "sweep_range": (0.5, max(1.0, 10 * prev_t1)),
                        }
                    )
                ),
                earlystop_snr=20,
            ),
            t2ramsey=zefd.T2RamseyTask(
                num_expts=151,
                detune_ratio=0.05,
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env["info"])
                    and (cur_t1 := info.get("smooth_t1", md.t1))
                    and (prev_t2r := info.last.get("smooth_t2r", md.t2r))
                    and (cur_pi2_pulse := info.get("pi2_pulse"))
                    and {
                        "modules": {
                            "pi2_pulse": cur_pi2_pulse,
                            "readout": "readout_dpm",
                        },
                        "relax_delay": max(1.0, 3 * cur_t1),
                        "reps": 1000,
                        "rounds": 10,
                        "sweep_range": (0, 2 * prev_t2r),
                    }
                ),
                earlystop_snr=20,
            ),
            t2echo=zefd.T2EchoTask(
                num_expts=151,
                detune_ratio=0.05,
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env["info"])
                    and (cur_t1 := info.get("smooth_t1", md.t1))
                    and (prev_t2e := info.last.get("smooth_t2e", md.t2e))
                    and (cur_pi_pulse := info.get("pi_pulse"))
                    and (cur_pi2_pulse := info.get("pi2_pulse"))
                    and {
                        "modules": {
                            "pi_pulse": cur_pi_pulse,
                            "pi2_pulse": cur_pi2_pulse,
                            "readout": "readout_dpm",
                        },
                        "relax_delay": max(1.0, 3 * cur_t1),
                        "reps": 1000,
                        "rounds": 10,
                        "sweep_range": (0, 2 * prev_t2e),
                    }
                ),
                earlystop_snr=20,
            ),
            mist_e=zefd.MistTask(
                gain_sweep=make_sweep(0.0, (100 / md.ac_stark_coeff) ** 0.5, 151),
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env["info"])
                    and (cur_t1 := info.get("smooth_t1", md.t1))
                    and (cur_pi_pulse := info.get("pi_pulse"))
                    and {
                        "modules": {
                            "mist_pulse": {
                                "waveform": ml.get_waveform(
                                    "mist_waveform",
                                    {"length": 5.0 / (2 * np.pi * md.rf_w) + 0.3},
                                ),
                                "ch": md.res_ch,
                                "nqz": 2,
                                "freq": md.readout_f,
                                "post_delay": 10 / (2 * np.pi * md.rf_w),
                            },
                            "pi_pulse": cur_pi_pulse,
                            "readout": "readout_dpm",
                        },
                        "relax_delay": max(1.0, 3 * cur_t1),
                        "reps": 2000,
                        "rounds": 10,
                    }
                ),
            ),
        )
    )
    .record_animation(f"{em.flx_dir}/{filename}.mp4")
)
_ = executor.run(
    dev_cfg={"flux_yoko": flux_yoko.get_info()},  # type: ignore
    predictor=preditor.clone(),
    env_dict={"soccfg": soccfg, "soc": soc, "ml": ml.clone()},
)
```

```python
executor.save(
    filepath=os.path.join(database_path, f"{filename}@{em.label}"),
    comment=datetime.now().strftime("Autofluxdep run at %Y-%m-%d %H:%M:%S")
    + "\n".join(f"{k}: {v}" for k, v in GlobalDeviceManager.get_all_info().items())
    + str(md),
)
```

```python

```
