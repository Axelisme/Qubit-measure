---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: axelenv
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
import zcu_tools.experiment.v2.autofluxdep as zefd
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.library import ModuleLibrary
from zcu_tools.table import MetaDict
from zcu_tools.utils.datasaver import create_datafolder
from zcu_tools.notebook.utils import make_sweep
```

```python
chip_name = "Q12_2D[5]"
res_name = "R1"
qub_name = "Q1"

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

soc, soccfg = make_proxy("192.168.10.63", 8887)
print(soccfg)
```

# Predefine Parameters

```python
res_ch = 0
# qub_0_1_ch = 11
qub_1_4_ch = 2
qub_4_5_ch = 5
# qub_5_6_ch = 5
# lo_flux_ch = 14

ro_ch = 0
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


flux_yoko = YOKOGS200(
    address="USB0::0x0B21::0x0039::91WB18859::INSTR", rm=resource_manager
)
GlobalDeviceManager.register_device("flux_yoko", flux_yoko)

flux_yoko.set_mode("current", rampstep=1e-6)
# flux_yoko.set_mode("voltage", rampstep=1e-3)
```

# Initial Tools

```python
preditor = FluxoniumPredictor.from_file(f"{result_dir}/params.json")
preditor.A_c = md.mA_c
preditor.period = 2 * abs(md.mA_e - md.mA_c)
```

# Start Measurement

```python
flux_yoko.set_current(-2.8e-3)
```

```python
import gc

# del executor
plt.close("all")
gc.collect()
```

```python
%matplotlib widget
flx_values = np.linspace(-2.8e-3, -2.2e-3, 251)

filename = f"{qub_name}_autofluxdep_animation"

init_pi_pulse = ml.get_module("pi_amp")

executor = (
    zefd.FluxDepExecutor(flx_values=flx_values)
    .add_measurement(
        "qubit_freq",
        zefd.QubitFreqMeasurementTask(
            detune_sweep=make_sweep(-5, 5, step=0.025),
            cfg_maker=lambda ctx, ml: (cur_qf := ctx.env_dict["info"]["predict_freq"])
            and zefd.QubitFreqCfgTemplate(
                {
                    "qub_pulse": {
                        "waveform": ml.get_waveform(
                            "qub_flat", override_cfg={"length": 3}
                        ),
                        "ch": qub_1_4_ch,
                        "nqz": 2 if cur_qf > 2000 else 1,
                        "gain": min(
                            1.0,
                            0.01
                            * ctx.env_dict["info"].last.get("gain_factor", 1.0)
                            / ctx.env_dict["info"]["m_ratio"],
                        ),
                        "freq": cur_qf,
                        # "mixer_freq": cur_qf,
                    },
                    "readout": "readout_dpm",
                    "relax_delay": 0.1,
                    "reps": 200,
                    "rounds": 50,
                }
            ),
            earlystop_snr=50,
        ),
    )
    .add_measurement(
        "lenrabi",
        zefd.LenRabiMeasurementTask(
            length_sweep=make_sweep(0.03, 1.0, 201),
            ref_pi_product=3
            * init_pi_pulse["waveform"]["length"]
            * init_pi_pulse["gain"],
            cfg_maker=lambda ctx, ml: (pi_pulse := ml.get_module("pi_len"))
            and (cur_qf := ctx.env_dict["info"].get("qubit_freq"))
            and (cur_t1 := ctx.env_dict["info"].last.get("smooth_t1", md.t1))
            and zefd.LenRabiCfgTemplate(
                {
                    "rabi_pulse": {
                        **pi_pulse,
                        "nqz": 2 if cur_qf > 2000 else 1,
                        "freq": cur_qf,
                        "gain": min(
                            1.0,
                            0.4
                            * pi_pulse["gain"]
                            * ctx.env_dict["info"].last.get("gain_factor", 1.0)
                            / ctx.env_dict["info"]["m_ratio"]
                            * ctx.env_dict["info"].first["fit_kappa"]
                            / ctx.env_dict["info"]["fit_kappa"],
                        ),
                        # "mixer_freq": cur_qf,
                    },
                    "readout": "readout_dpm",
                    "relax_delay": 3 * cur_t1,
                    "reps": 1000,
                    "rounds": 10,
                }
            ),
            earlystop_snr=30,
        ),
    )
    # .add_measurement(
    #     "t1",
    #     zefd.T1MeasurementTask(
    #         num_expts=151,
    #         cfg_maker=lambda ctx, ml: (
    #             cur_t1 := ctx.env_dict["info"].last.get("smooth_t1", md.t1)
    #         )
    #         and (cur_pi_pulse := ctx.env_dict["info"].get("pi_pulse"))
    #         and zefd.T1CfgTemplate(
    #             {
    #                 "pi_pulse": cur_pi_pulse,
    #                 "readout": "readout_dpm",
    #                 "relax_delay": max(1.0, 3 * cur_t1),
    #                 "reps": 1000,
    #                 "rounds": 10,
    #                 "sweep_range": (0.5, max(1.0, 10 * cur_t1)),
    #             }
    #         ),
    #         earlystop_snr=20,
    #     ),
    # )
    # .add_measurement(
    #     "t2ramsey",
    #     zefd.T2RamseyMeasurementTask(
    #         num_expts=151,
    #         detune_ratio=0.05,
    #         cfg_maker=lambda ctx, ml: (
    #             cur_t1 := ctx.env_dict["info"].get("smooth_t1", md.t1)
    #         )
    #         and (cur_t2r := ctx.env_dict["info"].last.get("smooth_t2r", md.t2r))
    #         and (cur_pi2_pulse := ctx.env_dict["info"].get("pi2_pulse"))
    #         and zefd.T2RamseyCfgTemplate(
    #             {
    #                 "pi2_pulse": cur_pi2_pulse,
    #                 "readout": "readout_dpm",
    #                 "relax_delay": max(1.0, 3 * cur_t1),
    #                 "reps": 1000,
    #                 "rounds": 10,
    #                 "sweep_range": (0, 2 * cur_t2r),
    #             }
    #         ),
    #         earlystop_snr=20,
    #     ),
    # )
    # .add_measurement(
    #     "t2echo",
    #     zefd.T2EchoMeasurementTask(
    #         num_expts=151,
    #         detune_ratio=0.05,
    #         cfg_maker=lambda ctx, ml: (
    #             cur_t1 := ctx.env_dict["info"].get("smooth_t1", md.t1)
    #         )
    #         and (cur_t2e := ctx.env_dict["info"].last.get("smooth_t2e", md.t2e))
    #         and (cur_pi_pulse := ctx.env_dict["info"].get("pi_pulse"))
    #         and (cur_pi2_pulse := ctx.env_dict["info"].get("pi2_pulse"))
    #         and zefd.T2EchoCfgTemplate(
    #             {
    #                 "pi_pulse": cur_pi_pulse,
    #                 "pi2_pulse": cur_pi2_pulse,
    #                 "readout": "readout_dpm",
    #                 "relax_delay": max(1.0, 3 * cur_t1),
    #                 "reps": 1000,
    #                 "rounds": 10,
    #                 "sweep_range": (0, 2 * cur_t2e),
    #             }
    #         ),
    #         earlystop_snr=20,
    #     ),
    # )
    .add_measurement(
        "mist_g",
        zefd.Mist_G_MeasurementTask(
            gain_sweep=make_sweep(0.0, 0.1, 71),
            cfg_maker=lambda ctx, ml: (
                cur_t1 := ctx.env_dict["info"].get("smooth_t1", md.t1)
            )
            and zefd.Mist_G_CfgTemplate(
                {
                    "mist_pulse": {
                        "waveform": ml.get_waveform("mist_waveform"),
                        "ch": res_ch,
                        "nqz": 2,
                        "freq": md.r_f,
                        "post_delay": 10 / (2 * np.pi * md.rf_w),
                    },
                    "readout": "readout_dpm",
                    "relax_delay": max(1.0, 3 * cur_t1),
                    "reps": 1000,
                    "rounds": 10,
                }
            ),
        ),
    )
    .add_measurement(
        "mist_e",
        zefd.Mist_E_MeasurementTask(
            gain_sweep=make_sweep(0.0, 0.1, 71),
            cfg_maker=lambda ctx, ml: (
                cur_t1 := ctx.env_dict["info"].get("smooth_t1", md.t1)
            )
            and (cur_pi_pulse := ctx.env_dict["info"].get("pi_pulse"))
            and zefd.Mist_E_CfgTemplate(
                {
                    "mist_pulse": {
                        "waveform": ml.get_waveform("mist_waveform"),
                        "ch": res_ch,
                        "nqz": 2,
                        "freq": md.r_f,
                        "post_delay": 10 / (2 * np.pi * md.rf_w),
                    },
                    "pi_pulse": cur_pi_pulse,
                    "readout": "readout_dpm",
                    "relax_delay": max(1.0, 3 * cur_t1),
                    "reps": 1000,
                    "rounds": 10,
                }
            ),
        ),
    )
    .record_animation(f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.mp4")
)
_ = executor.run(
    dev_cfg={
        "flux_yoko": {
            "label": "flux_dev",
            "mode": "current",
            **flux_yoko.get_info(),
        }
    },
    predictor=preditor.clone(),
    env_dict={
        "soccfg": soccfg,
        "soc": soc,
        "ml": ml.clone(),
    },
)
```

```python
executor.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    comment=datetime.now().strftime("Autofluxdep run at %Y-%m-%d %H:%M:%S"),
)
```

```python

```
