---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: zcu-tools (3.9.25)
    language: python
    name: python3
---

```python
%load_ext autoreload
import os
from pprint import pprint
from pathlib import Path
import json
from collections import OrderedDict

import numpy as np
from typing_extensions import cast

%autoreload 2
import zcu_tools.experiment.v2.autofluxdep as zefd
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.meta_tool import ExperimentManager
from zcu_tools.utils.datasaver import create_datafolder
from zcu_tools.notebook.utils import make_sweep, reconnect_devices, dump_device_info
from zcu_tools.program.v2 import PulseCfg
```

```python
chip_name = "Q3_2D[2]"
res_name = "R1"
qub_name = "Q1"

result_dir = os.path.join("..", "result", chip_name, qub_name)
database_path = create_datafolder(
    database_dir=os.path.join("..", "Database"),
    name=os.path.join(chip_name, qub_name),
)

em = ExperimentManager(os.path.join(result_dir, "exps"))
ml, md = em.use_flux(label="032023_0.122mA", readonly=True)
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
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.yoko import YOKOGS200

dev_info_path = os.path.join(em.flx_dir, "device_info.json")

with open(dev_info_path, "r") as f:
    dev_info = json.load(f)
pprint(dev_info)

resource_manager = reconnect_devices(dev_info)

flux_yoko = cast(YOKOGS200, GlobalDeviceManager.get_device("flux_yoko"))

GlobalDeviceManager.setup_devices(dev_info, progress=True)
```

# Initial Tools

```python
preditor = FluxoniumPredictor.from_file(os.path.join(result_dir, "params.json"))
# preditor.flx_half = md.flx_half
# preditor.flx_period = 2 * abs(md.flx_int - md.flx_half)
preditor.update_bias(md.flx_bias)
```

# Start Measurement

```python
flux_yoko.set_current(0.122e-3)  # Set to initial flux bias
```

```python
import gc
import matplotlib.pyplot as plt

plt.close("all")
gc.collect()
```

```python
%matplotlib widget
flx_values = np.linspace(0.122e-3, -3.5e-3, 101)

filename = f"{qub_name}_autofluxdep"

# snapshot of execution code
measure_code: str = In[-1]  # noqa: F821 # type: ignore

pi_pulse = cast(PulseCfg, ml.get_module("pi_amp"))
pi_len = pi_pulse["waveform"]["length"]
pi_product = pi_len * pi_pulse["gain"]

readout_cfg = ml.get_module("readout_dpm")
readout_freq = readout_cfg["pulse_cfg"]["freq"]
readout_gain = readout_cfg["pulse_cfg"]["gain"]


executor = (
    zefd.FluxDepExecutor(flx_values=flx_values)
    .add_measurements(
        OrderedDict(
            qubit_freq=zefd.QubitFreqTask(
                detune_sweep=make_sweep(-15, 15, step=0.2),
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env["info"])
                    and (pred_qf := info["predict_freq"])
                    and (prev_factor := info.last.get("qfw_factor", md.qf_w / 0.2))
                    and (opt_readout := info.last.get("opt_readout", readout_cfg))
                    and ml.make_cfg(
                        {
                            "modules": {
                                "qub_pulse": {
                                    "type": "pulse",
                                    "waveform": ml.get_waveform(
                                        "qub_flat", {"length": 5}
                                    ),
                                    "ch": md.qub_1_4_ch,
                                    "nqz": 2,
                                    "gain": min(1.0, 1.5 / prev_factor),
                                    "freq": pred_qf,
                                    # "mixer_freq": cur_qf,
                                },
                                "readout": opt_readout,
                            },
                            "relax_delay": 0.1,
                            "reps": 1000,
                            "rounds": 100,
                        }
                    )
                ),
                earlystop_snr=50,
            ),
            lenrabi=zefd.LenRabiTask(
                num_expts=101,
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env["info"])
                    and (cur_qf := info.get("qubit_freq"))
                    and (prev_t1 := info.last.get("smooth_t1", md.t1))
                    and (prev_pi_len := info.last.get("pi_length", pi_len))
                    and (prev_pi_pd := info.last.get("smooth_pi_product", pi_product))
                    and (opt_readout := info.last.get("opt_readout", readout_cfg))
                    and ml.make_cfg(
                        {
                            "modules": {
                                "rabi_pulse": {
                                    **pi_pulse,
                                    "nqz": 2,
                                    "freq": cur_qf,
                                    "gain": min(1.0, prev_pi_pd / (1.5 * pi_len)),
                                    # "mixer_freq": cur_qf,
                                },
                                "readout": opt_readout,
                            },
                            "relax_delay": 3 * prev_t1,
                            "reps": 1000,
                            "rounds": 10,
                            "sweep_range": (0.05, max(5 * prev_pi_len, 0.5)),
                        }
                    )
                ),
                earlystop_snr=30,
            ),
            ro_opt=zefd.RO_OptTask(
                freq_expts=10,
                gain_expts=10,
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env["info"])
                    and (prev_t1 := info.last.get("smooth_t1", md.t1))
                    and (prev_best_freq := info.last.get("best_ro_freq", readout_freq))
                    and (prev_best_gain := info.last.get("best_ro_gain", readout_gain))
                    and (cur_pi_pulse := info.get("pi_pulse"))
                    and ml.make_cfg(
                        {
                            "modules": {
                                "pi_pulse": cur_pi_pulse,
                                "readout": readout_cfg,
                            },
                            "relax_delay": 3 * prev_t1,
                            "reps": 1000,
                            "rounds": 10,
                            "freq_range": (
                                prev_best_freq - 0.2 * md.rf_w,
                                prev_best_freq + 0.2 * md.rf_w,
                            ),
                            "gain_range": (
                                max(0.0, prev_best_gain - 0.05),
                                min(1.0, prev_best_gain + 0.05),
                            ),
                        }
                    )
                ),
            ),
            t1=zefd.T1Task(
                num_expts=101,
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env["info"])
                    and (prev_t1 := info.last.get("smooth_t1", md.t1))
                    and (cur_pi_pulse := info.get("pi_pulse"))
                    and (opt_readout := info.last.get("opt_readout", readout_cfg))
                    and ml.make_cfg(
                        {
                            "modules": {
                                "pi_pulse": cur_pi_pulse,
                                "readout": opt_readout,
                            },
                            "relax_delay": max(1.0, 3 * prev_t1),
                            "reps": 1000,
                            "rounds": 10,
                            "sweep_range": (0.5, max(1.0, 5 * prev_t1)),
                        }
                    )
                ),
                earlystop_snr=20,
            ),
            t2ramsey=zefd.T2RamseyTask(
                num_expts=121,
                detune_ratio=0.05,
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env["info"])
                    and (cur_t1 := info.get("smooth_t1", md.t1))
                    and (prev_t2r := info.last.get("smooth_t2r", md.t2r))
                    and (cur_pi2_pulse := info.get("pi2_pulse"))
                    and (opt_readout := info.last.get("opt_readout", readout_cfg))
                    and ml.make_cfg(
                        {
                            "modules": {
                                "pi2_pulse": cur_pi2_pulse,
                                "readout": opt_readout,
                            },
                            "relax_delay": max(1.0, 3 * cur_t1),
                            "reps": 1000,
                            "rounds": 10,
                            "sweep_range": (0, 2.5 * prev_t2r),
                        }
                    )
                ),
                earlystop_snr=20,
            ),
            t2echo=zefd.T2EchoTask(
                num_expts=121,
                detune_ratio=0.05,
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env["info"])
                    and (cur_t1 := info.get("smooth_t1", md.t1))
                    and (prev_t2e := info.last.get("smooth_t2e", md.t2e))
                    and (cur_pi_pulse := info.get("pi_pulse"))
                    and (cur_pi2_pulse := info.get("pi2_pulse"))
                    and (opt_readout := info.last.get("opt_readout", readout_cfg))
                    and ml.make_cfg(
                        {
                            "modules": {
                                "pi_pulse": cur_pi_pulse,
                                "pi2_pulse": cur_pi2_pulse,
                                "readout": opt_readout,
                            },
                            "relax_delay": max(1.0, 3 * cur_t1),
                            "reps": 1000,
                            "rounds": 10,
                            "sweep_range": (0, 2.5 * prev_t2e),
                        }
                    )
                ),
                earlystop_snr=20,
            ),
        )
    )
    .record_animation(os.path.join(em.flx_dir, f"{filename}.mp4"))
)
_ = executor.run(
    dev_cfg={
        "flux_yoko": {
            **flux_yoko.get_info(),
            "label": "flux_dev",
        }
    },
    predictor=preditor.clone(),
    env_dict={"soccfg": soccfg, "soc": soc, "ml": ml.clone()},
    retry_time=0,
)
```

```python
filepath = Path(database_path, f"{filename}@{em.label}")

snapshot_dir = filepath.parent / f"{filepath.name}_snapshot"
snapshot_dir.mkdir(parents=True, exist_ok=True)

(snapshot_dir / "measure_code.py").write_text(measure_code)
dump_device_info(str(snapshot_dir / "device_info.json"))
ml.clone(dst_path=snapshot_dir / "module_cfg.yaml")
md.clone(dst_path=snapshot_dir / "meta_info.json")

executor.save(
    filepath=str(filepath),
    comment=f"Autofluxdep snapeshot: {snapshot_dir}",
)
del executor
```

```python

```
