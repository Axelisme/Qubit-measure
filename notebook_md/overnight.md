---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

```python
%load_ext autoreload
from pprint import pprint
from pathlib import Path
import json
from datetime import datetime

import numpy as np

%autoreload 2
import zcu_tools.experiment.v2.overnight as zeo
from pydantic import TypeAdapter
from zcu_tools.experiment.cfg_assembler import CfgEnv, make_cfg
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.notebook.plotting import NotebookPlotHost
from zcu_tools.plotting.plots import Plots
from zcu_tools.resources.context import ContextManager
from zcu_tools.datafile import create_datafolder
from zcu_tools.experiment.utils import make_sweep
from zcu_tools.notebook.utils import reconnect_devices, dump_device_info
```

```python
chip_name = "Q12_2D[6]"
res_name = "R1"
qub_name = "Q1_fs6881"

result_dir = f"../result/{chip_name}/{qub_name}"

database_path = create_datafolder(
    str(Path.cwd().parent), name=str(Path(chip_name, qub_name))
)
em = ContextManager(f"{result_dir}/exps")
ml, md = em.use_flux(label="0303_1.800mA", readonly=True)
```

# Connect ZCU216

```python
from zcu_tools.qick_remote import make_soc_proxy

soc, soccfg = make_soc_proxy("192.168.10.179", 8887)
print(soccfg)
```

# Connect Instruments

```python
from zcu_tools.device import DeviceManager, DeviceInfo

dev_info_path = f"{result_dir}/device_info.json"

with open(dev_info_path, "r") as f:
    dev_info = {name: TypeAdapter(DeviceInfo).validate_python(info) for name, info in json.load(f).items()}
pprint(dev_info)

device_manager = DeviceManager()
resource_manager = reconnect_devices(dev_info, device_manager)
device_manager.setup_devices(dev_info, progress=True)
env = CfgEnv(md=md, ml=ml, device_manager=device_manager)
```

# Start Measurement

```python
%matplotlib widget
filename = f"{qub_name}_overnight"

# snapshot of execution code
measure_code: str = In[-1]  # noqa: F821 # type: ignore

executor = zeo.OvernightExecutor(num_times=300, interval=120).add_measurements(
    dict(
        mist_g=zeo.singleshot.MistTask(
            make_cfg(
                {
                    "modules": {
                        # "reset": "reset_10",
                        "probe_pulse": {
                            "waveform": ml.get_waveform(
                                "mist_waveform",
                                {"length": 5.0 / (2 * np.pi * md.rf_w) + 0.3},
                            ),
                            "ch": md.res_ch,
                            "nqz": 2,
                            "freq": md.readout_f,
                            "post_delay": 10 / (2 * np.pi * md.rf_w),
                        },
                        "readout": "readout_dpm",
                    },
                    "sweep": {"gain": make_sweep(0.0, 0.22, 101)},
                    "relax_delay": 50.5,  # us
                },
                zeo.singleshot.MistCfg,
                env,
                overrides={"reps": 3000, "rounds": 1},
            ),
            md.g_center,
            md.e_center,
            md.ge_radius,
        ),
        mist_e=zeo.singleshot.MistTask(
            make_cfg(
                {
                    "modules": {
                        # "reset": "reset_10",
                        "init_pulse": "pi_amp",
                        "probe_pulse": {
                            "waveform": ml.get_waveform(
                                "mist_waveform",
                                {"length": 5.0 / (2 * np.pi * md.rf_w) + 0.3},
                            ),
                            "ch": md.res_ch,
                            "nqz": 2,
                            "freq": md.readout_f,
                            "post_delay": 10 / (2 * np.pi * md.rf_w),
                        },
                        "readout": "readout_dpm",
                    },
                    "sweep": {
                        "gain": make_sweep(0.0, 0.22, 101),
                    },
                    "relax_delay": 50.5,  # us
                },
                zeo.singleshot.MistCfg,
                env,
                overrides={"reps": 3000, "rounds": 1},
            ),
            md.g_center,
            md.e_center,
            md.ge_radius,
        ),
        mist_steady=zeo.singleshot.MistTask(
            make_cfg(
                {
                    "modules": {
                        "probe_pulse": {
                            "waveform": ml.get_waveform(
                                "mist_waveform",
                                {"length": 5.0 / (2 * np.pi * md.rf_w) + 50.0},
                            ),
                            "ch": md.res_ch,
                            "nqz": 2,
                            "freq": md.readout_f,
                            "post_delay": 10 / (2 * np.pi * md.rf_w),
                        },
                        "readout": "readout_dpm",
                    },
                    "sweep": {
                        "gain": make_sweep(0.0, 0.22, 101),
                    },
                    "relax_delay": 50.5,  # us
                },
                zeo.singleshot.MistCfg,
                env,
                overrides={"reps": 3000, "rounds": 1},
            ),
            md.g_center,
            md.e_center,
            md.ge_radius,
        ),
    )
)
run_plots = Plots(NotebookPlotHost())
run_context = RunContext(
    soc=soc, soccfg=soccfg, plots=run_plots,
    devices=device_manager.get_all_devices(), cancel_signal=StopSignal(),
)
try:
    run_results = executor.run(context=run_context, fail_retry=3)
finally:
    run_figures = run_plots.finish()
```

Keep `run_figures` to inspect or save the native Matplotlib figures. Call `run_plots.release()` when the widget presentation is no longer needed. Releasing it does not destroy the retained figures.

```python
filepath = Path(database_path, f"{filename}@{em.label}")

snapshot_dir = filepath.parent / f"{filepath.name}_snapshot"
snapshot_dir.mkdir(parents=True, exist_ok=True)

(snapshot_dir / "measure_code.py").write_text(measure_code)
dump_device_info(snapshot_dir / "device_info.json", device_manager)
ml.clone(dst_path=snapshot_dir / "module_cfg.yaml")
md.clone(dst_path=snapshot_dir / "meta_info.json")

executor.save(
    filepath=str(filepath),
    comment=datetime.now().strftime("Overnight run at %Y-%m-%d %H:%M:%S"),
)
del executor
```

```python

```

```python

```
