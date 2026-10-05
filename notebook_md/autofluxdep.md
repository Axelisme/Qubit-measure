---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
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
from pydantic import TypeAdapter

%autoreload 2
import zcu_tools.experiment.v2.autofluxdep as zefd
from zcu_tools.experiment.cfg_assembler import CfgEnv, make_cfg
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.notebook.plotting import NotebookPlotHost
from zcu_tools.plotting.plots import Plots
import zcu_tools.program.v2 as zp
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.resources.context import ContextManager
from zcu_tools.datafile import create_datafolder
from zcu_tools.experiment.utils import make_sweep
from zcu_tools.notebook.utils import (
    reconnect_devices,
    dump_device_info,
    gc_collect,
)
```

```python
chip_name = "Q5_2D"
res_name = "R1"
qub_name = "Q1"

result_dir = os.path.join("..", "result", chip_name, qub_name)
database_path = create_datafolder(
    database_dir=os.path.join("..", "Database"),
    name=os.path.join(chip_name, qub_name),
)

em = ContextManager(os.path.join(result_dir, "exps"))
ml, md = em.use_flux(label="051115_2.000mA", readonly=True)
```

# Connect ZCU216

```python
from zcu_tools.qick_remote import make_soc_proxy

soc, soccfg = make_soc_proxy("192.168.10.179", 8887)
print(soccfg)
```

```python
soc.get_sample_rates()
# soc.valid_sample_rates(tiletype='dac', tile=2)
```

# Connect Instruments

```python
from zcu_tools.device import DeviceManager, DeviceInfo
from zcu_tools.device.yoko import YOKOGS200

dev_info_path = os.path.join(em.flux_dir, "device_info.json")

with open(dev_info_path, "r") as f:
    info_json = json.load(f)
    dev_info = {
        k: TypeAdapter(DeviceInfo).validate_python(v) for k, v in info_json.items()
    }
pprint(dev_info)

device_manager = DeviceManager()
resource_manager = reconnect_devices(dev_info, device_manager)

flux_yoko = cast(YOKOGS200, device_manager.get_device("flux_yoko"))

device_manager.setup_devices(dev_info, progress=True)
```

# Initial Tools

```python
preditor = FluxoniumPredictor.from_file(os.path.join(result_dir, "params.json"))
preditor.flux_half = md.flx_half
preditor.flux_period = 2 * abs(md.flx_int - md.flx_half)
# preditor.update_bias(md.flux_bias)
```

# Start Measurement

```python
flux_yoko.set_current(2e-3)  # Set to initial flux bias
```

```python
gc_collect()
```

```python
%matplotlib widget
flux_values = np.linspace(2e-3, -0.2e-3, 101)

filename = f"{qub_name}_autofluxdep"

# snapshot of execution code
measure_code: str = In[-1]  # noqa: F821 # type: ignore

pi_pulse = ml.get_module("pi_amp", type=zp.PulseCfg)
pi_len = float(pi_pulse.waveform.length)
pi_product = pi_len * float(pi_pulse.gain)

readout_cfg = ml.get_module("readout_dpm", type=zp.PulseReadoutCfg)
readout_freq = float(readout_cfg.pulse_cfg.freq)
readout_gain = float(readout_cfg.pulse_cfg.gain)


executor = (
    zefd.FluxDepExecutor(flux_values=flux_values)
    .add_measurements(
        OrderedDict(
            qubit_freq=zefd.QubitFreqTask(
                detune_sweep=make_sweep(-20, 50, step=0.5),
                cfg_maker=lambda ctx, ml: (
                    (info := ctx.env.info)
                    and (pred_qf := info.predict_freq)
                    and (prev_factor := info.last_or("qfw_factor", md.qf_w / 0.05))
                    and (opt_readout := info.last_or("opt_readout", readout_cfg))
                    and make_cfg(
                        {
                            "modules": {
                                "qub_pulse": {
                                    "type": "pulse",
                                    "waveform": ml.get_waveform(
                                        "qub_flat", {"length": 0.1}
                                    ),
                                    "ch": md.qub_4_5_ch,
                                    "nqz": 2,
                                    "gain": min(1.0, 6.5 / prev_factor),
                                    # "gain": 0.05,
                                    "freq": pred_qf,
                                    # "freq": 5135,
                                    # "mixer_freq": cur_qf,
                                },
                                "readout": opt_readout,
                            },
                            "relax_delay": 0.5,
                            "reps": 1000,
                            "rounds": 100,
                        },
                        zefd.QubitFreqCfgTemplate,
                        CfgEnv(md=md, ml=ml, device_manager=device_manager),
                    )
                ),
                earlystop_snr=50,
            ),
            # lenrabi=zefd.LenRabiTask(
            #     num_expts=101,
            #     cfg_maker=lambda ctx, ml: (
            #         (info := ctx.env.info)
            #         and (cur_qf := info.require("qubit_freq"))
            #         and (prev_t1 := info.last_or("smooth_t1", md.t1))
            #         and (prev_pi_len := info.last_or("pi_length", pi_len))
            #         and (prev_pi_pd := info.last_or("smooth_pi_product", pi_product))
            #         and (opt_readout := info.last_or("opt_readout", readout_cfg))
            #         and make_cfg(
            #             {
            #                 "modules": {
            #                     "rabi_pulse": pi_pulse.with_updates(
            #                         nqz=2,
            #                         freq=cur_qf,
            #                         gain=min(1.0, prev_pi_pd / (1.5 * pi_len)),
            #                         # mixer_freq=cur_qf,
            #                     ),
            #                     "readout": opt_readout,
            #                 },
            #                 "relax_delay": 3 * prev_t1,
            #                 "reps": 1000,
            #                 "rounds": 10,
            #                 "sweep_range": (0.05, max(5 * prev_pi_len, 0.5)),
            #             },
            #             zefd.LenRabiCfgTemplate,
            #             CfgEnv(md=md, ml=ml, device_manager=device_manager),
            #         )
            #     ),
            #     earlystop_snr=30,
            # ),
            # ro_opt=zefd.RO_OptTask(
            #     freq_expts=10,
            #     gain_expts=10,
            #     cfg_maker=lambda ctx, ml: (
            #         (info := ctx.env.info)
            #         and (prev_t1 := info.last_or("smooth_t1", md.t1))
            #         and (prev_best_freq := info.last_or("best_ro_freq", readout_freq))
            #         and (prev_best_gain := info.last_or("best_ro_gain", readout_gain))
            #         and (cur_pi_pulse := info.require("pi_pulse"))
            #         and make_cfg(
            #             {
            #                 "modules": {
            #                     "pi_pulse": cur_pi_pulse,
            #                     "readout": readout_cfg,
            #                 },
            #                 "relax_delay": 3 * prev_t1,
            #                 "reps": 1000,
            #                 "rounds": 10,
            #                 "freq_range": (
            #                     prev_best_freq - 0.2 * md.rf_w,
            #                     prev_best_freq + 0.2 * md.rf_w,
            #                 ),
            #                 "gain_range": (
            #                     max(0.0, prev_best_gain - 0.05),
            #                     min(1.0, prev_best_gain + 0.05),
            #                 ),
            #             },
            #             zefd.RO_OptCfgTemplate,
            #             CfgEnv(md=md, ml=ml, device_manager=device_manager),
            #         )
            #     ),
            # ),
            # t1=zefd.T1Task(
            #     num_expts=101,
            #     cfg_maker=lambda ctx, ml: (
            #         (info := ctx.env.info)
            #         and (prev_t1 := info.last_or("smooth_t1", md.t1))
            #         and (cur_pi_pulse := info.require("pi_pulse"))
            #         and (opt_readout := info.last_or("opt_readout", readout_cfg))
            #         and make_cfg(
            #             {
            #                 "modules": {
            #                     "pi_pulse": cur_pi_pulse,
            #                     "readout": opt_readout,
            #                 },
            #                 "relax_delay": max(1.0, 3 * prev_t1),
            #                 "reps": 1000,
            #                 "rounds": 10,
            #                 "sweep_range": (0.5, max(1.0, 5 * prev_t1)),
            #             },
            #             zefd.T1CfgTemplate,
            #             CfgEnv(md=md, ml=ml, device_manager=device_manager),
            #         )
            #     ),
            #     earlystop_snr=20,
            # ),
            # t2ramsey=zefd.T2RamseyTask(
            #     num_expts=121,
            #     detune_ratio=0.05,
            #     cfg_maker=lambda ctx, ml: (
            #         (info := ctx.env.info)
            #         and (cur_t1 := (info.current.smooth_t1 if info.current.smooth_t1 is not None else md.t1))
            #         and (prev_t2r := info.last_or("smooth_t2r", md.t2r))
            #         and (cur_pi2_pulse := info.require("pi2_pulse"))
            #         and (opt_readout := info.last_or("opt_readout", readout_cfg))
            #         and make_cfg(
            #             {
            #                 "modules": {
            #                     "pi2_pulse": cur_pi2_pulse,
            #                     "readout": opt_readout,
            #                 },
            #                 "relax_delay": max(1.0, 3 * cur_t1),
            #                 "reps": 1000,
            #                 "rounds": 10,
            #                 "sweep_range": (0, 2.5 * prev_t2r),
            #             },
            #             zefd.T2RamseyCfgTemplate,
            #             CfgEnv(md=md, ml=ml, device_manager=device_manager),
            #         )
            #     ),
            #     earlystop_snr=20,
            # ),
            # t2echo=zefd.T2EchoTask(
            #     num_expts=121,
            #     detune_ratio=0.05,
            #     cfg_maker=lambda ctx, ml: (
            #         (info := ctx.env.info)
            #         and (cur_t1 := (info.current.smooth_t1 if info.current.smooth_t1 is not None else md.t1))
            #         and (prev_t2e := info.last_or("smooth_t2e", md.t2e))
            #         and (cur_pi_pulse := info.require("pi_pulse"))
            #         and (cur_pi2_pulse := info.require("pi2_pulse"))
            #         and (opt_readout := info.last_or("opt_readout", readout_cfg))
            #         and make_cfg(
            #             {
            #                 "modules": {
            #                     "pi_pulse": cur_pi_pulse,
            #                     "pi2_pulse": cur_pi2_pulse,
            #                     "readout": opt_readout,
            #                 },
            #                 "relax_delay": max(1.0, 3 * cur_t1),
            #                 "reps": 1000,
            #                 "rounds": 10,
            #                 "sweep_range": (0, 2.5 * prev_t2e),
            #             },
            #             zefd.T2EchoCfgTemplate,
            #             CfgEnv(md=md, ml=ml, device_manager=device_manager),
            #         )
            #     ),
            #     earlystop_snr=20,
            # ),
        )
    )
    .record_animation(os.path.join(em.flux_dir, f"{filename}.mp4"))
)
run_plots = Plots(NotebookPlotHost())
run_context = RunContext(
    soc=soc, soccfg=soccfg, plots=run_plots,
    devices=device_manager.get_all_devices(), cancel_signal=StopSignal(),
)
try:
    run_results = executor.run(
        dev_cfg={"flux_yoko": flux_yoko.get_info().with_updates(label="flux_dev")},
        predictor=preditor.clone(),
        context=run_context,
        ml=ml.clone(),
        retry_time=0,
    )
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
    comment=f"Autofluxdep snapeshot: {snapshot_dir}",
)
del executor
```

```python

```
