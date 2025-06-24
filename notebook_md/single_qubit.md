---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3
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

# Import Module

```python
%load_ext autoreload
import os

import numpy as np

%autoreload 2
import zcu_tools.notebook.single_qubit as zf  # noqa: E402
import zcu_tools.notebook.schedule.v2 as zs  # noqa: E402
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.program.v2 import visualize_pulse

# ruff: noqa: I001
from zcu_tools import (  # noqa: E402
    ModuleLibrary,
    create_datafolder,
    make_cfg,
    make_sweep,
    save_data,
    make_comment,
)
```

```python
import zcu_tools.config as zc

# zc.config.DATA_DRY_RUN = True  # don't save data
# zc.config.YOKO_DRY_RUN = True  # don't run yoko
```

# Create data folder

```python
chip_name = r"Test"

data_host = None
# data_host = "021-zcu216"

database_path = create_datafolder(os.path.join(os.getcwd(), ".."), prefix=chip_name)
```

# Connect to zcu216

```python
from zcu_tools.remote import make_proxy
from zcu_tools.program.base import MyProgram  # noqa: F401
from zcu_tools.tools import get_ip_address  # noqa: F401

# zc.config.LOCAL_IP = get_ip_address("Tailscale")
zc.config.LOCAL_IP = "192.168.10.232"
zc.config.LOCAL_PORT = 8887

soc, soccfg, rm_prog = make_proxy("192.168.10.113", 8887, proxy_prog=True)
MyProgram.init_proxy(rm_prog, test=True)
print(soccfg)
```

```python
# from myqick import QickSoc  # noqa: E402
# from zcu_tools.tools import get_bitfile

# soc = QickSoc(bitfile=get_bitfile("v2"))
# soccfg = soc
# print(soc)
```

# Predefine parameters

```python
timeFly = 0.45
res_ch = 0
qub_ch = 2
reset_ch = 14
reset_ch2 = 2

ro_ch = 0
```

```python
# DefaultCfg.dump("Q12_2D[2]-Q4_default_cfg_-0.42mA_0613.yaml")
# DefaultCfg.load("Q12_2D[2]-Q4_default_cfg_-0.42mA_0612.yaml")
```

# Initialize the flux

```python
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.yoko import YOKOGS200

import pyvisa

flux_dev = YOKOGS200(
    VISAaddress="USB0::0x0B21::0x0039::91T810992::INSTR", rm=pyvisa.ResourceManager()
)
GlobalDeviceManager.register_device("flux_yoko", flux_dev)
cur_A = flux_dev.get_current()
cur_A
```

```python
# cur_A = 1.0e-3
flux_dev.set_current(current=cur_A)
```

# Lookback

```python
timeFly = 0.4
```

```python
exp_cfg = {
    "readout": {
        "type": "base",
        "pulse_cfg": {
            "ch": res_ch,
            "nqz": 2,
            "style": "flat_top",
            "length": 0.65,  # us
            "raise_pulse": {"style": "cosine", "length": 0.1},
            "gain": 1.0,
            "freq": 5795.62,
            # "freq": r_f,
        },
        "ro_cfg": {
            "ro_length": 1.5,  # us
            "trig_offset": 0.4,  # us
        },
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, soft_avgs=5000)

Ts, signals = zs.measure_lookback(soc, soccfg, cfg, progress=True)
```

```python
predict_offset = zf.lookback_show(
    Ts, signals, ratio=0.3, smooth=1.0, ro_cfg=cfg["readout"]["ro_cfg"]
)
predict_offset
```

```python
timeFly = float(predict_offset)
timeFly
```

```python
filename = "lookback"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"timeFly = {timeFly}us"),
    tag="Lookback",
    server_ip=data_host,
)
```

# OneTone

```python
res_name = "3D_cavity"
```

```python
ModuleLibrary.register_waveform(
    ro_waveform={
        "style": "flat_top",
        "raise_pulse": {"style": "cosine", "length": 0.1},
        "length": 5.1,  # us
    }
)
```

## Resonator Frequency

```python
exp_cfg = {
    "readout": {
        "type": "base",
        "pulse_cfg": {
            **ModuleLibrary.get_waveform("ro_waveform"),
            "ch": res_ch,
            "nqz": 2,
            "gain": 1.0,
        },
        "ro_cfg": {
            "ro_length": 4.5,  # us
            "trig_offset": timeFly + 0.05,  # us
        },
    },
    "sweep": make_sweep(5750, 5850, 201),
    # "sweep": make_sweep(r_f-10, r_f+10, 101),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=10)

fpts, signals = zs.measure_res_freq(soc, soccfg, cfg)
```

```python
%matplotlib inline
f, kappa = zf.freq_analyze(fpts, signals, asym=True)
f
```

```python
r_f = f
rf_w = kappa
```

```python
filename = f"{res_name}_freq@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"resonator frequency = {r_f}MHz"),
    # comment=make_comment(cfg),
    tag="OneTone/single",
    server_ip=data_host,
)
```

```python
# r_f =  6026.1
r_f
```

## Power Dependence

```python
exp_cfg = {
    "readout": {
        "type": "base",
        "pulse_cfg": {
            **ModuleLibrary.get_waveform("ro_waveform"),
            "ch": res_ch,
            "nqz": 2,
            "gain": 1.0,
        },
        "ro_cfg": {
            "ro_length": 4.5,  # us
            "trig_offset": timeFly + 0.05,  # us
        },
    },
    "sweep": {
        "gain": make_sweep(0.01, 1.0, 51),
        "freq": make_sweep(r_f - 7, r_f + 7, 51),
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=100)

pdrs, fpts, signals2D = zs.measure_res_pdr_dep(
    soc, soccfg, cfg, dynamic_avg=True, gain_ref=0.03
)
```

```python
filename = f"{res_name}_pdr@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Power", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg),
    tag="OneTone/pdr",
    server_ip=data_host,
)
```

## Flux dependence

```python
cur_A = -1.5e-3
1e3 * flux_dev.set_current(cur_A)
```

```python
exp_cfg = {
    "readout": {
        "type": "base",
        "pulse_cfg": {
            **ModuleLibrary.get_waveform("ro_waveform"),
            "ch": res_ch,
            "nqz": 2,
            "gain": 1.0,
        },
        "ro_cfg": {
            "ro_length": 4.5,  # us
            "trig_offset": timeFly + 0.05,  # us
        },
    },
    "sweep": {
        "gain": make_sweep(0.01, 1.0, 51),
        "freq": make_sweep(r_f - 7, r_f + 7, 51),
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=100)

As, fpts, signals2D = zs.measure_res_flux_dep(soc, soccfg, cfg)
```

```python
filename = f"{res_name}_flux"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Current", "unit": "A", "values": As},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg),
    tag="OneTone/flux",
    server_ip=data_host,
)
```

```python
cur_A = 4e-3
1e3 * flux_dev.set_current(cur_A)
```

## Set readout pulse

```python
ModuleLibrary.register_module(
    readout_rf={
        "type": "base",
        "pulse_cfg": {
            **ModuleLibrary.get_waveform("ro_waveform"),
            "ch": res_ch,
            "nqz": 2,
            "freq": r_f,
            "gain": 1.0,
            "length": 5.1,
        },
        "ro_cfg": {
            "ro_length": 4.5,  # us
            "trig_offset": timeFly + 0.05,  # us
        },
        "desc": "lower power readout with exact resonator frequency",
    }
)
```

# TwoTone

```python
qub_name = "Q4"
```

```python
preditor = FluxoniumPredictor(f"../result/{chip_name}/params.json")
```

```python
# ModuleLibrary.set_pulse(
#     probe_qf={
#         "style": "flat_top",
#         "length": 2.0,
#         # "raise_pulse": {"style": "gauss", "length": 0.02, "sigma": 0.003},
#         "raise_pulse": {"style": "cosine", "length": 0.02},
#     },
# )
ModuleLibrary.register_waveform(
    qub_waveform={
        "style": "flat_top",
        "raise_pulse": {"style": "cosine", "length": 0.02},
        "length": 2.0,
    }
)
```

## Twotone Frequency

```python
cur_A = 0.0e-3
1e3 * flux_dev.set_current(cur_A)
```

```python
q_f = preditor.predict_freq(cur_A, transition=(0, 1))
q_f
```

```python
exp_cfg = {
    "readout": "readout_rf",
    "qub_pulse": {
        **ModuleLibrary.get_waveform("qub_waveform"),
        "ch": qub_ch,
        "nqz": 2,
        "gain": 0.01,
        "mixer_freq": q_f,
        # "post_delay": None,
    },
    "sweep": make_sweep(q_f - 50, q_f + 50, 101),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=1000)

fpts, signals = zs.measure_qub_freq(soc, soccfg, cfg)
```

```python
%matplotlib inline
f, kappa = zf.freq_analyze(fpts, signals, max_contrast=True)
f
```

```python
q_f = f
qf_w = kappa
```

```python
filename = f"{qub_name}_freq@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"frequency = {f}MHz"),
    tag="TwoTone/single",
    server_ip=data_host,
)
```

```python
bias = preditor.calculate_bias(cur_A, q_f)
bias * 1e3
```

```python
preditor.update_bias(bias)
```

## Reset

### One Pulse

```python
reset_f = r_f - q_f
reset_f
```

```python
exp_cfg = {
    # "init_pulse": {
    #     **ModuleLibrary.get_waveform("qub_waveform"),
    #     "ch": qub_ch,
    #     "nqz": 2,
    #     "gain": 0.01,
    #     "mixer_freq": q_f,
    # },
    "tested_reset": {
        "type": "pulse",
        "pulse_cfg": {
            **ModuleLibrary.get_waveform("qub_waveform"),
            "ch": reset_ch,
            "nqz": 2,
            "gain": 0.5,
            # "mixer_freq": reset_f,
            "post_delay": 5.0 / rf_w,  # 5 times the resonator linewidth
        },
    },
    "readout": "readout_rf",
    "sweep": make_sweep(reset_f - 150, reset_f + 150, 101),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=10)

fpts, signals = zs.measure_reset_freq(soc, soccfg, cfg, remove_bg=True)
```

```python
f, kappa = zf.freq_analyze(fpts, signals, max_contrast=True)
f
```

```python
reset_f = f
```

```python
filename = f"{qub_name}_freq@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"frequency = {f}MHz"),
    tag="TwoTone/single",
    server_ip=data_host,
)
```

#### Reset Time

```python
exp_cfg = {
    "reset": {
        "type": "pulse",
        "pulse_cfg": {
            **ModuleLibrary.get_waveform("qub_waveform"),
            "ch": reset_ch,
            "nqz": 2,
            "gain": 0.5,
            "freq": reset_f,
            # "mixer_freq": reset_f,
            "post_delay": 5.0 / rf_w,  # 5 times the resonator linewidth
        },
    },
    "qub_pulse": "pi_amp",
    "readout": "readout_rf",
    "sweep": make_sweep(0.03, 5.0, 51),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=100)

Ts, signals = zs.measure_reset_time(soc, soccfg, cfg)
```

```python
filename = f"{qub_name}_reset_time@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Length", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg),
    tag="TwoTone/reset/time",
    server_ip=data_host,
)
```

#### Set Reset Pulse

```python
ModuleLibrary.register_module(
    reset_10={
        **cfg["tested_reset"],
        "length": 5.0,  # us
        "desc": "Reset with one pulse from 1 to 0",
    },
)
```

### Two pulse

```python
reset1_trans = (1, 2)
reset_f1 = preditor.predict_freq(cur_A, transition=reset1_trans)
reset_f1
```

```python
reset2_trans = (2, 0)
reset_f2 = abs(r_f + preditor.predict_freq(cur_A, transition=reset2_trans))
reset_f2
```

```python
exp_cfg = {
    "init_pulse": {
        **ModuleLibrary.get_waveform("qub_waveform"),
        "ch": qub_ch,
        "nqz": 2,
        "gain": 0.01,
        "mixer_freq": q_f,
    },
    "tested_reset": {
        "type": "two_pulse",
        "pulse_cfg1": {
            **ModuleLibrary.get_waveform("qub_waveform"),
            "ch": reset_ch,
            "nqz": 2,
            "gain": 0.5,
            # "mixer_freq": reset_f1,
            "post_delay": None,
        },
        "pulse_cfg2": {
            **ModuleLibrary.get_waveform("qub_waveform"),
            "ch": reset_ch2,
            "nqz": 1,
            "gain": 0.5,
            # "mixer_freq": reset_f2,
            "post_delay": 5.0 / rf_w,  # 5 times the resonator linewidth
        },
    },
    "readout": "readout_rf",
    "sweep": {
        "freq1": make_sweep(reset_f1 - 100, reset_f1 + 100, 51),
        "freq2": make_sweep(reset_f2 - 100, reset_f2 + 100, 51),
        # "freq1": make_sweep(3170, 3250, 51),
        # "freq2": make_sweep(1920, 2000, 51),
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=100, soft_avgs=100)

fpts1, fpts2, signals = zs.measure_mux_reset_freq(soc, soccfg, cfg)
```

```python
%matplotlib inline
xlabal = f"|{reset1_trans[0]}, 0> - |{reset1_trans[1]}, 0>"
ylabal = f"|{reset2_trans[0]}, 0> - |{reset2_trans[1]}, 1>"
f1, f2 = zf.mux_reset_analyze(
    fpts1, fpts2, signals, xlabel=xlabal, ylabel=ylabal, smooth=0.5
)
f1, f2
```

```python
reset_f1 = f1
reset_f2 = f2
```

```python
filename = f"{qub_name}_mux_reset_freq@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": xlabal, "unit": "Hz", "values": fpts1 * 1e6},
    y_info={"name": ylabal, "unit": "Hz", "values": fpts2 * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
    comment=make_comment(cfg, f"frequency = ({reset_f1:.1f}, {reset_f2:.1f})MHz"),
    tag="TwoTone/mux_reset/freq",
    server_ip=data_host,
)
```

#### Set Mux Reset Pulse

```python
mux_reset_len = 10.0
ModuleLibrary.register_module(
    reset_120={
        "type": "two_pulse",
        "pulse_cfg1": {
            **cfg["tested_reset"]["pulse_cfg1"],
            "freq": reset_f1,
            "length": mux_reset_len,
        },
        "pulse_cfg2": {
            **cfg["tested_reset"]["pulse_cfg2"],
            "freq": reset_f2,
            "length": mux_reset_len,
        },
        "desc": f"Reset with two pulse: {reset1_trans} and {reset2_trans}",
    },
)
```

#### Reset Gain

```python
exp_cfg = {
    "init_pulse": {
        **ModuleLibrary.get_waveform("qub_waveform"),
        "ch": qub_ch,
        "nqz": 2,
        "gain": 0.01,
        "mixer_freq": q_f,
    },
    "tested_reset": "reset_120",
    "readout": "readout_rf",
    "sweep": {
        "gain1": make_sweep(0.0, 1.0, 51),
        "gain2": make_sweep(0.0, 1.0, 51),
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=100, soft_avgs=100)

pdrs1, pdrs2, signals2D = zs.measure_mux_reset_pdr(soc, soccfg, cfg)
```

```python
%matplotlib inline
xlabal = f"|{reset1_trans[0]}, 0> - |{reset1_trans[1]}, 0>"
ylabal = f"|{reset2_trans[0]}, 0> - |{reset2_trans[1]}, 1>"
gain1, gain2 = zf.mux_reset_pdr_analyze(
    pdrs1, pdrs2, signals2D, xlabel=xlabal, ylabel=ylabal
)
```

```python
filename = f"{qub_name}_mux_reset_gain@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": xlabal, "unit": "a.u.", "values": pdrs1},
    y_info={"name": ylabal, "unit": "a.u.", "values": pdrs2},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
    comment=make_comment(cfg, f"best gain = ({gain1:.1f}, {gain2:.1f})"),
    tag="TwoTone/mux_reset/pdr",
    server_ip=data_host,
)
```

```python
ModuleLibrary.update_module(
    "reset_120",
    override_cfg={
        "pulse_cfg1": {"gain": gain1},
        "pulse_cfg2": {"gain": gain2},
    },
)
```

#### Reset Time

```python
exp_cfg = {
    "reset": "reset_120",
    "qub_pulse": "pi_amp",
    "readout": "readout_rf",
    "sweep": make_sweep(0.05, 1.0, 31),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=100)

Ts, signals = zs.measure_mux_reset_time(soc, soccfg, cfg)
```

```python
filename = f"{qub_name}_mux_reset_time@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Length", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg),
    tag="TwoTone/mux_reset/time",
    server_ip=data_host,
)
```

```python
mux_reset_len = 1.0  # us
ModuleLibrary.update_module(
    "reset_120",
    override_cfg={
        "pulse_cfg1": {"length": mux_reset_len},
        "pulse_cfg2": {"length": mux_reset_len},
    },
)
```

### Check Reset

```python
exp_cfg = {
    "reset": "reset_120",
    "init_pulse": "pi_amp",
    "tested_reset": "reset_120",
    "readout": "readout_rf",
    "sweep": make_sweep(0.0, 1.0, 51),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=10)

# zs.visualize_reset_amprabi(soccfg, cfg, time_fly=timeFly)
pdrs, signals = zs.measure_reset_amprabi(soc, soccfg, cfg)
```

```python
filename = f"{qub_name}_check_reset@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Amplitude", "unit": "a.u.", "values": pdrs},
    y_info={"name": "GE", "unit": "None", "values": np.array([0, 1])},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg),
    tag="TwoTone/check_reset",
    server_ip=data_host,
)
```

## Power dependence

```python
exp_cfg = {
    "qub_pulse": {
        **ModuleLibrary.get_waveform("qub_waveform"),
        "ch": qub_ch,
        "nqz": 2,
        "length": 5,  # us
    },
    "readout": "readout_rf",
    "sweep": {
        "gain": make_sweep(0.05, 1.0, 30),
        "freq": make_sweep(1700, 2000, 30),
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=50, soft_avgs=50)

fpts, pdrs, signals2D = zs.measure_qub_pdr_dep(soc, soccfg, cfg)
```

```python
filename = f"{qub_name}_pdr@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Power", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg),
    tag="TwoTone/pdr",
    server_ip=data_host,
)
```

## Flux Dependence

```python
cur_A = 3.5e-3
1e3 * flux_dev.set_current(cur_A)
```

```python
exp_cfg = {
    "qub_pulse": {
        **ModuleLibrary.get_waveform("qub_waveform"),
        "ch": qub_ch,
        "nqz": 2,
        "length": 5,  # us
    },
    "readout": "readout_rf",
    "relax_delay": 0.0,  # us
    "sweep": {
        "flux": make_sweep(3.5e-3, -0.5e-3, 301),
        "freq": make_sweep(3000, 5000, 501),
    },
}
cfg = make_cfg(exp_cfg, reps=10000, soft_avgs=10)

As, fpts, signals2D = zs.measure_qub_flux_dep(soc, soccfg, cfg)
```

```python
filename = f"{qub_name}_flux_dep"
save_data(
    filepath=os.path.join(database_path, filename),
    y_info={"name": "Current", "unit": "A", "values": As},
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg),
    tag="TwoTone/flux",
    server_ip=data_host,
)
```

```python
cur_A = 0.0e-3
1e3 * flux_dev.set_current(cur_A)
```

## Dispersive Shift

```python
exp_cfg = {
    "reset": "reset_120",
    "qub_pulse": "pi_amp",
    "readout": "readout_rf",
    "sweep": make_sweep(r_f - 20, r_f + 20, 101),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=10000, soft_avgs=10)

pdrs, signals = zs.measure_reset_amprabi(soc, soccfg, cfg)
```

```python
%matplotlib inline
chi, rf_w = zf.analyze_dispersive(fpts, signals, asym=True)
chi
```

```python
filename = f"{res_name}_dispersive@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Amplitude", "unit": "None", "values": np.array([0, 1])},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"chi = {chi:.3g} MHz, kappa = {rf_w:.3g} MHz"),
    tag="TwoTone/dispersive",
    server_ip=data_host,
)
```

## AC Stark Shift

```python
ac_qub_len = ModuleLibrary.get_module("pi_amp")["length"]  # us
exp_cfg = {
    "reset": "reset_120",
    "stark_pulse1": {
        **ModuleLibrary.get_module("readout_rf")["pulse_cfg"],
        "length": 6.0 / rf_w + ac_qub_len,  # us
        "post_delay": None,
    },
    "stark_pulse2": {
        **ModuleLibrary.get_module("pi_amp"),
        "length": ac_qub_len,  # us
        "t": 5.0 / rf_w,
        "post_delay": 5.0 / rf_w,
    },
    "readout": "readout_rf",
    "sweep": {
        "gain": make_sweep(0.00, 0.35, 101),
        "freq": make_sweep(q_f - 100, q_f + 10, 101),
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=100, soft_avgs=100)

# zs.visualize_ac_stark(soccfg, cfg, time_fly=timeFly)
pdrs, fpts, signals2D = zs.measure_ac_stark(soc, soccfg, cfg)
```

```python
ac_stark_coeff = zf.analyze_ac_stark_shift(
    pdrs, fpts, signals2D, chi, kappa, deg=1, cutoff=None
)
ac_stark_coeff
```

```python
filename = f"{qub_name}_ac_stark@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Power", "unit": "a.u.", "values": pdrs},
    y_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg, f"ac_stark_coeff = {ac_stark_coeff:.3g} MHz"),
    tag="TwoTone/ac_stark",
    server_ip=data_host,
)
```

# Rabi

## Length Rabi

```python
exp_cfg = {
    "qub_pulse": {
        **ModuleLibrary.get_waveform("qub_waveform"),
        "ch": qub_ch,
        "nqz": 2,
        "freq": q_f,
        "gain": 0.3,
        # "gain": pi_gain,
        "length": 5,  # us
        # "mixer_freq": q_f,
    },
    "readout": "readout_rf",
    "relax_delay": 0.0,  # us
    "sweep": make_sweep(0.03, 0.3, 101),
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=20)

# zs.visualize_lenrabi(soccfg, cfg, time_fly=timeFly)
pdrs, fpts, signals2D = zs.measure_lenrabi(soc, soccfg, cfg)
```

```python
%matplotlib inline
pi_len, pi2_len = zf.rabi_analyze(
    Ts, signals, decay=False, max_contrast=True, xlabel="Time (us)"
)
pi_len, pi2_len
```

```python
filename = f"{qub_name}_len_rabi@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Pulse Length", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"pi len = {pi_len}us\npi/2 len = {pi2_len}us"),
    tag="TimeDomain/len_rabi",
    server_ip=data_host,
)
```

### Set Pi Pulse

```python
# pi_len = 1.0
# pi2_len = 0.5
```

```python
ModuleLibrary.register_module(
    pi_len={
        **exp_cfg["qub_pulse"],
        "length": pi_len,
        "desc": "len pi pulse",
    },
    pi2_len={
        **exp_cfg["qub_pulse"],
        "length": pi2_len,
        "desc": "len pi/2 pulse",
    },
)
```

## Amplitude Rabi

```python
pi_gain = ModuleLibrary.get_module("pi_len")["gain"]
max_gain = min(5 * pi_gain, 1.0)
exp_cfg = {
    "reset": "reset_120",
    "qub_pulse": {
        **ModuleLibrary.get_waveform("qub_waveform"),
        "ch": qub_ch,
        "nqz": 2,
        "freq": q_f,
        "length": 5,  # us
        # "length": pi_len,
        # "mixer_freq": q_f,
    },
    "readout": "readout_rf",
    "relax_delay": 0.0,  # us
    "sweep": make_sweep(0.0, max_gain, 51),
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=10)

# zs.visualize_amprabi(soccfg, cfg, time_fly=timeFly)
pdrs, fpts, signals2D = zs.measure_amprabi(soc, soccfg, cfg)
```

```python
%matplotlib inline
pi_gain, pi2_gain = zf.rabi_analyze(
    pdrs, signals, decay=False, max_contrast=True, xlabel="Power (a.u.)"
)
pi_gain = int(pi_gain + 0.5) if pi_gain > 1.0 else pi_gain
pi2_gain = int(pi2_gain + 0.5) if pi2_gain > 1.0 else pi2_gain
pi_gain, pi2_gain
```

```python
filename = f"{qub_name}_amp_rabi@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Amplitude", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"pi gain = {pi_gain}\npi/2 gain = {pi2_gain}"),
    tag="TimeDomain/amp_rabi",
    server_ip=data_host,
)
```

### Set Pi Pulse

```python
# pi_gain = 1.0
# pi2_gain = 0.5
```

```python
ModuleLibrary.register_module(
    pi_amp={
        **cfg["qub_pulse"],
        "gain": pi_gain,
        "desc": "amp pi pulse",
    },
    pi2_amp={
        **cfg["qub_pulse"],
        "gain": pi2_gain,
        "desc": "amp pi/2 pulse",
    },
)
```

# Optimize Readout

## Frequency tuning

```python
exp_cfg = {
    "reset": "reset_120",
    "qub_pulse": "pi_amp",
    "readout": "readout_rf",
    "relax_delay": 0.0,  # us
    "sweep": make_sweep(r_f - 10, r_f + 10, 51),
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=10)

fpts, snrs = zs.qubit.measure_ge_freq_dep(soc, soccfg, cfg)
```

```python
fpt_max = zf.optimize_1d(fpts, snrs, xlabel="Frequency (MHz)")
```

```python
filename = f"{qub_name}_ge_fpt@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "SNR", "unit": "a.u.", "values": snrs},
    comment=make_comment(cfg),
    tag="TwoTone/dispersive/fpt",
    server_ip=data_host,
)
```

## Power tuning

```python
exp_cfg = {
    "reset": "reset_120",
    "qub_pulse": "pi_amp",
    "readout": ModuleLibrary.get_module(
        "readout_rf",
        {
            "pulse_cfg": {
                "freq": fpt_max,
            }
        },
    ),
    "relax_delay": 0.0,  # us
    "sweep": make_sweep(0.01, 1.0, 31),
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=10)

pdrs, snrs = zs.qubit.measure_ge_pdr_dep(soc, soccfg, cfg)
```

```python
pdr_max = zf.optimize_1d(pdrs, snrs, xlabel="Probe Power (a.u)")
```

```python
filename = f"{qub_name}_ge_pdr@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Probe Power (a.u)", "unit": "s", "values": pdrs},
    z_info={"name": "SNR", "unit": "a.u.", "values": snrs},
    comment=make_comment(cfg),
    tag="TwoTone/dispersive/pdr",
    server_ip=data_host,
)
```

## Readout Length tuning

```python
exp_cfg = {
    "reset": "reset_120",
    "qub_pulse": "pi_amp",
    "readout": ModuleLibrary.get_module(
        "readout_rf",
        {
            "pulse_cfg": {
                "freq": fpt_max,
                "gain": pdr_max,
            }
        },
    ),
    "relax_delay": 0.0,  # us
    "sweep": make_sweep(0.1, 15.0, 31),
}
cfg = make_cfg(exp_cfg, reps=10000, soft_avgs=1)

ro_lens, snrs = zs.qubit.measure_ge_ro_dep(soc, soccfg, cfg)
```

```python
ro_max = zf.optimize_ro_len(ro_lens, snrs, t0=30.0)
ro_max
```

```python
filename = f"{qub_name}_ge_len@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Readout length", "unit": "s", "values": ro_lens * 1e-6},
    z_info={"name": "SNR", "unit": "a.u.", "values": snrs},
    comment=make_comment(cfg),
    tag="TwoTone/dispersive/ro_len",
    server_ip=data_host,
)
```

```python
# ro_max = 3.0
ModuleLibrary.register_module(
    readout_dpm=ModuleLibrary.get_module(
        "readout_rf",
        {
            "pulse_cfg": {
                "freq": fpt_max,
                "gain": pdr_max,
                "length": ro_max + 0.2,
            },
            "ro_cfg": {
                "ro_length": ro_max,
            },
            "desc": "Readout with largest dispersive shift",
        },
    )
)
```

# T1 & T2

## T2Ramsey

```python
orig_qf = ModuleLibrary.get_module("pi2_amp")["freq"]
orig_qf
```

```python
exp_cfg = {
    "reset": "reset_120",
    "pi2_pulse": "pi2_amp",
    "readout": "readout_dpm",
    "relax_delay": 0.0,  # us
    "sweep": make_sweep(0, 20.0, 101),  # us
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=10)

activate_detune = 10.0 / (cfg["sweep"]["stop"] - cfg["sweep"]["start"])
print(f"activate_detune: {activate_detune:.2f}")
Ts, signals = zs.measure_t2ramsey(soc, soccfg, cfg, detune=activate_detune)
```

```python
%matplotlib inline
t2r, detune, _, _ = zf.T2fringe_analyze(Ts, signals, max_contrast=True)
print(f"real detune: {(detune - activate_detune) * 1e3:.1f}kHz")
```

```python
filename = f"{qub_name}_t2ramsey@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"detune = {detune}MHz\nt2r = {t2r}us"),
    tag="TimeDomain/t2ramsey",
    server_ip=data_host,
)
```

```python
q_f = orig_qf + activate_detune - detune
q_f
```

## T1

```python
exp_cfg = {
    "reset": "reset_120",
    "pi_pulse": "pi_amp",
    "readout": "readout_dpm",
    "relax_delay": 0.0,  # us
    "sweep": make_sweep(0.0, 50, 51),
    # "sweep": make_sweep(0.01, 5 * t1, 51),
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=10)

Ts, signals = zs.measure_t1(soc, soccfg, cfg)
```

```python
%matplotlib inline
start = 1
t1, _ = zf.T1_analyze(Ts[start:], signals[start:], max_contrast=True, dual_exp=False)
t1
```

```python
filename = f"{qub_name}_t1@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"t1 = {t1}us"),
    tag="TimeDomain/t1",
    server_ip=data_host,
)
```

## T2Echo

```python
exp_cfg = {
    "dac": {
        "res_pulse": "readout_dpm",
        "pi_pulse": "pi_amp",
        "pi2_pulse": "pi2_amp",
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
        "reset": "mux_dual_pulse",
        "reset_pulse1": "mux_reset1",
        "reset_pulse2": "mux_reset2",
    },
    "adc": {
        "relax_delay": 0.5,  # us
        # "relax_delay": 3*t1,  # us
    },
}
```

```python
exp_cfg = {
    "reset": "reset_120",
    "pi_pulse": "pi_amp",
    "pi2_pulse": "pi2_amp",
    "readout": "readout_dpm",
    "relax_delay": 0.0,  # us
    "sweep": make_sweep(0.0, 1.5 * t2r, 101),
    # "sweep": make_sweep(0.0, 1.5 * t2e, 101),
    # "sweep": make_sweep(0.01, 5 * t1, 51),
}
cfg = make_cfg(exp_cfg, reps=1000, soft_avgs=10)

activate_detune = 5.0 / (cfg["sweep"]["stop"] - cfg["sweep"]["start"])
print(f"activate_detune: {activate_detune:.2f}")
Ts, signals = zs.measure_t2echo(soc, soccfg, cfg, detune=activate_detune)
```

```python
%matplotlib inline
t2e, detune, _, _ = zf.T2fringe_analyze(Ts, signals, max_contrast=True)
# t2e, _ = zf.T2decay_analyze(Ts, signals, max_contrast=True)
```

```python
filename = f"{qub_name}_t2echo@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"t2echo = {t2e}us"),
    tag="TimeDomain/t2echo",
    server_ip=data_host,
)
```

# Single shot

## Ground state & Excited state

```python
exp_cfg = {
    "dac": {
        # "res_pulse": "readout_rf",
        "res_pulse": {
            **ModuleLibrary.get_pulse("readout_dpm"),
            # "length": 0.2 * t1 + timeFly + 0.1,
            # "ro_length": 0.2 * t1,
        },
        "qub_pulse": "pi_amp",
        # "qub_pulse": "reset_red",
        # "qub_pulse": {
        #     **DefaultCfg.get_pulse("pi_amp"),
        #     # "gain": 0.0,
        # },
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
        "reset": "two_pulse",
        "reset_pulse1": "mux_reset1",
        "reset_pulse2": "mux_reset2",
    },
    "adc": {
        # "relax_delay": 5 * t1,  # us
        "relax_delay": 0.0,  # us
    },
}
exp_cfg["dac"]["res_pulse"]["ro_length"]
```

```python
exp_cfg = {
    "reset": "reset_120",
    "qub_pulse": "pi_amp",
    "readout": ModuleLibrary.get_module(
        "readout_rf",
        {
            "pulse_cfg": {
                "length": 0.2 * t1 + timeFly + 0.1,
            },
            "ro_cfg": {
                "ro_length": 0.2 * t1,
            },
        },
    ),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, shots=1000000)
print("readout length: ", cfg["readout"]["ro_cfg"]["ro_length"])

signals = zs.measure_singleshot(soc, soccfg, cfg)
```

```python
%matplotlib inline
fid, _, _, pops = zf.singleshot_ge_analysis(signals, backend="pca")
print(f"Optimal fidelity after rotation = {fid:.1%}")
```

```python
n_gg, n_ge, n_eg, n_ee = pops
n_g = n_gg
n_e = n_ge
if n_e > n_g:
    n_g, n_e = n_e, n_g
n_g, n_e
```

```python
eff_T, err_T = zf.effective_temperature(population=[(n_g, 0.0), (n_e, q_f)], plot=False)
eff_T, err_T
```

```python
filename = f"{qub_name}_singleshot_ge@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "shot", "unit": "point", "values": np.arange(cfg["shots"])},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    y_info={"name": "ge", "unit": "", "values": np.array([0, 1])},
    comment=make_comment(
        cfg, f"fide: {fid:.1%}, (n_g, n_e): ({n_g:.1%}, {n_e:.1%}), eff_T: {eff_T:.1f}"
    ),
    tag="SingleShot/ge",
    server_ip=data_host,
)
```

```python

```
