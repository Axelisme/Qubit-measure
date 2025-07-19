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

# Import Module

```python
%load_ext autoreload
import os

import numpy as np

%autoreload 2
import zcu_tools.experiment.v2 as ze
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.default_cfg import ModuleLibrary
from zcu_tools.notebook.utils import make_sweep, make_comment
from zcu_tools.auto import make_cfg
from zcu_tools.utils.datasaver import create_datafolder
```

```python
import zcu_tools.config as zc

# zc.config.DATA_DRY_RUN = True  # don't save data
# zc.config.YOKO_DRY_RUN = True  # don't run yoko
```

# Create data folder

```python
chip_name = r"Q12_2D[3]/Q1"

data_host = None
# data_host = "021-zcu216"

database_path = create_datafolder(os.path.join(os.getcwd(), ".."), prefix=chip_name)
```

# Connect to zcu216

```python
from zcu_tools.remote import make_proxy
from zcu_tools.program.base import MyProgram  # noqa: F401
from zcu_tools.notebook.utils import get_ip_address  # noqa: F401

# zc.config.LOCAL_IP = get_ip_address("Tailscale")
zc.config.LOCAL_IP = "192.168.10.232"
zc.config.LOCAL_PORT = 8887

soc, soccfg, rm_prog = make_proxy("192.168.10.7", 8887, proxy_prog=True)
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
res_ch = 0
qub_all_ch = 14
qub_1_4_ch = 2
qub_4_5_ch = 5

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
    VISAaddress="USB0::0x0B21::0x0039::91WB18859::INSTR", rm=pyvisa.ResourceManager()
)
GlobalDeviceManager.register_device("flux_yoko", flux_dev)
cur_A = flux_dev.get_current()
cur_A
```

```python
# cur_A = 0.0e-3
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
            "freq": 5351.0,
            # "freq": r_f,
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": 1.5,  # us
            "trig_offset": 0.4,  # us
        },
    },
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, rounds=5000)

lookback_exp = ze.LookbackExperiment()
Ts, signals = lookback_exp.run(soc, soccfg, cfg)
```

```python
predict_offset = lookback_exp.analyze(ratio=0.1, smooth=1.0, ro_cfg=cfg["readout"]["ro_cfg"])
predict_offset
```

```python
timeFly = float(predict_offset)
timeFly
```

```python
lookback_exp.save(
    filepath=os.path.join(database_path, "lookback"),
    comment=make_comment(cfg, f"timeFly = {timeFly}us"),
    server_ip=data_host,
)
```

# OneTone

```python
res_name = "R1"
```

```python
res_probe_len = 5.1  # us
ModuleLibrary.register_waveform(
    ro_waveform={
        "style": "flat_top",
        "raise_pulse": {"style": "cosine", "length": 0.1},
        "length": res_probe_len,  # us
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
            "gain": 0.15,
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": res_probe_len - 0.1,  # us
            "trig_offset": timeFly + 0.05,  # us
        },
    },
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": make_sweep(5330, 5370, 101),
    # "sweep": make_sweep(r_f-4, r_f+4, 101),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

res_freq_exp = ze.onetone.FreqExperiment()
fpts, signals = res_freq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
f, kappa = res_freq_exp.analyze(asym=True)
f
```

```python
r_f = f
rf_w = kappa
```

```python
res_freq_exp.save(
    filepath=os.path.join(database_path, f"{res_name}_freq@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"resonator frequency = {r_f}MHz"),
    server_ip=data_host,
)
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
            "freq": r_f,  # MHz
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": res_probe_len - 0.1,  # us
            "trig_offset": timeFly + 0.05,  # us
        },
    },
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": {
        "gain": make_sweep(0.01, 0.5, 51),
        "freq": make_sweep(r_f - 10, r_f + 10, 101),
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=30)

res_pdr_exp = ze.onetone.PowerDepExperiment()
pdrs, fpts, signals2D = res_pdr_exp.run(
    soc, soccfg, cfg, dynamic_avg=True, gain_ref=0.03
)
```

```python
res_pdr_exp.save(
    filepath=os.path.join(database_path, f"{res_name}_pdr@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
    server_ip=data_host,
)
```

## Flux dependence

```python
cur_A = -6.0e-3
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
            "gain": 0.1,
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": res_probe_len - 0.1,  # us
            "trig_offset": timeFly + 0.05,  # us
        },
    },
    "dev": {
        "flux_dev": "yoko"
    },
    "sweep": {
        "flux": make_sweep(6.0e-3, 8e-3, 201),
        "freq": make_sweep(r_f - 5, r_f + 5, 61),
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=100)

res_flux_exp = ze.onetone.FluxDepExperiment()
As, fpts, signals2D = res_flux_exp.run(soc, soccfg, cfg)
```

```python
res_flux_exp.save(
    filepath=os.path.join(database_path, f"{res_name}_flux"),
    comment=make_comment(cfg),
    server_ip=data_host,
)
```

```python
%matplotlib widget
actline = res_flux_exp.analyze()
```

```python
mA_c, mA_e = actline.get_positions()
mA_c, mA_e
```

```python
cur_A = 8.0e-3
1e3 * flux_dev.set_current(cur_A)
```

## Set readout pulse

```python
ro_pulse_len = 3.1  # us
ModuleLibrary.register_module(
    readout_rf={
        "type": "base",
        "pulse_cfg": {
            **ModuleLibrary.get_waveform("ro_waveform"),
            "ch": res_ch,
            "nqz": 2,
            "freq": r_f,
            "gain": 0.1,
            "length": ro_pulse_len,  # us
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": ro_pulse_len - 0.1,  # us
            "trig_offset": timeFly + 0.05,  # us
        },
        "desc": "lower power readout with exact resonator frequency",
    }
)
```

# TwoTone

```python
# preditor = FluxoniumPredictor(f"../result/{chip_name}/params.json")
preditor = FluxoniumPredictor("../result/Q12_2D[3]/Q1/params.json")
```

```python
qub_name = "Q1"
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
# cur_A = 0.0e-3
1e3 * flux_dev.set_current(cur_A)
```

```python
q_f = preditor.predict_freq(cur_A, transition=(0, 1))
q_f
```

```python
exp_cfg = {
    "reset": "reset_120",
    # "qub_pulse": {
    #     **ModuleLibrary.get_waveform("qub_waveform"),
    #     "ch": qub_all_ch,
    #     "nqz": 1,
    #     "gain": 0.25,
    #     "length": 5.0,  # us
    #     "mixer_freq": 310,
    #     # "mixer_freq": q_f,
    #     # "post_delay": None,
    # },
    "qub_pulse": "pi_amp",
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": make_sweep(q_f - 3, q_f + 3, step=0.05),
    # "sweep": make_sweep(305, 312, step=0.05),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=100)

qub_freq_exp = ze.twotone.FreqExperiment()
fpts, signals = qub_freq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
f, kappa = qub_freq_exp.analyze(max_contrast=True)
f
```

```python
q_f = f
qf_w = kappa
```

```python
qub_freq_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_freq@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"frequency = {f}MHz"),
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
            "ch": qub_1_4_ch,
            "nqz": 2,
            "gain": 0.5,
            # "mixer_freq": reset_f,
            "post_delay": 5.0 / rf_w,  # 5 times the resonator linewidth
        },
    },
    "readout": "readout_rf",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": make_sweep(reset_f - 150, reset_f + 150, 101),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

single_reset_freq_exp = ze.twotone.reset.single_tone.FreqExperiment()
fpts, signals = single_reset_freq_exp.run(soc, soccfg, cfg, remove_bg=True)
```

```python
f, kappa = single_reset_freq_exp.analyze(max_contrast=True)
f
```

```python
reset_f = f
```

```python
single_reset_freq_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_single_reset_freq@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"frequency = {f}MHz"),
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
            "ch": qub_1_4_ch,
            "nqz": 2,
            "gain": 0.5,
            "freq": reset_f,
            # "mixer_freq": reset_f,
            "post_delay": 5.0 / rf_w,  # 5 times the resonator linewidth
        },
    },
    "qub_pulse": "pi_amp",
    "readout": "readout_rf",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": make_sweep(0.03, 5.0, 51),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=100)

single_reset_length_exp = ze.twotone.reset.single_tone.LengthExperiment()
Ts, signals = single_reset_length_exp.run(soc, soccfg, cfg)
```

```python
single_reset_length_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_single_reset_time@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
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

#### Check Reset

```python
exp_cfg = {
    "reset": "reset_10",
    "init_pulse": "pi_amp",
    "tested_reset": "reset_10",
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": make_sweep(0.0, 0.6, 51),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

single_reset_check_exp = ze.twotone.reset.single_tone.RabiCheckExperiment()
pdrs, signals = single_reset_check_exp.run(soc, soccfg, cfg)
```

```python
single_reset_check_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_reset_check@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
    server_ip=data_host,
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
    # "reset": "reset_120",
    # "init_pulse": {
    #     **ModuleLibrary.get_waveform("qub_waveform"),
    #     "ch": qub_all_ch,
    #     "nqz": 2,
    #     "gain": 0.01,
    #     "mixer_freq": q_f,
    # },
    "init_pulse": "pi_amp",
    "tested_reset": {
        "type": "two_pulse",
        "pulse1_cfg": {
            **ModuleLibrary.get_waveform("qub_waveform"),
            "ch": qub_4_5_ch,
            "nqz": 2,
            "gain": 0.5,
            "length": 10.0,  # us
            "mixer_freq": reset_f1,
            "post_delay": None,
        },
        "pulse2_cfg": {
            **ModuleLibrary.get_waveform("qub_waveform"),
            "ch": qub_1_4_ch,
            "nqz": 1,
            "gain": 1.0,
            "length": 10.0,  # us
            # "mixer_freq": reset_f2,
            "post_delay": 5.0 / rf_w,  # 5 times the resonator linewidth
        },
    },
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": {
        # "freq1": make_sweep(reset_f1 - 100, reset_f1 + 100, 51),
        # "freq2": make_sweep(reset_f2 - 100, reset_f2 + 100, 51),
        "freq1": make_sweep(3332.5, 3334.5, 51),
        "freq2": make_sweep(1697, 1708, 51),
    },
    "relax_delay": 1.0,  # us
}
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

dual_reset_freq_exp = ze.twotone.reset.dual_tone.FreqExperiment()
fpts1, fpts2, signals = dual_reset_freq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
xlabal = f"|{reset1_trans[0]}, 0> - |{reset1_trans[1]}, 0>"
ylabal = f"|{reset2_trans[0]}, 0> - |{reset2_trans[1]}, 1>"
f1, f2 = dual_reset_freq_exp.analyze(
    smooth=0.5, xname=xlabal, yname=ylabal
)
f1, f2
```

```python
reset_f1 = f1
reset_f2 = f2
```

```python
dual_reset_freq_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_mux_reset_freq@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"frequency = ({reset_f1:.1f}, {reset_f2:.1f})MHz"),
    server_ip=data_host,
)
```

#### Set Mux Reset Pulse

```python
mux_reset_len = 10.0
ModuleLibrary.register_module(
    reset_120={
        "type": "two_pulse",
        "pulse1_cfg": {
            **cfg["tested_reset"]["pulse1_cfg"],
            "freq": reset_f1,
            "length": mux_reset_len,
        },
        "pulse2_cfg": {
            **cfg["tested_reset"]["pulse2_cfg"],
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
    "reset": "reset_120",
    # "init_pulse": {
    #     **ModuleLibrary.get_waveform("qub_waveform"),
    #     "ch": qub_all_ch,
    #     "nqz": 2,
    #     "gain": 0.01,
    #     "mixer_freq": q_f,
    # },
    "init_pulse": "pi_amp",
    "tested_reset": "reset_120",
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": {
        "gain1": make_sweep(0.0, 1.0, 51),
        "gain2": make_sweep(0.0, 1.0, 51),
    },
    "relax_delay": 0.0,  # us
    # "relax_delay": 3 * t1,
}
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

dual_reset_pdr_exp = ze.twotone.reset.dual_tone.PowerExperiment()
pdrs1, pdrs2, signals2D = dual_reset_pdr_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
xlabal = f"|{reset1_trans[0]}, 0> - |{reset1_trans[1]}, 0>"
ylabal = f"|{reset2_trans[0]}, 0> - |{reset2_trans[1]}, 1>"
gain1, gain2 = dual_reset_pdr_exp.analyze(xname=xlabal, yname=ylabal)
```

```python
dual_reset_pdr_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_mux_reset_gain@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"best gain = ({gain1:.1f}, {gain2:.1f})"),
    server_ip=data_host,
)
```

```python
ModuleLibrary.update_module(
    "reset_120",
    override_cfg={
        "pulse1_cfg": {"gain": gain1},
        "pulse2_cfg": {"gain": gain2},
    },
)
```

#### Reset Time

```python
exp_cfg = {
    "reset": "reset_120",
    "init_pulse": "pi_amp",
    "tested_reset": "reset_120",
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": make_sweep(0.05, 35.0, 51),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

dual_reset_len_exp = ze.twotone.reset.dual_tone.LengthExperiment()
Ts, signals = dual_reset_len_exp.run(soc, soccfg, cfg)
```

```python
dual_reset_len_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_mux_reset_time@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
    server_ip=data_host,
)
```

```python
mux_reset_len = 20.0  # us
ModuleLibrary.update_module(
    "reset_120",
    override_cfg={
        "pulse1_cfg": {"length": mux_reset_len},
        "pulse2_cfg": {"length": mux_reset_len},
    },
)
```

#### Check Reset

```python
exp_cfg = {
    "reset": "reset_120",
    "init_pulse": "pi_amp",
    "tested_reset": "reset_120",
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": make_sweep(0.0, 0.6, 51),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

dual_reset_check_exp = ze.twotone.reset.dual_tone.RabiCheckExperiment()
pdrs, signals = dual_reset_check_exp.run(soc, soccfg, cfg)
```

```python
dual_reset_check_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_mux_reset_check@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
    server_ip=data_host,
)
```

## Power dependence

```python
exp_cfg = {
    # "reset": "reset_120",
    # "qub_pulse": {
    #     **ModuleLibrary.get_waveform("qub_waveform"),
    #     "ch": qub_all_ch,
    #     "nqz": 2,
    #     "length": 5,  # us
    #     # "mixer_freq": 4700,
    #     "mixer_freq": q_f,
    #     # "post_delay": None,
    # },
    "qub_pulse": "pi_amp",
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": {
        "gain": make_sweep(0.05, 1.0, 30),
        # "freq": make_sweep(1700, 2000, 30),
        "freq": make_sweep(q_f - 3, q_f + 3, 101),
    },
    "relax_delay": 10.0,  # us
}
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

qub_pdr_exp = ze.twotone.PowerDepExperiment()
fpts, pdrs, signals2D = qub_pdr_exp.run(soc, soccfg, cfg)
```

```python
qub_pdr_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_pdr@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
    server_ip=data_host,
)
```

## Flux Dependence

```python
cur_A = 8.0e-3
1e3 * flux_dev.set_current(cur_A)
```

```python
exp_cfg = {
    "qub_pulse": {
        **ModuleLibrary.get_waveform("qub_waveform"),
        "ch": qub_all_ch,
        "nqz": 2,
        "gain": 0.2,
        "length": 5,  # us
        "mixer_freq": 4000,
    },
    "readout": "readout_rf",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": {
        "flux": make_sweep(8.0e-3, -4.0e-3, 251),
        "freq": make_sweep(3000, 4800, 801),
    },
    "relax_delay": 0.0,  # us

}
cfg = make_cfg(exp_cfg, reps=1000, rounds=20)

qub_flux_exp = ze.twotone.FluxDepExperiment()
As, fpts, signals2D = qub_flux_exp.run(soc, soccfg, cfg)
```

```python
qub_flux_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_flux"),
    comment=make_comment(cfg),
    server_ip=data_host,
)
```

```python
%matplotlib widget
actline = qub_flux_exp.analyze()
```

```python
mA_c, mA_e = actline.get_positions()
mA_c, mA_e
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
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": make_sweep(r_f - 5, r_f + 5, 101),
    "relax_delay": 0.0,  # us
    # "relax_delay": 2 * t1, # us
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

dispersive_shift_exp = ze.twotone.DispersiveExperiment()
fpts, signals = dispersive_shift_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
chi, rf_w = dispersive_shift_exp.analyze(asym=True)
chi
```

```python
dispersive_shift_exp.save(
    filepath=os.path.join(database_path, f"{res_name}_dispersive@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"chi = {chi:.3g} MHz, kappa = {rf_w:.3g} MHz"),
    server_ip=data_host,
)
```

## AC Stark Shift

```python
ac_qub_len = ModuleLibrary.get_module("pi_amp")["length"]  # us
exp_cfg = {
    # "reset": "reset_120",
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
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "sweep": {
        "gain": make_sweep(0.01, 0.2, 201),
        "freq": make_sweep(q_f - 5, q_f + 100, step=0.5),
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=100)

ac_stark_exp = ze.twotone.AcStarkExperiment()
pdrs, fpts, signals2D = ac_stark_exp.run(soc, soccfg, cfg, sqrt_uniform=True)
```

```python
ac_stark_coeff = ac_stark_exp.analyze(chi=chi, kappa=rf_w, deg=2, cutoff=0.15)
ac_stark_coeff
```

```python
ac_stark_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_ac_stark@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"ac_stark_coeff = {ac_stark_coeff:.3g} MHz"),
    server_ip=data_host,
)
```

## All XY

```python
exp_cfg = {
    "reset": "reset_120",
    "X180_pulse": {
        **ModuleLibrary.get_module("pi_amp"),
    },
    "Y180_pulse": {
        **ModuleLibrary.get_module("pi_amp"),
        "phase": 90,  # degrees
    },
    "X90_pulse": {
        **ModuleLibrary.get_module("pi2_amp"),
    },
    "Y90_pulse": {
        **ModuleLibrary.get_module("pi2_amp"),
        "phase": 90,  # degrees
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=100)

allxy_exp = ze.twotone.AllXYExperiment()
sequence, signals = allxy_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
allxy_exp.analyze()
```

```python
allxy_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_allxy@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
    server_ip=data_host,
)
```

# Rabi

## Length Rabi

```python
exp_cfg = {
    "reset": "reset_120",
    "qub_pulse": {
        **ModuleLibrary.get_waveform("qub_waveform"),
        "ch": qub_all_ch,
        "nqz": 1,
        "freq": q_f,
        "gain": 0.3,
        # "gain": pi_gain,
        "mixer_freq": q_f,
    },
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "relax_delay": 0.0,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.1, 5.0, 101),
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=20)

qub_lenrabi_exp = ze.twotone.LenRabiExperiment()
Ts, signals = qub_lenrabi_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
pi_len, pi2_len = qub_lenrabi_exp.analyze(decay=True, max_contrast=True)
pi_len, pi2_len
```

```python
qub_lenrabi_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_len_rabi@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"pi len = {pi_len}us\npi/2 len = {pi2_len}us"),
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
        "ch": qub_all_ch,
        "nqz": 1,
        "freq": q_f,
        # "length": 0.3,  # us
        "length": 1.5*pi_len,
        "mixer_freq": q_f,
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "relax_delay": 0.0,  # us
    # "relax_delay": 5 * t1,
    "sweep": make_sweep(0.0, 0.4, 51),
    # "sweep": make_sweep(0.0, max_gain, 51),
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

qub_amprabi_exp = ze.twotone.AmpRabiExperiment()
pdrs, signals = qub_amprabi_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
pi_gain, pi2_gain = qub_amprabi_exp.analyze(decay=False, max_contrast=True)
pi_gain = int(pi_gain + 0.5) if pi_gain > 1.0 else pi_gain
pi2_gain = int(pi2_gain + 0.5) if pi2_gain > 1.0 else pi2_gain
pi_gain, pi2_gain
```

```python
qub_amprabi_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_amp_rabi@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"pi gain = {pi_gain}\npi/2 gain = {pi2_gain}"),
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
    # "reset": "reset_120",
    "qub_pulse": "pi_amp",
    "readout": ModuleLibrary.get_module(
        "readout_rf",
        # override_cfg={"gain": pdr_max},
    ),
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "relax_delay": 10.0,  # us
    # "relax_delay": 3 * t1,  # us
    "sweep": make_sweep(r_f - 3, r_f + 3, 51),
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

opt_ro_freq_exp = ze.twotone.ro_optimize.OptimizeFreqExperiment()
fpts, snrs = opt_ro_freq_exp.run(soc, soccfg, cfg)
```

```python
fpt_max = opt_ro_freq_exp.analyze()
```

```python
opt_ro_freq_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_ro_opt_freq@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"optimal frequency = {fpt_max:.1f}MHz"),
    server_ip=data_host,
)
```

## Power tuning

```python
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": "pi_amp",
    "readout": ModuleLibrary.get_module(
        "readout_rf",
        {
            "pulse_cfg": {
                "freq": fpt_max,
            }
        },
    ),
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "relax_delay": 10.0,  # us
    # "relax_delay": 3 * t1,  # us
    "sweep": make_sweep(0.01, 1.0, 51),
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

opt_ro_pdr_exp = ze.twotone.ro_optimize.OptimizePowerExperiment()
pdrs, snrs = opt_ro_pdr_exp.run(soc, soccfg, cfg)
```

```python
pdr_max = opt_ro_pdr_exp.analyze()
```

```python
opt_ro_pdr_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_ro_opt_pdr@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"optimal power = {pdr_max:.2f}"),
    server_ip=data_host,
)
```

## Readout Length tuning

```python
exp_cfg = {
    # "reset": "reset_120",
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
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "relax_delay": 10.0,  # us
    # "relax_delay": 3 * t1,  # us
    "sweep": make_sweep(0.1, 15.0, 31),
}
cfg = make_cfg(exp_cfg, reps=10000, rounds=1)

opt_ro_len_exp = ze.twotone.ro_optimize.OptimizeLengthExperiment()
ro_lens, snrs = opt_ro_len_exp.run(soc, soccfg, cfg)
```

```python
ro_max = opt_ro_len_exp.analyze(t0=30.0)
ro_max
```

```python
opt_ro_len_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_ro_opt_len@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"optimal readout length = {ro_max:.2f}us"),
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

```python
# t1 = 5.0
```

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
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "relax_delay": 0.0,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0, 10.0, 101),  # us
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

activate_detune = 0.1 / cfg["sweep"]["step"]
print(f"activate_detune: {activate_detune:.2f}")

t2ramsey_exp = ze.twotone.T2RamseyExperiment()
Ts, signals = t2ramsey_exp.run(soc, soccfg, cfg, detune=activate_detune)
```

```python
%matplotlib inline
t2r, _, detune, _ = t2ramsey_exp.analyze(max_contrast=True)
print(f"real detune: {(detune - activate_detune) * 1e3:.1f}kHz")
```

```python
t2ramsey_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_t2ramsey@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"detune = {detune:.3f}MHz\nt2r = {t2r:.3f}us"),
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
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "relax_delay": 0.0,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.0, 150, 51),
    # "sweep": make_sweep(0.01*t1, 5 * t1, 51),
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

t1_exp = ze.twotone.T1Experiment()
Ts, signals = t1_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
t1, _ = t1_exp.analyze(max_contrast=True, dual_exp=False)
t1
```

```python
t1_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_t1@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"t1 = {t1:.3f}us"),
    server_ip=data_host,
)
```

## T2Echo

```python
exp_cfg = {
    "reset": "reset_120",
    "pi_pulse": "pi_amp",
    "pi2_pulse": "pi2_amp",
    "readout": "readout_dpm",
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "relax_delay": 0.0,  # us
    # "relax_delay": 5 * t1,  # us
    # "sweep": make_sweep(0.0, 1.5 * t2r, 101),
    # "sweep": make_sweep(0.0, 1.5 * t2e, 101),
    "sweep": make_sweep(0.01, 2 * t1, 51),
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

activate_detune = 0.1 / cfg["sweep"]["step"]
print(f"activate_detune: {activate_detune:.2f}")

t2echo_exp = ze.twotone.T2EchoExperiment()
Ts, signals = t2echo_exp.run(soc, soccfg, cfg, detune=activate_detune)
```

```python
%matplotlib inline
t2e, _, detune, _ = t2echo_exp.analyze(max_contrast=True)
```

```python
t2echo_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_t2echo@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"detune = {detune:.3f}MHz\nt2echo = {t2e:.3f}us"),
    server_ip=data_host,
)
```

# Single shot

## Ground state & Excited state

```python
exp_cfg = {
    "reset": "reset_120",
    "qub_pulse": "pi_amp",
    "readout": ModuleLibrary.get_module(
        "readout_dpm",
        {
            "pulse_cfg": {
                # "length": 0.2 * t1 + timeFly + 0.1,
            },
            "ro_cfg": {
                # "ro_length": 0.2 * t1,
            },
        },
    ),
    "dev": {
        "flux_dev": "yoko",
        "flux": cur_A,  # A
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, shots=100000)
print("readout length: ", cfg["readout"]["ro_cfg"]["ro_length"])

singleshot_exp = ze.twotone.SingleShotExperiment()
signals = singleshot_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib widget
%matplotlib inline
fid, _, _, pops = singleshot_exp.analyze(backend="pca")
print(f"Optimal fidelity after rotation = {fid:.1%}")
```

```python
from zcu_tools.simulate.temp import effective_temperature

n_g = pops[0][0] # n_gg
n_e = pops[0][1] # n_ge

n_g, n_e = (n_g, n_e) if n_g > n_e else (n_e, n_g)  # ensure n_g >= n_e

eff_T, err_T = effective_temperature(population=[(n_g, 0.0), (n_e, q_f)])
eff_T, err_T
```

```python
singleshot_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_singleshot_ge@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(
        cfg, f"fide: {fid:.1%}, (n_g, n_e): ({n_g:.1%}, {n_e:.1%}), eff_T: {eff_T:.1f}"
    ),
    server_ip=data_host,
)
```

```python

```
