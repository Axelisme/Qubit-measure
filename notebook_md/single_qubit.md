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
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

%autoreload 2
import zcu_tools.experiment.v2 as ze
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.library import ModuleLibrary
from zcu_tools.notebook.utils import make_sweep, make_comment, savefig
from zcu_tools.utils.datasaver import create_datafolder
```

# Create data folder and cfg

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
qub_1_4_ch = 14
# qub_5_6_ch = 2
# lo_flux_ch = 14

ro_ch = 0
```

# Initialize devices

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
cur_A
# cur_V = flux_yoko.get_voltage()
# cur_V
```

```python
# # cur_A = 0.0e-3
flux_yoko.set_current(current=cur_A)
# cur_V = 0.0
# flux_yoko.set_voltage(voltage=cur_V)
```

## RF Source

```python
from zcu_tools.device.rf_source import RFSource

rf_source = RFSource(
    VISAaddress="USB0::0x0B21::0x0039::91WB18859::INSTR", rm=resource_manager
)
GlobalDeviceManager.register_device("rf_source", rf_source)
```

# Lookback

```python
timeFly = 0.6
```

```python
%matplotlib widget
exp_cfg = {
    "readout": {
        "type": "base",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": 0.65},
            "ch": res_ch,
            "nqz": 2,
            "gain": 1.0,
            "freq": 5930,
            # "freq": r_f,
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

```python
timeFly = float(predict_offset)
timeFly
```

```python
lookback_exp.save(
    filepath=os.path.join(database_path, "lookback"),
    comment=make_comment(cfg, f"timeFly = {timeFly}us"),
)
```

# OneTone

```python
res_name = "R59"
```

```python
res_probe_len = 5.0  # us
ml.register_waveform(
    ro_waveform={
        "style": "flat_top",
        "raise_waveform": {"style": "cosine", "length": 0.1},
        "length": res_probe_len,  # us
    }
)
```

## Resonator Frequency

```python
%matplotlib widget
exp_cfg = {
    "readout": {
        "type": "base",
        "pulse_cfg": {
            "waveform": ml.get_waveform("ro_waveform"),
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
    "sweep": make_sweep(5910, 5950, 101),
    # "sweep": make_sweep(r_f-4, r_f+4, 101),
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

res_freq_exp = ze.onetone.FreqExperiment()
fpts, signals = res_freq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
f, kappa, params, fig = res_freq_exp.analyze(model_type="auto")
```

```python
r_f = f
rf_w = kappa
```

```python
savefig(fig, f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{res_name}_freq.png")
plt.close(fig)
res_freq_exp.save(
    filepath=os.path.join(database_path, f"{res_name}_freq@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{res_name}_freq@{cur_V:.3f}V"),
    comment=make_comment(cfg, str(params)),
)
```

## Power Dependence

```python
%matplotlib widget
exp_cfg = {
    # "init_pulse": "pi_amp",
    "readout": {
        "type": "base",
        "pulse_cfg": {
            "waveform": ml.get_waveform("ro_waveform"),
            "ch": res_ch,
            "nqz": 2,
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": res_probe_len - 0.1,  # us
            "trig_offset": timeFly + 0.05,  # us
        },
    },
    "sweep": {
        "gain": make_sweep(0.01, 1.0, 101),
        "freq": make_sweep(r_f - 20, r_f + 10, 201),
    },
    "relax_delay": 1.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=1000)

res_pdr_exp = ze.onetone.PowerDepExperiment()
pdrs, fpts, signals2D = res_pdr_exp.run(soc, soccfg, cfg, earlystop_snr=20.0)
```

```python
res_pdr_exp.save(
    filepath=os.path.join(database_path, f"{res_name}_pdr@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{res_name}_pdr@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## Flux dependence

```python
cur_A = 0.0e-3
1e3 * flux_yoko.set_current(cur_A)
# cur_V = 0.0
# flux_yoko.set_voltage(cur_V)
```

```python
exp_cfg = {
    "readout": {
        "type": "base",
        "pulse_cfg": {
            "waveform": ml.get_waveform("ro_waveform"),
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
        "flux_yoko": {
            "label": "flux_dev",
            # "mode": "voltage",
            "mode": "current",
        }
    },
    "sweep": {
        "flux": make_sweep(0.5e-3, 1.75e-3, 101),
        # "flux": make_sweep(5.0, -5.0, 101),
        "freq": make_sweep(r_f - 30, r_f + 10, 301),
    },
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=30)

res_flux_exp = ze.onetone.FluxDepExperiment()
As, fpts, signals2D = res_flux_exp.run(soc, soccfg, cfg)
```

```python
res_flux_exp.save(
    filepath=os.path.join(database_path, f"{res_name}_flux"),
    comment=make_comment(cfg),
)
```

```python
%matplotlib widget
actline = res_flux_exp.analyze(
    # mA_c=preditor.A_c, mA_e=preditor.A_c + 0.5 * preditor.period
)
```

```python
mA_c, mA_e = actline.get_positions()
mA_c, mA_e
```

```python
cur_A = mA_e
1e3 * flux_yoko.set_current(cur_A)
# cur_V = mA_c
# flux_yoko.set_voltage(cur_V)
```

## Set readout pulse

```python
ro_pulse_len = 0.52  # us
ml.register_module(
    readout_rf={
        "type": "base",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": ro_pulse_len},
            "ch": res_ch,
            "nqz": 2,
            "freq": r_f,
            # "freq": 5927,
            "gain": 0.3,
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": ro_pulse_len - 0.1,  # us
            "trig_offset": timeFly + 0.01,  # us
        },
        "desc": "lower power readout with exact resonator frequency",
    }
)
```

# TwoTone

```python
preditor = FluxoniumPredictor(f"../result/{chip_name}/params.json")
# preditor = FluxoniumPredictor("../result/SF008/params.json")
```

```python
qub_name = "Si001"
ml.register_waveform(
    qub_flat={
        "style": "flat_top",
        "raise_waveform": {"style": "cosine", "length": 0.02},
        "length": 2.0,
    },
    qub_cos={
        "style": "cosine",
        "length": 2.0,
    },
)
```

```python
from zcu_tools.sample_table import SampleTable

sample_table = SampleTable(f"../result/{chip_name}/samples.csv")
```

## Twotone Frequency

```python
# print(preditor.predict_freq(preditor.flx_to_A(0.53)))
cur_A = 0.95e-3
# cur_A = preditor.flx_to_A(0.6)
# cur_A = flux_yoko.get_current()
1e3 * flux_yoko.set_current(cur_A)
# cur_V = 0.0
# flux_yoko.set_voltage(cur_V)
```

```python
q_f = preditor.predict_freq(cur_A, transition=(0, 1))
# q_f = preditor.predict_freq(cur_V, transition=(0, 1))
q_f
```

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_120",
    # "init_pulse": "pi_len",
    "qub_pulse": {
        "waveform": ml.get_waveform("qub_flat", override_cfg={"length": 4.0}),
        "ch": qub_1_4_ch,
        "nqz": 2,
        "gain": 0.1,
        "mixer_freq": q_f,
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": make_sweep(q_f - 10, q_f + 10, step=0.25),
    # "sweep": make_sweep(3800, 3900, step=1.0),
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

qub_freq_exp = ze.twotone.FreqExperiment()
fpts, signals = qub_freq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
f, kappa, fig = qub_freq_exp.analyze(max_contrast=True)
f
```

```python
q_f = f
qf_w = kappa

savefig(fig, f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_freq.png")
plt.close(fig)
```

```python
qub_freq_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_freq@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_freq@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"frequency = {f}MHz"),
)
```

```python
bias = preditor.calculate_bias(cur_A, q_f)
bias * 1e3
# bias = preditor.calculate_bias(cur_V, q_f)
# bias
```

```python
preditor.update_bias(bias)
```

## Reset

```python
# cur_A = 0.0e-3
# 1e3 * flux_dev.set_current(cur_A)
# cur_V = -12.61
flux_yoko.set_voltage(cur_V)
```

### One Pulse

```python
reset_f = r_f - q_f
reset_f
```

```python
%matplotlib widget
exp_cfg = {
    # "init_pulse": {
    #     **ml.get_waveform("qub_waveform"),
    #     "ch": qub_ch,
    #     "nqz": 2,
    #     "gain": 0.01,
    #     "mixer_freq": q_f,
    # },
    "init_pulse": "pi_amp",
    "tested_reset": {
        "type": "pulse",
        "pulse_cfg": {
            "waveform": ml.get_waveform("qub_flat", {"length": 3.0}),
            "ch": qub_1_4_ch,
            "nqz": 2,
            "gain": 1.0,
            # "mixer_freq": reset_f,
            "mixer_freq": 0.5 * (reset_f + q_f),
            "post_delay": 5 / (2 * np.pi * rf_w),
        },
    },
    "readout": "readout_dpm",
    "sweep": make_sweep(reset_f - 15, reset_f + 15, step=0.25),
    "relax_delay": 20.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

single_reset_freq_exp = ze.twotone.reset.single_tone.FreqExperiment()
fpts, signals = single_reset_freq_exp.run(soc, soccfg, cfg)
```

```python
f, kappa = single_reset_freq_exp.analyze()
f
```

```python
reset_f = f
```

```python
single_reset_freq_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_reset_freq@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_reset_freq@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"frequency = {f}MHz"),
)
```

#### Reset Time

```python
exp_cfg = {
    "init_pulse": "pi_amp",
    "tested_reset": {
        "type": "pulse",
        "pulse_cfg": {
            "waveform": ml.get_waveform("qub_flat"),
            "ch": qub_1_4_ch,
            "nqz": 2,
            "gain": 1.0,
            "freq": reset_f,
            # "mixer_freq": reset_f,
            "mixer_freq": 0.5 * (reset_f + q_f),
            "post_delay": 5 / (2 * np.pi * rf_w),
        },
    },
    "readout": "readout_dpm",
    "sweep": make_sweep(0.1, 7.0, 50),
    "relax_delay": 15.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

single_reset_length_exp = ze.twotone.reset.single_tone.LengthExperiment()
Ts, signals = single_reset_length_exp.run(soc, soccfg, cfg)
```

```python
single_reset_length_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_reset_time@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_reset_time@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

#### Set Reset Pulse

```python
cfg["tested_reset"]["pulse_cfg"]["waveform"].update(length=4.0)  # us
ml.register_module(
    reset_10={
        **cfg["tested_reset"],
        "desc": "Reset with one pulse from 1 to 0",
    },
)
```

#### Check Reset

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    "rabi_pulse": {
        **ml.get_module("pi_amp"),
        "mixer_freq": 0.5 * (reset_f + q_f),
    },
    # "tested_reset": "reset_10",
    "tested_reset": ml.get_module(
        "reset_10",
        {
            "pulse_cfg": {
                "mixer_freq": 0.5 * (reset_f + q_f),
            }
        },
    ),
    "readout": "readout_dpm",
    "sweep": make_sweep(0.0, 1.0, 51),
    "relax_delay": 20.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

single_reset_check_exp = ze.twotone.reset.RabiCheckExperiment()
pdrs, signals = single_reset_check_exp.run(soc, soccfg, cfg)
```

```python
single_reset_check_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_reset_check@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_reset_check@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

### Two pulse

```python
reset1_trans = (1, 2)
# reset_f1 = preditor.predict_freq(cur_A, transition=reset1_trans)
reset_f1 = preditor.predict_freq(cur_V, transition=reset1_trans)
reset_f1
```

#### Reset Freq 1

```python
exp_cfg = {
    # "reset": "reset_120",
    "init_pulse": "pi_amp",
    "qub_pulse": {
        "waveform": ml.get_waveform(
            "qub_flat",
            override_cfg={"length": 5.0},
        ),
        "ch": qub_1_4_ch,
        "nqz": 1,
        "gain": 0.03,
        "mixer_freq": reset_f1,
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": make_sweep(reset_f1 - 15, reset_f1 + 15, step=0.1),
    # "sweep": make_sweep(4680, 4710, step=0.1),
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=1000)

dualreset_freq1_exp = ze.twotone.FreqExperiment()
fpts, signals = dualreset_freq1_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
f, kappa = dualreset_freq1_exp.analyze(max_contrast=True)
f
```

```python
reset_f1 = f
resetf1_w = kappa
```

```python
dualreset_freq1_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_dualreset_freq1@{cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_dualreset_freq1@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"frequency = {f}MHz"),
)
```

```python
# bias = preditor.calculate_bias(cur_A, reset_f1, transition=reset1_trans)
# bias * 1e3
bias = preditor.calculate_bias(cur_V, reset_f1, transition=reset1_trans)
bias
```

```python
preditor.update_bias(bias)
```

#### Reset Freq 2

```python
reset2_trans = (2, 0)
# reset_f2 = abs(r_f + preditor.predict_freq(cur_A, transition=reset2_trans))
reset_f2 = abs(r_f + preditor.predict_freq(cur_V, transition=reset2_trans))
reset_f2
```

```python
exp_cfg = {
    # "reset": "reset_120",
    "init_pulse": "pi_amp",
    "tested_reset": {
        "type": "two_pulse",
        "pulse1_cfg": {
            "waveform": ml.get_waveform(
                "qub_flat",
                override_cfg={"length": 5.0},
            ),
            "ch": qub_1_4_ch,
            "nqz": 1,
            "gain": 0.03,
            "mixer_freq": reset_f1,
            "block_mode": False,
        },
        "pulse2_cfg": {
            "waveform": ml.get_waveform(
                "qub_flat",
                override_cfg={"length": 5.0},
            ),
            "ch": qub_1_4_ch,
            "nqz": 1,
            "gain": 1.0,
            "mixer_freq": reset_f2,
            "post_delay": 5.0 / rf_w,  # 5 times the resonator linewidth
        },
    },
    "readout": "readout_dpm",
    "sweep": {
        "freq1": make_sweep(reset_f1 - 5, reset_f1 + 5, step=0.2),
        "freq2": make_sweep(reset_f2 - 5, reset_f2 + 5, step=0.2),
        # "freq1": make_sweep(1438, 1453, step=0.5),
        # "freq2": make_sweep(2610, 2620, step=0.1),
    },
    # "relax_delay": 5 / rf_w,  # us
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=1000)

dualreset_freq2_exp = ze.twotone.reset.dual_tone.FreqExperiment()
fpts1, fpts2, signals = dualreset_freq2_exp.run(soc, soccfg, cfg, method="hard")
```

```python
%matplotlib inline
xlabal = f"|{reset1_trans[0]}, 0> - |{reset1_trans[1]}, 0>"
ylabal = f"|{reset2_trans[0]}, 0> - |{reset2_trans[1]}, 1>"
f1, f2 = dualreset_freq2_exp.analyze(smooth=0.5, xname=xlabal, yname=ylabal)
f1, f2
```

```python
reset_f1 = f1
reset_f2 = f2
```

```python
dualreset_freq2_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_mux_reset_freq@{cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_mux_reset_freq@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"frequency = ({reset_f1:.1f}, {reset_f2:.1f})MHz"),
)
```

#### Set Mux Reset Pulse

```python
mux_reset_len = 10.0
ml.register_module(
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
    #     **ml.get_waveform("qub_waveform"),
    #     "ch": qub_all_ch,
    #     "nqz": 2,
    #     "gain": 0.01,
    #     "mixer_freq": q_f,
    # },
    "init_pulse": "pi_amp",
    "tested_reset": "reset_120",
    "readout": "readout_dpm",
    "sweep": {
        "gain1": make_sweep(0.0, 0.3, 51),
        "gain2": make_sweep(0.5, 1.0, 51),
    },
    "relax_delay": 0.0,  # us
    # "relax_delay": 3 * t1,
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

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
    # filepath=os.path.join(database_path, f"{qub_name}_mux_reset_gain@{cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_mux_reset_gain@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"best gain = ({gain1:.1f}, {gain2:.1f})"),
)
```

```python
ml.update_module(
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
    "sweep": make_sweep(0.05, 6.0, 51),
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

dual_reset_len_exp = ze.twotone.reset.dual_tone.LengthExperiment()
Ts, signals = dual_reset_len_exp.run(soc, soccfg, cfg)
```

```python
dual_reset_len_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_mux_reset_time@{cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_mux_reset_time@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

```python
mux_reset_len = 4.0  # us
ml.update_module(
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
    "sweep": make_sweep(0.0, 1.0, 51),
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

dual_reset_check_exp = ze.twotone.reset.dual_tone.RabiCheckExperiment()
pdrs, signals = dual_reset_check_exp.run(soc, soccfg, cfg)
```

```python
dual_reset_check_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_mux_reset_check@{cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_mux_reset_check@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

### Bath Reset

#### rabi frequency

```python
%matplotlib widget
exp_cfg = {
    "reset": "reset_10",
    "qub_pulse": {
        **ml.get_module("pi_amp"),
        "waveform": ml.get_waveform("qub_flat"),
        "gain": 0.5,
    },
    "readout": "readout_dpm",
    "relax_delay": 0.0,  # us
    "sweep": make_sweep(0.03, 3.0, 151),
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

rabifreq_exp = ze.twotone.LenRabiExperiment()
Ts, signals = rabifreq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
*_, rabi_f, fig = rabifreq_exp.analyze(decay=True)
plt.show(fig)
savefig(
    fig, f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_rabifreq.png"
)
plt.close(fig)
```

#### bath frequency

```python
%matplotlib widget
exp_cfg = {
    "reset": "reset_10",
    "init_pulse": "pi_amp",
    "tested_reset": {
        "type": "bath",
        "qubit_tone_cfg": {
            **ml.get_module("pi_amp"),
            "waveform": ml.get_waveform(
                "qub_flat",
                override_cfg={"length": 3.0},
            ),
            "block_mode": False,
        },
        "cavity_tone_cfg": {
            **ml.get_module("readout_dpm")["pulse_cfg"],
            "waveform": ml.get_waveform(
                "qub_flat",
                override_cfg={"length": 3.0},
            ),
            "gain": 0.3,
            "post_delay": 5.0 / (2 * np.pi * rf_w),  # 5 times the resonator linewidth
        },
        "pi2_cfg": {
            **ml.get_module("pi2_amp"),
            # "phase": 90,
        },
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": {
        "freq": make_sweep(r_f - 4.0 * rabi_f, r_f - 0.25 * rabi_f, 51),
        "gain": make_sweep(0.05, 1.0, 51),
    },
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

bathreset_freq_exp = ze.twotone.reset.bath.FreqGainExperiment()
gains, fpts, signals = bathreset_freq_exp.run(soc, soccfg, cfg)
```

```python
bathreset_gain, bathreset_freq = bathreset_freq_exp.analyze(smooth=1, background="min")
```

```python
bathreset_freq_exp.save(
    filepath=os.path.join(
        database_path, f"{qub_name}_bathreset_freqgain@{cur_A * 1e3:.3f}mA"
    ),
    # filepath=os.path.join(database_path, f"{qub_name}_bathreset_freqgain@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

#### Length

```python
exp_cfg = {
    # "reset": "reset_10",
    # "init_pulse": "pi_amp",
    "tested_reset": {
        "type": "bath",
        "qubit_tone_cfg": {
            **ml.get_module("pi_amp"),
            "waveform": ml.get_waveform("qub_flat"),
            "gain": bathreset_gain,
            "block_mode": False,
        },
        "cavity_tone_cfg": {
            **ml.get_module("readout_dpm")["pulse_cfg"],
            "waveform": ml.get_waveform("qub_flat"),
            "freq": bathreset_freq,
            "gain": 0.3,
            "post_delay": 5.0 / (2 * np.pi * rf_w),  # 5 times the resonator linewidth
        },
        "pi2_cfg": "pi2_amp",
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": make_sweep(0.03, 5.0, 151),
    "relax_delay": 20.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

activate_detune = 0.05 / cfg["sweep"]["step"]
print(f"activate_detune: {activate_detune:.2f}")

bathreset_len_exp = ze.twotone.reset.bath.LengthExperiment()
lens, signals = bathreset_len_exp.run(soc, soccfg, cfg, detune=activate_detune)
```

```python
bathreset_len_exp.save(
    filepath=os.path.join(
        database_path, f"{qub_name}_bathreset_length@{cur_A * 1e3:.3f}mA"
    ),
    # filepath=os.path.join(database_path, f"{qub_name}_bathreset_length@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

#### Set Bath reset pulse

```python
bath_reset_len = 2.0  # us
cfg["tested_reset"]["qubit_tone_cfg"]["waveform"].update(length=bath_reset_len)
cfg["tested_reset"]["cavity_tone_cfg"]["waveform"].update(length=bath_reset_len)
ml.register_module(
    reset_bath={
        **cfg["tested_reset"],
        "desc": "Reset with cavity-assisted bath reset",
    },
)
```

#### Phase

```python
exp_cfg = {
    "reset": "reset_10",
    "tested_reset": "reset_bath",
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": make_sweep(-170.0, 170.0, 51),
    "relax_delay": 1.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

bathreset_phase_exp = ze.twotone.reset.bath.PhaseExperiment()
phases, signals = bathreset_phase_exp.run(soc, soccfg, cfg)
```

```python
max_phase, min_phase = bathreset_phase_exp.analyze()
```

```python
bathreset_phase_exp.save(
    filepath=os.path.join(
        database_path, f"{qub_name}_bathreset_phase@{cur_A * 1e3:.3f}mA"
    ),
    # filepath=os.path.join(database_path, f"{qub_name}_bathreset_phase@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

```python
ml.update_module(
    name="reset_bath",
    override_cfg={
        "pi2_cfg": {
            "phase": max_phase,
            # "phase": min_phase,
        }
    },
)
```

#### Check reset

```python
exp_cfg = {
    "reset": "reset_10",
    "rabi_pulse": "pi_amp",
    "tested_reset": "reset_bath",
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": make_sweep(0.0, 1.0, 31),
    "relax_delay": 1.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

bathreset_rabicheck_exp = ze.twotone.reset.RabiCheckExperiment()
pdrs, signals = bathreset_rabicheck_exp.run(soc, soccfg, cfg)
```

```python
bathreset_rabicheck_exp.save(
    filepath=os.path.join(
        database_path, f"{qub_name}_bathreset_rabicheck@{cur_A * 1e3:.3f}mA"
    ),
    # filepath=os.path.join(database_path, f"{qub_name}_bathreset_rabicheck@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## Flux Dependence


### Frequency

```python
cur_A = 0.5e-3
1e3 * flux_yoko.set_current(cur_A)
# cur_V = 1.75
# flux_yoko.set_voltage(cur_V)
```

```python
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": {
        "waveform": ml.get_waveform("qub_flat", override_cfg={"length": 1.0}),
        "ch": qub_5_6_ch,
        "nqz": 1,
        "gain": 0.001,
        # "mixer_freq": q_f,
    },
    "readout": "readout_bare_rf",
    "dev": {
        "flux_yoko": {
            "label": "flux_dev",
            # "mode": "voltage",
            "mode": "current",
        },
    },
    "sweep": {
        "flux": make_sweep(1.2e-3, 1.5e-3, 101),
        # "flux": make_sweep(-4.0, -3.0, 201),
        "freq": make_sweep(5700, 5900, step=0.25),
    },
    "relax_delay": 1.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

qub_flux_exp = ze.twotone.flux_dep.FreqExperiment()
As, fpts, signals2D = qub_flux_exp.run(
    soc,
    soccfg,
    cfg,
    method="yoko",
)
```

```python
qub_flux_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_flux"),
    # filepath=os.path.join(database_path, f"{qub_name}_flux@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_flux@{cur_V:.3f}V"),
    comment=make_comment(cfg),
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
%matplotlib widget
point_finder = qub_flux_exp.extract_points()
```

```python
freq_map = point_finder.get_positions()
```

### Auto Measure


#### T1

```python
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": {
        "waveform": ml.get_waveform("qub_flat", override_cfg={"length": 3.0}),
        "ch": qub_1_4_ch,
        "nqz": 1,
        "gain": 0.03,
        "mixer_freq": q_f,
    },
    "pi_pulse": "pi_amp",
    "readout": "readout_dpm",
    "dev": {
        "flux_yoko": {
            "label": "flux_dev",
            # "mode": "voltage",
            "mode": "current",
        },
    },
    "sweep": {
        # "flux": make_sweep(-1.0e-3, 1.0e-3, 51),
        "flux": make_sweep(preditor.flx_to_A(0.6), preditor.flx_to_A(0.53), 51),
        "detune": make_sweep(-12.0, 12.0, step=0.2),
        "rabi_length": make_sweep(0.03, 1.6, 31),
        "t1_length": make_sweep(0.1, 100.0, 51),
    },
    "relax_delay": 80.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=50)

t1_flux_exp = ze.twotone.flux_dep.T1Experiment()
values, detunes, rabilens, t1lens, signals_dict, fig = t1_flux_exp.run(
    soc,
    soccfg,
    cfg,
    predictor=preditor,
    # ref_flux=cur_V,
    ref_flux=cur_A,
    earlystop_snr=20,
)

savefig(
    fig,
    f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_t1_over_flux.png",
)
plt.close(fig)
```

```python
%matplotlib inline
values, t1s, t1errs, freqs = t1_flux_exp.analyze(start_idx=0, snr_threshold=5)
```

```python
t1_flux_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_t1_flux@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_t1_flux@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

```python
from datetime import datetime

sample_table.extend_samples(
    **{
        "calibrated mA": values,
        "Freq (MHz)": freqs,
        "T1 (us)": t1s,
        "T1err (us)": t1errs,
        "comment": f"Auto measured on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    }
)
```

#### T2Ramsey

```python
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": {
        "waveform": ml.get_waveform("qub_flat", override_cfg={"length": 5.0}),
        "ch": qub_0_1_ch,
        "nqz": 1,
        "gain": 0.02,
        "mixer_freq": q_f,
    },
    "pi2_pulse": "pi2_len",
    "readout": "readout_rf",
    "dev": {
        "flux_yoko": {
            "label": "flux_dev",
            "mode": "voltage",
        },
    },
    "sweep": {
        "flux": make_sweep(-1.0e-3, 1.0e-3, 51),
        # "flux": make_sweep(1.19, 1.22, 51),
        "detune": make_sweep(-12, 12, step=0.5),
        "length": make_sweep(0.0, 40.0, 101),
    },
    "relax_delay": 50.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=30)

activate_detune = 0.1 / cfg["sweep"]["length"]["step"]
print(f"activate_detune: {activate_detune:.2f}")

t2r_flux_exp = ze.twotone.flux_dep.T2RamseyExperiment()
values, detunes, lens, signals_dict, fig = t2r_flux_exp.run(
    soc,
    soccfg,
    cfg,
    predictor=preditor,
    activate_detune=activate_detune,
    # ref_flux=cur_V,
    ref_flux=cur_A,
    earlystop_snr=5,
)

savefig(
    fig,
    f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_t2r_over_flux.png",
)
plt.close(fig)
```

```python
%matplotlib inline
_ = t2r_flux_exp.analyze(freq_map=freq_map, activate_detune=activate_detune)
```

```python
t2r_flux_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_t2r_flux@{cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_t2r_flux@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

#### Mist

```python
mist_pulse_len = 0.5  # us
ml.register_waveform(
    mist_waveform={
        "style": "const",
        "length": mist_pulse_len,  # us
    }
)
```

```python
exp_cfg = {
    "qub_pulse": {
        "waveform": ml.get_waveform("qub_flat", override_cfg={"length": 3.0}),
        "ch": qub_1_4_ch,
        "nqz": 2,
        "gain": 0.1,
        "mixer_freq": q_f,
    },
    "pi_pulse": "pi_amp",
    "probe_pulse": {
        "waveform": ml.get_waveform("mist_waveform"),
        "ch": res_ch,
        "nqz": 2,
        # "freq": r_f,
        "freq": 5927.0,
        "post_delay": 10 / (2 * np.pi * rf_w),
    },
    "readout": "readout_dpm",
    "dev": {
        "flux_yoko": {
            "label": "flux_dev",
            # "mode": "voltage",
            "mode": "current",
        },
    },
    "sweep": {
        "flux": make_sweep(0.9409e-3, 0.9412e-3, 101),
        # "flux": make_sweep(preditor.flx_to_A(0.6), preditor.flx_to_A(0.8), 201),
        "detune": make_sweep(-15.0, 15.0, step=0.2),
        "rabi_length": make_sweep(0.03, 2.0, 101),
        "gain": make_sweep(0.0, 1.0, 201),
    },
    "relax_delay": 10.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=50)

mist_flux_exp = ze.twotone.flux_dep.AutoMistExperiment()
values, detunes, gains, signals_dict, fig = mist_flux_exp.run(
    soc,
    soccfg,
    cfg,
    predictor=preditor,
    # ref_flux=cur_V,
    ref_flux=cur_A,
    earlystop_snr=20,
)

savefig(
    fig,
    f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_automist_over_flux.png",
)
plt.close(fig)
```

```python
fig = mist_flux_exp.analyze()
plt.show(fig)
savefig(
    fig,
    f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_automist_over_flux_only_ge.png",
)
plt.close(fig)
```

```python
mist_flux_exp.save(
    filepath=os.path.join(
        database_path, f"{qub_name}_automist_flux@{cur_A * 1e3:.3f}mA"
    ),
    # filepath=os.path.join(database_path, f"{qub_name}_automist_flux@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## Power dependence

```python
exp_cfg = {
    # "reset": "reset_120",
    # "qub_pulse": {
    #     **ml.get_waveform("qub_waveform"),
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
    "sweep": {
        "gain": make_sweep(0.05, 1.0, 30),
        # "freq": make_sweep(1700, 2000, 30),
        "freq": make_sweep(q_f - 3, q_f + 3, 101),
    },
    "relax_delay": 10.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

qub_pdr_exp = ze.twotone.PowerDepExperiment()
fpts, pdrs, signals2D = qub_pdr_exp.run(soc, soccfg, cfg)
```

```python
qub_pdr_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_pdr@{cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
)
```

## Dispersive Shift

```python
exp_cfg = {
    "reset": "reset_10",
    "qub_pulse": "pi_amp",
    "readout": ml.get_module(
        "readout_rf",
        {
            "pulse_cfg": {
                "gain": 0.1,
            }
        },
    ),
    "sweep": make_sweep(r_f - 20, r_f + 20, step=0.25),
    "relax_delay": 1.0,  # us
    # "relax_delay": 2 * t1, # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=1000)

dispersive_shift_exp = ze.twotone.DispersiveExperiment()
fpts, signals = dispersive_shift_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
chi, rf_w, fig = dispersive_shift_exp.analyze()
plt.show(fig)
savefig(
    fig,
    f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_dispersive_shift.png",
)
plt.close(fig)
```

```python
dispersive_shift_exp.save(
    filepath=os.path.join(database_path, f"{res_name}_dispersive@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{res_name}_dispersive@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"chi = {chi:.3g} MHz, kappa = {rf_w:.3g} MHz"),
)
```

## AC Stark Shift

```python
%matplotlib widget
ac_qub_len = 2.5  # us
exp_cfg = {
    "reset": "reset_10",
    "stark_pulse1": {
        "waveform": ml.get_waveform(
            "mist_waveform", {"length": 5.1 / (2 * np.pi * rf_w) + ac_qub_len}
        ),
        "ch": res_ch,
        "nqz": 2,
        "freq": r_f,
        "block_mode": False,
    },
    "stark_pulse2":  {
        "waveform": ml.get_waveform("qub_flat", override_cfg={"length": ac_qub_len}),
        "ch": qub_1_4_ch,
        "nqz": 2,
        "gain": 0.1,
        # "mixer_freq": q_f,
        "mixer_freq": 0.5 * (reset_f + q_f),
        "pre_delay": 5.0 / (2 * np.pi * rf_w),
        "post_delay": 3.1 / (2 * np.pi * rf_w),
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": {
        "gain": make_sweep(0.01, 0.5, 101),
        "freq": make_sweep(q_f - 100, q_f + 10, step=0.5),
    },
    "relax_delay": 0.1,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

ac_stark_exp = ze.twotone.AcStarkExperiment()
pdrs, fpts, signals2D = ac_stark_exp.run(soc, soccfg, cfg, earlystop_snr=20)
```

```python
%matplotlib inline
ac_stark_coeff, fig = ac_stark_exp.analyze(chi=chi, kappa=rf_w, deg=1, cutoff=0.35)
savefig(fig, f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_ac_stark.png")
plt.close(fig)
```

```python
ac_stark_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_ac_stark@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_ac_stark@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"ac_stark_coeff = {ac_stark_coeff:.3g} MHz"),
)
```

## All XY

```python
exp_cfg = {
    # "reset": "reset_120",
    "X180_pulse": "pi_len",
    "X90_pulse": "pi2_len",
    "Y180_pulse": ml.get_module("pi_len", {"phase": 90}),
    "Y90_pulse": ml.get_module("pi2_len", {"phase": 90}),
    "readout": "readout_rf",
    # "readout": "readout_dpm",
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

allxy_exp = ze.twotone.AllXYExperiment()
signals_dict = allxy_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
allxy_exp.analyze()
```

```python
allxy_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_allxy@{cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_allxy@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## Zig-Zag

```python
exp_cfg = {
    # "reset": "reset_120",
    "X90_pulse": "pi2_len",
    "X180_pulse": "pi_len",
    "readout": "readout_rf",
    # "readout": "readout_dpm",
    "sweep": list(range(0, 11)),
    "relax_delay": 30.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

zigzag_exp = ze.twotone.ZigZagExperiment()
times, signals = zigzag_exp.run(soc, soccfg, cfg, repeat_on="X180_pulse")
```

```python
zigzag_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_zigzag@{cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_zigzag@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

### Sweep parameters

```python
exp_cfg = {
    # "reset": "reset_120",
    "X90_pulse": "pi2_amp",
    "X180_pulse": "pi_amp",
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": {
        "times": list(range(0, 7)),
        "gain": make_sweep(max(0.0, pi_gain - 0.02), min(1.0, pi_gain + 0.02), 31),
        # "gain": make_sweep(max(0.0, pi2_gain - 0.02), min(1.0, pi2_gain + 0.02), 31),
        # "freq": make_sweep(q_f - 1, q_f + 1, 31),
    },
    "relax_delay": 30.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=30)

zigzag_sweep_exp = ze.twotone.ZigZagSweepExperiment()
times, values, signals = zigzag_sweep_exp.run(soc, soccfg, cfg, repeat_on="X180_pulse")
```

```python
best_x = zigzag_sweep_exp.analyze()
```

```python
zigzag_sweep_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_zigzag_sweep@{cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_zigzag_sweep@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

```python
pi_gain = best_x
# pi2_gain = best_x
ml.update_module(
    name="pi_amp",
    override_cfg={"gain": pi_gain},
    # name="pi2_amp",
    # override_cfg={"gain": pi2_gain}
)
```

# Rabi

```python

```

## Length Rabi

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": {
        "waveform": ml.get_waveform("qub_flat"),
        "ch": qub_1_4_ch,
        "nqz": 2,
        "freq": q_f,
        "gain": 0.6,
        # "gain": pi_gain,
        "mixer_freq": q_f,
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "relax_delay": 20.0,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.03, 1.0, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

qub_lenrabi_exp = ze.twotone.LenRabiExperiment()
Ts, signals = qub_lenrabi_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
pi_len, pi2_len, rabi_f, fig = qub_lenrabi_exp.analyze(decay=True, max_contrast=True)
pi_len, pi2_len, rabi_f
```

```python
savefig(
    fig,
    f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_rabi_length.png",
)
plt.close(fig)
qub_lenrabi_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_len_rabi@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_len_rabi@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"pi len = {pi_len}us\npi/2 len = {pi2_len}us"),
)
```

```python
# pi_len = 1.0
# pi2_len = 0.5
ml.register_module(
    pi_len={
        **exp_cfg["qub_pulse"],
        "waveform": {**exp_cfg["qub_pulse"]["waveform"], "length": pi_len},
        "pre_delay": 0.005,
        "post_delay": 0.005,
        "desc": "len pi pulse",
    },
    pi2_len={
        **exp_cfg["qub_pulse"],
        "waveform": {**exp_cfg["qub_pulse"]["waveform"], "length": pi2_len},
        "pre_delay": 0.005,
        "post_delay": 0.005,
        "desc": "len pi/2 pulse",
    },
)
```

## Amplitude Rabi

```python
%matplotlib widget
pi_gain = ml.get_module("pi_len")["gain"]
max_gain = min(5 * pi_gain, 1.0)
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": {
        "waveform": ml.get_waveform(
            "qub_flat",
            override_cfg={"length": 2.0 * pi_len},
        ),
        "ch": qub_1_4_ch,
        "nqz": 2,
        "freq": q_f,
        # "mixer_freq": q_f,
        "mixer_freq": 0.5 * (reset_f + q_f),
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "relax_delay": 20.0,  # us
    # "relax_delay": 5 * t1,
    "sweep": make_sweep(0.01, 1.0, 51),
    # "sweep": make_sweep(0.0, max_gain, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

qub_amprabi_exp = ze.twotone.AmpRabiExperiment()
pdrs, signals = qub_amprabi_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
pi_gain, pi2_gain, fig = qub_amprabi_exp.analyze(decay=False, max_contrast=True)
pi_gain, pi2_gain
```

```python
savefig(
    fig,
    f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_rabi_amplitude.png",
)
plt.close(fig)
qub_amprabi_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_amp_rabi@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_amp_rabi@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"pi gain = {pi_gain}\npi/2 gain = {pi2_gain}"),
)
```

```python
# pi_gain = 1.0
# pi2_gain = 0.5
ml.register_module(
    pi_amp={
        **cfg["qub_pulse"],
        "gain": pi_gain,
        "pre_delay": 0.005,
        "post_delay": 0.005,
        "desc": "amp pi pulse",
    },
    pi2_amp={
        **cfg["qub_pulse"],
        "gain": pi2_gain,
        "pre_delay": 0.005,
        "post_delay": 0.005,
        "desc": "amp pi/2 pulse",
    },
)
```

# Optimize Readout

```python

```

## Frequency tuning

```python
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": "pi_amp",
    "readout": ml.get_module(
        "readout_rf",
        override_cfg={
            "pulse_cfg": {
                "waveform": {"length": 1.5},
                "gain": 0.2,
                # "gain": pdr_max,
            },
            "ro_cfg": {
                "ro_length": 1.4,
            },
        },
    ),
    "relax_delay": 10.0,  # us
    # "relax_delay": 3 * t1,  # us
    # "sweep": make_sweep(r_f - 3, r_f + 3, 101),
    "sweep": make_sweep(5920, 5940, step=0.25),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=50)

opt_ro_freq_exp = ze.twotone.ro_optimize.OptimizeFreqExperiment()
fpts, snrs = opt_ro_freq_exp.run(soc, soccfg, cfg)
```

```python
fpt_max = opt_ro_freq_exp.analyze(smooth=1)
```

```python
opt_ro_freq_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_ro_opt_freq@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_ro_opt_freq@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"optimal frequency = {fpt_max:.1f}MHz"),
)
```

## Power tuning

```python
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": "pi_amp",
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

```python
opt_ro_pdr_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_ro_opt_pdr@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_ro_opt_pdr@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"optimal power = {pdr_max:.2f}"),
)
```

## Readout Length tuning

```python
# pdr_max = 0.6
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": "pi_amp",
    "readout": ml.get_module(
        "readout_bare_rf",
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
ro_max = opt_ro_len_exp.analyze(t0=3.0)
ro_max
```

```python
opt_ro_len_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_ro_opt_len@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_ro_opt_len@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"optimal readout length = {ro_max:.2f}us"),
)
```

```python
# ro_max = 1.5
ml.register_module(
    readout_dpm=ml.get_module(
        "readout_bare_rf",
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
exp_cfg = {
    "reset": "reset_10",
    "pi2_pulse": "pi2_amp",
    "readout": "readout_dpm",
    "relax_delay": 0.1,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0, 2.0, 101),  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

# activate_detune = 1.5
activate_detune = 0.1 / cfg["sweep"]["step"]
print(f"activate_detune: {activate_detune:.2f}")

t2ramsey_exp = ze.twotone.time_domain.T2RamseyExperiment()
Ts, signals = t2ramsey_exp.run(soc, soccfg, cfg, detune=activate_detune)
```

```python
%matplotlib inline
t2r, _, detune, _, fig = t2ramsey_exp.analyze(max_contrast=True)
print(f"real detune: {(detune - activate_detune) * 1e3:.1f}kHz")
```

```python
savefig(
    fig, f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_t2ramsey.png"
)
plt.close(fig)
t2ramsey_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_t2ramsey@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_t2ramsey@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"detune = {detune:.3f}MHz\nt2r = {t2r:.3f}us"),
)
```

```python
q_f = ml.get_module("pi2_amp")["freq"] + activate_detune - detune
q_f
```

## T1

```python
exp_cfg = {
    # "reset": "reset_120",
    "pi_pulse": "pi_amp",
    # "pi_pulse": {
    #     "waveform": ml.get_waveform("qub_flat", override_cfg={"length": 15.0}),
    #     "ch": qub_1_4_ch,
    #     "nqz": 1,
    #     "gain": 0.5,
    #     "freq": q_f,
    #     "mixer_freq": q_f,
    # },
    "readout": "readout_rf",
    # "readout": "readout_dpm",
    "relax_delay": 10.0,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.0, 7.1, 51),
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

```python
os.makedirs(f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/", exist_ok=True)
fig.savefig(f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_t1.png")
plt.close(fig)
```

```python
t1_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_t1@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_t1@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"t1 = {t1:.3f}us"),
)
```

```python
sample_table.add_sample(
    **{
        "calibrated mA": cur_A,
        "Freq (MHz)": q_f,
        "T1 (us)": t1,
        "T1err (us)": t1err,
        "comment": "Manual Added",
    }
)
```

### With Tone

```python
exp_cfg = {
    # "reset": "reset_120",
    "pi_pulse": "pi_amp",
    "test_pulse": {
        **ml.get_module("readout_dpm")["pulse_cfg"],
        "pre_delay": rf_w / 5.0,
        "post_delay": rf_w / 5.0,
    },
    "readout": "readout_dpm",
    "relax_delay": 50.0,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(1.0, 10, 101),
    # "sweep": make_sweep(0.01*t1, 5 * t1, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

t1_with_tone_exp = ze.twotone.time_domain.T1WithToneExperiment()
Ts, signals = t1_with_tone_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
t1_with_tone, _ = t1_with_tone_exp.analyze(max_contrast=True, dual_exp=False)
t1_with_tone
```

```python
t1_with_tone_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_t1@{cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_t1_with_tone@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"t1 = {t1_with_tone:.3f}us"),
)
```

### With Sweep Tone

```python
exp_cfg = {
    # "reset": "reset_120",
    "pi_pulse": "pi_amp",
    "test_pulse": {
        "waveform": {
            "style": "flat_top",
            "raise_waveform": {"style": "cosine", "length": 1 / rf_w},
        },
        "ch": res_ch,
        "nqz": 2,
        "freq": r_f,
        "pre_delay": 5.0 / rf_w,
        "post_delay": 5.0 / rf_w,
    },
    "readout": "readout_dpm",
    "relax_delay": 40.0,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": {
        "gain": make_sweep(0.0, 0.2, 51),
        "length": make_sweep(1.0, 20, 51),
    },
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

t1_with_tone_sweep_exp = ze.twotone.time_domain.T1WithToneSweepExperiment()
values, Ts, signals = t1_with_tone_sweep_exp.run(soc, soccfg, cfg)
```

```python
_ = t1_with_tone_sweep_exp.analyze()
```

```python
t1_with_tone_sweep_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_t1@{cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_t1_with_tone_sweep@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"t1 = {t1_with_tone:.3f}us"),
)
```

## T2Echo

```python
exp_cfg = {
    # "reset": "reset_120",
    "pi_pulse": "pi_amp",
    "pi2_pulse": "pi2_amp",
    "readout": "readout_dpm",
    "relax_delay": 5.0,  # us
    # "relax_delay": 5 * t1,  # us
    # "sweep": make_sweep(0.0, 5 * t2r, 51),
    # "sweep": make_sweep(0.0, 1.5 * t2e, 101),
    "sweep": make_sweep(0.01, 3.0, 101),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

activate_detune = 0.1 / cfg["sweep"]["step"]
print(f"activate_detune: {activate_detune:.2f}")

t2echo_exp = ze.twotone.time_domain.T2EchoExperiment()
Ts, signals = t2echo_exp.run(soc, soccfg, cfg, detune=activate_detune)
```

```python
%matplotlib inline
t2e, _, detune, _, fig = t2echo_exp.analyze(max_contrast=True, fit_method="fringe")
```

```python
os.makedirs(f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/", exist_ok=True)
fig.savefig(
    f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_t2echo.png"
)
plt.close(fig)
```

```python
t2echo_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_t2echo@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_t2echo@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"detune = {detune:.3f}MHz\nt2echo = {t2e:.3f}us"),
)
```

## CPMG

```python
ml.get_module("pi_len")["waveform"]["length"]
```

```python
exp_cfg = {
    # "reset": "reset_120",
    "pi_pulse": "pi_len",
    "pi2_pulse": {
        **ml.get_module("pi2_len"),
        "phase": 90,  # Y/2 gate
    },
    "readout": "readout_rf",
    "relax_delay": 50.0,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": {
        "times": list(range(1, 101, 10)),
        "length": make_sweep(0.0, 15, 31),
    },
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=30)

cpmg_exp = ze.twotone.time_domain.CPMGExperiment()
times, Ts, signals = cpmg_exp.run(soc, soccfg, cfg)
```

```python
_, _, fig = cpmg_exp.analyze()
```

```python
cpmg_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_t2echo@{cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_cpmg@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

# Single shot

```python

```

## Ground state & Excited state

```python
exp_cfg = {
    # "reset": "reset_120",
    "probe_pulse": "pi_amp",
    "readout": ml.get_module(
        "readout_dpm",
        # {
        #     "pulse_cfg": {
        #         "waveform": {
        #             # "length": 0.2 * t1 + 0.1,
        #             "length": 10.0
        #         },
        #         # "gain": 0.2,
        #     },
        #     "ro_cfg": {
        #         # "ro_length": 0.2 * t1,
        #         "ro_length": 10.0,
        #     },
        # },
    ),
    "relax_delay": 10.0,  # us
}
cfg = ml.make_cfg(exp_cfg, shots=100000)
print("readout length: ", cfg["readout"]["ro_cfg"]["ro_length"])

singleshot_exp = ze.twotone.SingleShotExperiment()
signals = singleshot_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
fid, _, _, pops, _, fig = singleshot_exp.analyze(
    backend="pca",
    # init_p0=0.0,
    # length_ratio=cfg["readout"]["ro_cfg"]["ro_length"] / t1,
    # logscale=True,
)
print(f"Optimal fidelity after rotation = {fid:.1%}")
```

```python
os.makedirs(f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/", exist_ok=True)
fig.savefig(
    f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_singleshot.png"
)
plt.close(fig)
```

```python
singleshot_exp.save(
    filepath=os.path.join(
        database_path, f"{qub_name}_singleshot_ge@{cur_A * 1e3:.3f}mA"
    ),
    # filepath=os.path.join(database_path, f"{qub_name}_singleshot_ge@{cur_V:.3f}V"),
    comment=make_comment(cfg, f"fide: {fid:.1%}"),
)
```

```python
from zcu_tools.simulate.temp import effective_temperature

n_g = pops[0][0]  # n_gg
n_e = pops[0][1]  # n_ge

n_g, n_e = (n_g, n_e) if n_g > n_e else (n_e, n_g)  # ensure n_g >= n_e

eff_T, err_T = effective_temperature(population=[(n_g, 0.0), (n_e, q_f)])
eff_T, err_T
```

# MIST

```python
mist_pulse_len = 0.5  # us
ml.register_waveform(
    mist_waveform={
        "style": "const",
        "length": mist_pulse_len,  # us
    }
)
```

## Flux dependence

```python
%matplotlib widget
exp_cfg = {
    # "init_pulse": "pi_amp",
    "probe_pulse": {
        "waveform": ml.get_waveform("mist_waveform"),
        "ch": res_ch,
        "nqz": 2,
        # "freq": r_f,
        "freq": 5927,
        "post_delay": 10 / (2 * np.pi * rf_w),
    },
    "readout": "readout_dpm",
    "dev": {
        "flux_yoko": {
            "label": "flux_dev",
            # "mode": "voltage",
            "mode": "current",
        },
    },
    "sweep": {
        # "flux": make_sweep(0.9409e-3, 0.9411e-3, 1001),
        "flux": make_sweep(preditor.flx_to_A(0.4), preditor.flx_to_A(1.1), 1001),
        "gain": make_sweep(0.0, 1.0, 101),
    },
    "relax_delay": 10.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=30)

mist_flux_exp = ze.twotone.flux_dep.MistExperiment()
values, gains, signals = mist_flux_exp.run(soc, soccfg, cfg)
```

```python
fig = mist_flux_exp.analyze()
savefig(
    fig,
    f"../result/{chip_name}/exp_image/{cur_A * 1e3:.3f}mA/{qub_name}_mist_over_flux_bare.png",
)
```

```python
mist_flux_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_mist_flux_bare@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_mist_flux@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## Single Trace

```python
# cur_A = 1.162e-3
flux_yoko.set_current(current=cur_A)
# cur_V = 0.0
# flux_yoko.set_voltage(voltage=cur_V)
```

```python
exp_cfg = {
    "init_pulse": "pi_amp",
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

```python
mist_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_mist@{cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_mist@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## Overnight

```python
exp_cfg = {
    "init_pulse": "pi_amp",
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
    "interval": 300,  # s
}
cfg = ml.make_cfg(exp_cfg, reps=10000, rounds=3)

mist_overnight_exp = ze.twotone.mist.MISTPowerDepOvernight()
iters, gains, signals = mist_overnight_exp.run(
    soc, soccfg, cfg, num_times=120, fail_retry=2
)
```

```python
mist_overnight_exp.save(
    filepath=os.path.join(
        database_path, f"{qub_name}_mist_overnight@{cur_A * 1e3:.3f}mA"
    ),
    # filepath=os.path.join(database_path, f"{qub_name}_mist_overnight@{cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

```python

```
