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
    display_name: zcu-tools
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
    version: 3.9.25
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
from zcu_tools.table import SampleTable, MetaDict
from zcu_tools.notebook.utils import make_sweep, make_comment, savefig
from zcu_tools.utils.datasaver import create_datafolder
```

# Create data folder and cfg

```python
chip_name = "Q12_2D[6]"
res_name = "R1"
qub_name = "Q1"

result_dir = f"../result/{chip_name}/{qub_name}/"

database_path = create_datafolder(
    os.path.join(os.getcwd(), ".."), prefix=os.path.join(chip_name, qub_name)
)
ml = ModuleLibrary(cfg_path=f"{result_dir}/module_cfg.yaml")
md = MetaDict(f"{result_dir}/meta_info.json")
```

# Connect to zcu216

```python
from zcu_tools.remote import make_proxy

soc, soccfg = make_proxy("192.168.10.179", 8887)
print(soccfg)
```

# Predefine parameters

```python
res_ch = 0
qub_0_1_ch = 11
qub_1_4_ch = 2
# qub_4_5_ch = 5
# qub_5_6_ch = 5
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


### Qubit Flux

```python
from zcu_tools.device.yoko import YOKOGS200


flux_yoko = YOKOGS200(
    address="USB0::0x0B21::0x0039::90ZB35281::INSTR", rm=resource_manager
)
GlobalDeviceManager.register_device("flux_yoko", flux_yoko)

flux_yoko.set_mode("current", rampstep=1e-6)
# flux_yoko.set_mode("voltage", rampstep=1e-3)
```

```python
md.cur_A = flux_yoko.get_current()
md.cur_A * 1e3
# md.cur_V = flux_yoko.get_voltage()
# md.cur_V
```

```python
# md.cur_A = 0.0e-3
flux_yoko.set_current(current=md.cur_A)
# md.cur_V = 0.0
# flux_yoko.set_voltage(voltage=md.cur_V)
```

### JPA Flux

```python
from zcu_tools.device.yoko import YOKOGS200


jpa_yoko = YOKOGS200(
    address="USB0::0x0B21::0x0039::91T810992::INSTR", rm=resource_manager
)
GlobalDeviceManager.register_device("jpa_yoko", jpa_yoko)

jpa_yoko.set_mode("current", rampstep=1e-6)
```

```python
md.cur_jpa_A = jpa_yoko.get_current()
md.cur_jpa_A
```

```python
# md.cur_jpa_A = 0.0e-3
jpa_yoko.set_current(current=md.cur_jpa_A)
```

## RF Source


### JPA Pump

```python
from zcu_tools.device.sgs100a import RohdeSchwarzSGS100A

jpa_sgs = RohdeSchwarzSGS100A(
    address="TCPIP0::192.168.10.89::inst0::INSTR", rm=resource_manager
)
GlobalDeviceManager.register_device("jpa_sgs", jpa_sgs)
```

```python
# jpa_sgs.set_power(-20)  # dBm
# jpa_sgs.IQ_off()
# jpa_sgs.output_on()
jpa_sgs.output_off()
```

```python
jpa_sgs.get_info()
```

# Lookback

```python
md.timeFly = 0.6
```

```python
%matplotlib widget
exp_cfg = {
    "readout": {
        "type": "base",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": 1.0},
            # "waveform": {
            #     "style": "padding",
            #     "length": 1.03,
            #     "pre_length": 0.01,
            #     "post_length": 0.005,
            #     "pre_gain": 1.0,
            #     "post_gain": -1.0,
            # },
            "ch": res_ch,
            "nqz": 2,
            "gain": 1.0,
            "freq": 5930,
            # "freq": r_f,
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": 1.5,  # us
            "trig_offset": md.timeFly - 0.1,  # us
        },
    },
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, rounds=500)

lookback_exp = ze.LookbackExp()
_ = lookback_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
predict_offset, fig = lookback_exp.analyze(
    ratio=0.1, smooth=1.0, ro_cfg=cfg["readout"]["ro_cfg"]
)
predict_offset
```

```python
md.timeFly = float(predict_offset)
md.timeFly
```

```python
filename = "lookback"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
lookback_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg, f"timeFly = {md.timeFly}us"),
)
```

# OneTone

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
            "gain": 0.05,
            # "gain": 1.0,
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": res_probe_len - 0.1,  # us
            "trig_offset": md.timeFly + 0.05,  # us
        },
    },
    # "sweep": make_sweep(5350, 5355, 101),
    "sweep": make_sweep(md.r_f - 2 * md.rf_w, md.r_f + 2 * md.rf_w, 101),
    "relax_delay": 1.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

res_freq_exp = ze.onetone.FreqExp()
fpts, signals = res_freq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
f, kappa, params, fig = res_freq_exp.analyze(model_type="auto")
```

```python
md.r_f = f
md.rf_w = kappa
```

```python
filename = f"{res_name}_freq"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
res_freq_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
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
            "trig_offset": md.timeFly + 0.05,  # us
        },
    },
    "sweep": {
        "gain": make_sweep(0.001, 0.3, 101),
        "freq": make_sweep(md.r_f - 1.5 * md.rf_w, md.r_f + 1.5 * md.rf_w, 201),
    },
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

res_pdr_exp = ze.onetone.PowerDepExp()
pdrs, fpts, signals2D = res_pdr_exp.run(soc, soccfg, cfg, earlystop_snr=100.0)
```

```python
res_pdr_exp.save(
    filepath=os.path.join(database_path, f"{res_name}_pdr@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{res_name}_pdr@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## Flux dependence

```python
md.cur_A = 1.561e-3
1e3 * flux_yoko.set_current(md.cur_A)
# md.cur_V = 0.0
# flux_yoko.set_voltage(md.cur_V)
```

```python
%matplotlib widget
exp_cfg = {
    "readout": {
        "type": "base",
        "pulse_cfg": {
            "waveform": ml.get_waveform("ro_waveform"),
            "ch": res_ch,
            "nqz": 2,
            "gain": 0.01,
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": res_probe_len - 0.1,  # us
            "trig_offset": md.timeFly + 0.05,  # us
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
        "flux": make_sweep(4.0e-3, -3.0e-3, 301),
        # "flux": make_sweep(5.0, -5.0, 101),
        "freq": make_sweep(md.r_f - 2 * md.rf_w, md.r_f + 2 * md.rf_w, 101),
    },
    "relax_delay": 0.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=1)

res_flux_exp = ze.onetone.FluxDepExp()
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
md.mA_c, md.mA_e = actline.get_positions()
md.mA_c, md.mA_e, 2 * abs(md.mA_e - md.mA_c)
```

```python
# md.cur_A = md.mA_e
md.cur_A = (0.84 - 0.5) * (md.mA_e - md.mA_c) / 0.5 + md.mA_c
1e3 * flux_yoko.set_current(md.cur_A)
# md.cur_V = mA_c
# flux_yoko.set_voltage(md.cur_V)
```

## Set readout pulse

```python
ro_pulse_len = 1.0  # us
ml.register_module(
    readout_rf={
        "type": "base",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": ro_pulse_len},
            "ch": res_ch,
            "nqz": 2,
            "freq": md.r_f,
            # "freq": 5927,
            "gain": 0.1,
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_length": ro_pulse_len - 0.1,  # us
            "trig_offset": md.timeFly + 0.01,  # us
        },
        "desc": "lower power readout with exact resonator frequency",
    }
)
```

# JPA

```python
jpa_sgs.IQ_off()
jpa_sgs.set_power(-15)  # dBm
jpa_sgs.set_frequency(1e6 * (md.r_f * 2 + 200))  # Hz
jpa_sgs.output_on()
jpa_sgs.get_info()
```

## JPA Flux by Onetone

```python
%matplotlib widget
exp_cfg = {
    "readout": "readout_rf",
    "dev": {
        "jpa_yoko": {
            "label": "jpa_flux_dev",
            "mode": "current",
        },
    },
    "sweep": {
        "jpa_flux": make_sweep(-7.0e-3, 7.0e-3, 301),
        # "freq": make_sweep(md.r_f - 2 * md.rf_w, md.r_f + 2 * md.rf_w, 101),
        "freq": make_sweep(5000, 5500, 501),
    },
    "relax_delay": 0.1,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=10)


jpa_flux_onetone_exp = ze.jpa.JPAFluxByOneToneExp()
flxs, fpts, signals2D = jpa_flux_onetone_exp.run(soc, soccfg, cfg)
```

```python
jpa_flux_onetone_exp.save(
    filepath=os.path.join(database_path, "JPA_flux_onetone"),
    comment=make_comment(cfg),
)
```

```python
jpa_yoko.set_current(0.8e-3)
```

## JPA frequency

```python
%matplotlib widget
exp_cfg = {
    "reset": "reset_bath",
    "pi_pulse": "pi_len",
    "readout": "readout_rf",
    "dev": {"jpa_sgs": {"label": "jpa_rf_dev"}},
    # "sweep": make_sweep(2 * md.r_f + 100, 2 * md.r_f + 500, step=0.25),
    "sweep": make_sweep(11750, 11800, 501),
    "relax_delay": 0.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=10000, rounds=1)

jpa_freq_exp = ze.jpa.JPAFreqExperiment()
jpa_freqs, signals2D = jpa_freq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
md.best_jpa_freq, fig = jpa_freq_exp.analyze()
md.best_jpa_freq
```

```python
filename = "JPA_freq"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
jpa_freq_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
)
```

```python
jpa_sgs.set_frequency(1e6 * md.best_jpa_freq)  # Hz
```

## JPA Flux

```python
jpa_sgs.get_info()
```

```python
%matplotlib widget
exp_cfg = {
    "reset": "reset_bath",
    "pi_pulse": "pi_amp",
    "readout": "readout_rf",
    "dev": {
        "jpa_yoko": {
            "label": "jpa_flux_dev",
            "mode": "current",
        },
    },
    "sweep": make_sweep(-5.0e-3, 5.0e-3, 1001),
    "relax_delay": 0.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=10000, rounds=1)

jpa_flux_exp = ze.jpa.JPAFluxExperiment()
flxs, signals2D = jpa_flux_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
md.best_jpa_flux, fig = jpa_flux_exp.analyze()
md.best_jpa_flux * 1e3
```

```python
filename = "JPA_flux"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
jpa_flux_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
)
```

```python
md.cur_jpa_A = jpa_yoko.set_current(md.best_jpa_flux)
md.cur_jpa_A
```

## JPA Power

```python
%matplotlib widget
exp_cfg = {
    "reset": "reset_bath",
    "pi_pulse": "pi_len",
    "readout": "readout_rf",
    "dev": {
        "jpa_sgs": {
            "label": "jpa_rf_dev",
        }
    },
    "sweep": make_sweep(-20, 1, 501),
    "relax_delay": 0.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=10000, rounds=1)

jpa_pdr_exp = ze.jpa.JPAPowerExperiment()
pdrs, signals2D = jpa_pdr_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
md.best_jpa_power, fig = jpa_pdr_exp.analyze()
md.best_jpa_power
```

```python
filename = "JPA_power"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
jpa_pdr_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
)
```

```python
jpa_sgs.set_power(md.best_jpa_power)  # dBm
```

## Auto Optimize

```python
1e3 * jpa_yoko.set_current(-4.0e-3)
```

```python
import gc

plt.close("all")
gc.collect()
```

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_bath",
    "pi_pulse": "pi_amp",
    "readout": "readout_rf",
    "dev": {
        "jpa_sgs": {"label": "jpa_rf_dev", "output": "on"},
        "jpa_yoko": {
            "label": "jpa_flux_dev",
            "mode": "current",
        },
    },
    "sweep": {
        "jpa_flux": make_sweep(-4e-3, 2e-3, 100),
        "jpa_freq": make_sweep(2 * md.r_f + 50, 2 * md.r_f + 500, 100),
        # "jpa_freq": make_sweep(11750, 11850, 100),
        "jpa_power": make_sweep(-25, -5, 50),
    },
    "relax_delay": 30.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=1)

jpa_opt_exp = ze.jpa.JPAAutoOptimizeExp()
_ = jpa_opt_exp.run(soc, soccfg, cfg, num_points=10000)
```

```python
%matplotlib inline
md.best_jpa_flux, md.best_jpa_freq, md.best_jpa_power, fig = jpa_opt_exp.analyze()
1e3 * md.best_jpa_flux, 1e-3 * md.best_jpa_freq, md.best_jpa_power
```

```python
filename = "JPA_opt"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
jpa_opt_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
)
```

```python
jpa_sgs.set_frequency(1e6 * md.best_jpa_freq)  # Hz
jpa_sgs.set_power(md.best_jpa_power)  # dBm
md.cur_jpa_A = jpa_yoko.set_current(md.best_jpa_flux)
```

```python
jpa_sgs.output_off()
```

## JPA Check

```python
%matplotlib widget
exp_cfg = {
    "readout": "readout_rf",
    "dev": {"jpa_sgs": {"label": "jpa_rf_dev"}},
    "sweep": make_sweep(md.r_f - 20, md.r_f + 20, 101),
    "relax_delay": 0.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=5)

jpa_check_exp = ze.jpa.JPACheckExp()
outputs, jpa_freqs, signals2D = jpa_check_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
fig = jpa_check_exp.analyze()
```

```python
filename = "JPA_check"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
jpa_check_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
)
```

```python
jpa_sgs.output_on()
# jpa_sgs.output_off()
jpa_sgs.get_info()
```

# TwoTone

```python
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
sample_table = SampleTable(f"{result_dir}/samples.csv")
```

```python
preditor = FluxoniumPredictor.from_file(f"{result_dir}/params.json")
preditor.A_c = md.mA_c
preditor.period = 2 * abs(md.mA_e - md.mA_c)
```

## Twotone Frequency

```python
md.dump_json(f"../result/{qub_name}/exp_image/{md.cur_A * 1e3:.3f}mA/metadata.json")
```

```python
# print(preditor.predict_freq(preditor.flx_to_A(0.53)))
# md.cur_A = -2.5e-3
# md.cur_A = preditor.flx_to_A(0.84)
# md.cur_A = flux_yoko.get_current()
1e3 * flux_yoko.set_current(md.cur_A)
# md.cur_V = 0.0
# flux_yoko.set_voltage(md.cur_V)
```

```python
md.q_f = preditor.predict_freq(md.cur_A, transition=(0, 1))
# q_f = preditor.predict_freq(md.cur_V, transition=(0, 1))
md.q_f
```

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_bath",
    # "init_pulse": "pi_len",
    "qub_pulse": {
        "waveform": ml.get_waveform("qub_flat", override_cfg={"length": 5.0}),
        "ch": qub_1_4_ch,
        "nqz": 2,
        "gain": 0.05,
        # "mixer_freq": md.q_f,
    },
    "readout": "readout_rf",
    # "readout": "readout_dpm",
    # "sweep": make_sweep(md.q_f - 200, md.q_f + 200, step=0.25),
    "sweep": make_sweep(md.q_f - 20, md.q_f + 20, step=0.1),
    # "sweep": make_sweep(4000, 5500, step=1.0),
    "relax_delay": 0.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

qub_freq_exp = ze.twotone.FreqExp()
_ = qub_freq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
f, kappa, fig = qub_freq_exp.analyze()
f
```

```python
md.q_f = f
md.qf_w = kappa
```

```python
filename = f"{qub_name}_freq"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
qub_freq_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"frequency = {f}MHz"),
)
```

```python
md.bias = preditor.calculate_bias(md.cur_A, md.q_f)
md.bias * 1e3
# bias = preditor.calculate_bias(md.cur_V, q_f)
# bias
```

```python
preditor.update_bias(md.bias)
```

## Reset

```python
# md.cur_A = 0.0e-3
# 1e3 * flux_dev.set_current(md.cur_A)
# md.cur_V = -12.61
# flux_yoko.set_voltage(md.cur_V)
```

### One Pulse

```python
md.reset_f = md.r_f - md.q_f
md.reset_f
```

```python
%matplotlib widget
exp_cfg = {
    # "init_pulse": "pi_amp",
    "init_pulse": ml.get_module(
        "pi_amp",
        {
            # "mixer_freq": 0.5 * (reset_f + q_f),
        },
    ),
    "tested_reset": {
        "type": "pulse",
        "pulse_cfg": {
            "waveform": ml.get_waveform("qub_flat", {"length": 5.0}),
            "ch": qub_1_4_ch,
            "nqz": 1,
            "gain": 1.0,
            # "mixer_freq": md.reset_f,
            # "mixer_freq": 0.5 * (md.reset_f + md.q_f),
            "post_delay": 5 / (2 * np.pi * md.rf_w),
        },
    },
    "readout": "readout_dpm",
    # "sweep": make_sweep(md.reset_f - 50, md.reset_f + 50, step=1.5),
    "sweep": make_sweep(50, 1500, 1001),
    "relax_delay": 0.1,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

single_reset_freq_exp = ze.twotone.reset.single_tone.FreqExp()
fpts, signals = single_reset_freq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
f, kappa, fig = single_reset_freq_exp.analyze()
f
```

```python
md.reset_f = f
```

```python
filename = f"{qub_name}_sidereset_freq"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
single_reset_freq_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"frequency = {f}MHz"),
)
```

#### Reset Time

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_bath",
    "init_pulse": "pi_amp",
    "tested_reset": {
        "type": "pulse",
        "pulse_cfg": {
            "waveform": ml.get_waveform("qub_flat"),
            "ch": qub_1_4_ch,
            "nqz": 1,
            "gain": 1.0,
            "freq": md.reset_f,
            # "mixer_freq": reset_f,
            # "mixer_freq": 0.5 * (reset_f + q_f),
            "post_delay": 5 / (2 * np.pi * md.rf_w),
        },
    },
    "readout": "readout_dpm",
    "sweep": make_sweep(0.1, 20.0, 50),
    "relax_delay": 30.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

single_reset_length_exp = ze.twotone.reset.single_tone.LengthExperiment()
Ts, signals = single_reset_length_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
fig = single_reset_length_exp.analyze()
```

```python
filename = f"{qub_name}_sidereset_time"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
single_reset_length_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

#### Set Reset Pulse

```python
cfg["tested_reset"]["pulse_cfg"]["waveform"].update(length=25.0)  # us
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
        # "mixer_freq": 0.5 * (md.reset_f + md.q_f),
    },
    "tested_reset": "reset_10",
    # "tested_reset": ml.get_module(
    #     "reset_10",
    #     {
    #         "pulse_cfg": {
    #             # "mixer_freq": 0.5 * (md.reset_f + md.q_f),
    #         }
    #     },
    # ),
    "readout": "readout_dpm",
    "sweep": make_sweep(0.0, 1.0, 51),
    "relax_delay": 70.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

single_reset_check_exp = ze.twotone.reset.RabiCheckExperiment()
pdrs, signals = single_reset_check_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
fig = single_reset_check_exp.analyze()
```

```python
filename = f"{qub_name}_sidereset_check"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
single_reset_check_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

### Two pulse

```python
jpa_sgs.output_off()
```

#### Reset Freq 1

```python
reset1_trans = (0, 3)
md.reset_f1 = preditor.predict_freq(md.cur_A, transition=reset1_trans)
# md.reset_f1 = preditor.predict_freq(md.cur_V, transition=reset1_trans)
md.reset_f1
```

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_120",
    "init_pulse": "pi_amp",
    "qub_pulse": {
        "waveform": ml.get_waveform(
            "qub_flat",
            override_cfg={"length": 5.0},
        ),
        "ch": res_ch,
        "nqz": 2,
        "gain": 1.0,
        # "mixer_freq": md.reset_f1,
        # "mixer_freq": md.q_f,
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": make_sweep(md.reset_f1 - 55, md.reset_f1 + 55, step=0.1),
    # "sweep": make_sweep(4680, 4710, step=0.1),
    "relax_delay": 30.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=1000)

dualreset_freq1_exp = ze.twotone.FreqExperiment()
fpts, signals = dualreset_freq1_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
f, kappa, fig = dualreset_freq1_exp.analyze()
f
```

```python
md.reset_f1 = f
md.resetf1_w = kappa
```

```python
filename = f"{qub_name}_dualreset_freq1"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
dualreset_freq1_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"frequency = {f}MHz"),
)
```

```python
bias = preditor.calculate_bias(md.cur_A, md.reset_f1, transition=reset1_trans)
bias * 1e3
# bias = preditor.calculate_bias(md.cur_V, md.reset_f1, transition=reset1_trans)
# bias
```

```python
preditor.update_bias(bias)
```

#### Reset Freq 2

```python
reset2_trans = (3, 1)
md.reset_f2 = abs(md.r_f + preditor.predict_freq(md.cur_A, transition=reset2_trans))
# md.reset_f2 = abs(md.r_f + preditor.predict_freq(md.cur_V, transition=reset2_trans))
md.reset_f2
```

```python
%matplotlib widget
dualreset_len = 5.0  # us
exp_cfg = {
    "reset": "reset_bath",
    # "init_pulse": "pi_amp",
    "tested_reset": {
        "type": "two_pulse",
        "pulse1_cfg": {
            "waveform": ml.get_waveform(
                "qub_flat",
                override_cfg={"length": dualreset_len},
            ),
            "ch": res_ch,
            "nqz": 1,
            "gain": 1.0,
            # "mixer_freq": md.reset_f1,
            # "mixer_freq": md.q_f,
            "block_mode": False,
        },
        "pulse2_cfg": {
            "waveform": ml.get_waveform(
                "qub_flat",
                override_cfg={"length": dualreset_len},
            ),
            "ch": qub_1_4_ch,
            "nqz": 2,
            "gain": 1.0,
            # "mixer_freq": md.reset_f2,
            "post_delay": 5.0 / (2 * np.pi * md.rf_w),
        },
    },
    "readout": "readout_dpm",
    "sweep": {
        "freq1": make_sweep(md.reset_f1 - 1.5, md.reset_f1 + 1.5, step=0.05),
        "freq2": make_sweep(md.reset_f2 - 15, md.reset_f2 - 3, step=0.05),
        # "freq1": make_sweep(1438, 1453, step=0.5),
        # "freq2": make_sweep(2610, 2620, step=0.1),
    },
    # "relax_delay": 5 / rf_w,  # us
    "relax_delay": 0.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=1000)

dualreset_freq2_exp = ze.twotone.reset.dual_tone.FreqExperiment()
fpts1, fpts2, signals = dualreset_freq2_exp.run(soc, soccfg, cfg, method="hard")
```

```python
%matplotlib inline
xlabal = f"|{reset1_trans[0]}, 0> - |{reset1_trans[1]}, 0>"
ylabal = f"|{reset2_trans[0]}, 0> - |{reset2_trans[1]}, 1>"
f1, f2, fig = dualreset_freq2_exp.analyze(smooth=0.5, xname=xlabal, yname=ylabal)
f1, f2
```

```python
reset_f1 = f1
reset_f2 = f2
```

```python
filename = f"{qub_name}_dualreset_both_freq"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
dualreset_freq2_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"frequency = ({reset_f1:.1f}, {reset_f2:.1f})MHz"),
)
```

#### Set Mux Reset Pulse

```python
mux_reset_len = 30.0
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
%matplotlib widget
exp_cfg = {
    "reset": "reset_bath",
    # "init_pulse": {
    #     **ml.get_waveform("qub_waveform"),
    #     "ch": qub_all_ch,
    #     "nqz": 2,
    #     "gain": 0.01,
    #     "mixer_freq": q_f,
    # },
    # "init_pulse": "pi_amp",
    "tested_reset": "reset_120",
    "readout": "readout_dpm",
    "sweep": {
        "gain1": make_sweep(0.0, 1.0, 51),
        "gain2": make_sweep(0.5, 1.0, 51),
    },
    "relax_delay": 0.5,  # us
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
gain1, gain2, fig = dual_reset_pdr_exp.analyze(xname=xlabal, yname=ylabal)
```

```python
filename = f"{qub_name}_mux_reset_gain"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
dual_reset_pdr_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
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
%matplotlib widget
exp_cfg = {
    "reset": "reset_bath",
    # "init_pulse": "pi_amp",
    "tested_reset": "reset_120",
    "readout": "readout_dpm",
    "sweep": make_sweep(0.05, 40.0, 51),
    "relax_delay": 0.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

dual_reset_len_exp = ze.twotone.reset.dual_tone.LengthExperiment()
Ts, signals = dual_reset_len_exp.run(soc, soccfg, cfg)
```

```python
dual_reset_len_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_mux_reset_time@{md.cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_mux_reset_time@{md.cur_V:.3f}V"),
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
    # filepath=os.path.join(database_path, f"{qub_name}_mux_reset_check@{md.cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_mux_reset_check@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

### Bath Reset

```python

```

#### rabi frequency

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    "qub_pulse": {
        **ml.get_module("pi_amp"),
        "waveform": ml.get_waveform("qub_flat"),
        "gain": 1.0,
    },
    "readout": "readout_dpm",
    "relax_delay": 30.5,  # us
    "sweep": make_sweep(0.03, 1.5, 151),
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

rabifreq_exp = ze.twotone.rabi.LenRabiExp()
_ = rabifreq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
*_, md.rabi_f, fig = rabifreq_exp.analyze(decay=True)
```

```python
filename = f"{qub_name}_rabi_freq"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
rabifreq_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"pi len = {md.pi_len}us\npi/2 len = {md.pi2_len}us"),
)
```

#### bath frequency

```python
%matplotlib widget
probe_len = 2.0
exp_cfg = {
    # "reset": "reset_10",
    "init_pulse": "pi_amp",
    "tested_reset": {
        "type": "bath",
        "qubit_tone_cfg": {
            **ml.get_module("pi_amp"),
            "waveform": ml.get_waveform("qub_flat", {"length": probe_len}),
            "gain": 1.0,
            "block_mode": False,
        },
        "cavity_tone_cfg": {
            "waveform": ml.get_waveform("qub_flat", {"length": probe_len}),
            "ch": res_ch,
            "nqz": 2,
            "post_delay": 5.0 / (2 * np.pi * md.rf_w),
        },
        "pi2_cfg": {
            **ml.get_module("pi2_amp"),
            "phase": 90,
        },
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": {
        "freq": make_sweep(md.r_f - 4.0 * md.rabi_f, md.r_f - 0.5 * md.rabi_f, 51),
        "gain": make_sweep(0.0, 0.1, 51),
    },
    "relax_delay": 30.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

bathreset_freq_exp = ze.twotone.reset.bath.FreqGainExp()
_ = bathreset_freq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
md.bathreset_gain, md.bathreset_freq, fig = bathreset_freq_exp.analyze(
    smooth=1, find="max"
)
```

```python
filename = f"{qub_name}_bathreset_freqgain"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
bathreset_freq_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

#### Length

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    # "init_pulse": "pi_amp",
    "tested_reset": {
        "type": "bath",
        "qubit_tone_cfg": {
            **ml.get_module("pi_amp"),
            "waveform": ml.get_waveform("qub_flat"),
            "gain": 1.0,
            "block_mode": False,
        },
        "cavity_tone_cfg": {
            "waveform": ml.get_waveform("qub_flat"),
            "ch": res_ch,
            "nqz": 2,
            "gain": md.bathreset_gain,
            "freq": md.bathreset_freq,
            "post_delay": 5.0 / (2 * np.pi * md.rf_w),
        },
        "pi2_cfg": "pi2_amp",
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": make_sweep(0.03, 2.5, 151),
    "relax_delay": 0.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

activate_detune = 0.1 / cfg["sweep"]["step"]
print(f"activate_detune: {activate_detune:.2f}")

bathreset_len_exp = ze.twotone.reset.bath.LengthExp()
lens, signals = bathreset_len_exp.run(soc, soccfg, cfg, detune=activate_detune)
```

```python
%matplotlib inline
fig = bathreset_len_exp.analyze()
```

```python
filename = f"{qub_name}_bathreset_len"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
bathreset_len_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

#### Set Bath reset pulse

```python
bath_reset_len = 1.5  # us
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
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    "tested_reset": "reset_bath",
    "readout": "readout_rf",
    # "readout": "readout_dpm",
    "sweep": make_sweep(-170.0, 170.0, 51),
    "relax_delay": 30.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

bathreset_phase_exp = ze.twotone.reset.bath.PhaseExp()
phases, signals = bathreset_phase_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
max_phase, min_phase, fig = bathreset_phase_exp.analyze()
```

```python
filename = f"{qub_name}_bathreset_phase"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
bathreset_phase_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

```python
ml.register_module(
    reset_bath={
        **cfg["tested_reset"],
        "pi2_cfg": {
            **cfg["tested_reset"]["pi2_cfg"],
            # "phase": max_phase,
            "phase": min_phase,
        },
    },
    reset_bath_e={
        **cfg["tested_reset"],
        "pi2_cfg": {
            **cfg["tested_reset"]["pi2_cfg"],
            # "phase": min_phase,
            "phase": max_phase,
        },
    },
)
```

#### Check reset

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    "rabi_pulse": "pi_amp",
    "tested_reset": "reset_bath",
    # "post_pulse": ml.get_module("pi2_amp"),
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": make_sweep(0.0, 1.0, 51),
    "relax_delay": 70.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

bathreset_rabicheck_exp = ze.twotone.reset.RabiCheckExp()
pdrs, signals = bathreset_rabicheck_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
fig = bathreset_rabicheck_exp.analyze()
```

```python
filename = f"{qub_name}_bathreset_rabicheck"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
bathreset_rabicheck_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## Flux Dependence

```python
md.cur_A = 4e-3
1e3 * flux_yoko.set_current(md.cur_A)
# md.cur_V = 1.75
# flux_yoko.set_voltage(md.cur_V)
```

```python
import gc

plt.close("all")
gc.collect()
```

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_120",
    "qub_pulse": {
        "waveform": ml.get_waveform("qub_flat", override_cfg={"length": 5.0}),
        "ch": qub_1_4_ch,
        "nqz": 2,
        "gain": 1.0,
        # "mixer_freq": md.q_f,
    },
    "readout": "readout_rf",
    "dev": {
        "flux_yoko": {
            "label": "flux_dev",
            "mode": "current",
        },
    },
    "sweep": {
        "flux": make_sweep(4e-3, -3e-3, 301),
        # "flux": make_sweep(-4.0, -3.0, 201),
        "freq": make_sweep(1000, 5000, step=0.5),
    },
    "relax_delay": 0.1,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=3)

qub_flux_exp = ze.twotone.FreqFluxDepExp()
_ = qub_flux_exp.run(soc, soccfg, cfg, fail_retry=3)
```

```python
qub_flux_exp.save(
    filepath=os.path.join(database_path, f"{qub_name}_flux"),
    # filepath=os.path.join(database_path, f"{qub_name}_flux@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_flux@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

```python
%matplotlib widget
actline = qub_flux_exp.analyze(mA_c=md.mA_c, mA_e=md.mA_e)
```

```python
md.mA_c, md.mA_e = actline.get_positions()
md.mA_c, md.mA_e
```

## Other TwoTone


### Power dependence

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
    filepath=os.path.join(database_path, f"{qub_name}_pdr@{md.cur_A * 1e3:.3f}mA"),
    comment=make_comment(cfg),
)
```

### Dispersive Shift

```python
jpa_sgs.output_off()
```

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    "qub_pulse": "pi_amp",
    "readout": ml.get_module(
        "readout_dpm",
        {
            "pulse_cfg": {
                "gain": 0.01,
            }
        },
    ),
    "sweep": make_sweep(md.r_f - 2.25 * md.rf_w, md.r_f + 2.0 * md.rf_w, step=0.1),
    "relax_delay": 30.5,  # us
    # "relax_delay": 2 * t1, # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=1000)

dispersive_shift_exp = ze.twotone.DispersiveExp()
fpts, signals = dispersive_shift_exp.run(soc, soccfg, cfg)
```

```python
plt.close("all")
gc.collect()
```

```python
%matplotlib inline
md.chi, rf_w, fig = dispersive_shift_exp.analyze()
```

```python
filename = f"{qub_name}_dispersive_shift_gain{cfg['readout']['pulse_cfg']['gain']:.3f}"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
dispersive_shift_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"chi = {md.chi:.3g} MHz, kappa = {rf_w:.3g} MHz"),
)
```

### AC Stark Shift

```python
%matplotlib widget
ac_qub_len = 10.0  # us
exp_cfg = {
    # "reset": "reset_bath",
    "stark_pulse1": {
        "waveform": ml.get_waveform(
            "mist_waveform", {"length": 5.1 / (2 * np.pi * md.rf_w) + ac_qub_len}
        ),
        "ch": res_ch,
        "nqz": 2,
        "freq": md.r_f,
        "block_mode": False,
    },
    "stark_pulse2": {
        "waveform": ml.get_waveform("qub_flat", override_cfg={"length": ac_qub_len}),
        "ch": qub_1_4_ch,
        "nqz": 2,
        "gain": 0.01,
        # "mixer_freq": md.q_f,
        # "mixer_freq": 0.5 * (reset_f + q_f),
        "pre_delay": 5.0 / (2 * np.pi * md.rf_w),
        "post_delay": 3.1 / (2 * np.pi * md.rf_w),
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": {
        "gain": make_sweep(0.0, 0.01, 101),
        "freq": make_sweep(md.q_f - 7.0, md.q_f + 1.0, step=0.05),
    },
    "relax_delay": 0.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=30)

ac_stark_exp = ze.twotone.AcStarkExp()
pdrs, fpts, signals2D = ac_stark_exp.run(soc, soccfg, cfg, earlystop_snr=50)
```

```python
%matplotlib inline
md.ac_stark_coeff, fig = ac_stark_exp.analyze(
    chi=md.chi, kappa=md.rf_w, deg=1, cutoff=0.006
)
```

```python
filename = f"{qub_name}_ac_stark"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
ac_stark_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    # comment=make_comment(cfg, f"ac_stark_coeff = {md.ac_stark_coeff:.3g} MHz"),
    comment=make_comment(cfg),
)
```

### All XY

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_bath",
    "X180_pulse": "pi_amp",
    "X90_pulse": "pi2_amp",
    "Y180_pulse": ml.get_module("pi_amp", {"phase": 90}),
    "Y90_pulse": ml.get_module("pi2_amp", {"phase": 90}),
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "relax_delay": 40.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

allxy_exp = ze.twotone.AllXY_Exp()
signals_dict = allxy_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
allxy_exp.analyze()
```

```python
allxy_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_allxy@{md.cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_allxy@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

### Zig-Zag

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    "X90_pulse": "pi2_amp",
    # "X180_pulse": "pi_amp",
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": list(range(0, 11)),
    "relax_delay": 30.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

repeat_on = "X90_pulse"

zigzag_exp = ze.twotone.ZigZagExp()
_ = zigzag_exp.run(soc, soccfg, cfg, repeat_on=repeat_on)
```

```python
filename = f"{qub_name}_zigzag_{repeat_on}"
zigzag_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{qub_name}_zigzag@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

#### Sweep parameters

```python
%matplotlib widget
repeat_on = "X180_pulse"

exp_cfg = {
    # "reset": "reset_bath",
    "X90_pulse": "pi2_amp",
    "X180_pulse": "pi_amp",
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": {"times": list(range(0, 7))},
    "relax_delay": 25.5,  # us
}
if repeat_on == "X90_pulse":
    exp_cfg["sweep"].update(
        gain=make_sweep(max(0.0, md.pi2_gain - 0.05), min(1.0, md.pi2_gain + 0.05), 101)
    )
elif repeat_on == "X180_pulse":
    exp_cfg["sweep"].update(
        gain=make_sweep(max(0.0, md.pi_gain - 0.05), min(1.0, md.pi_gain + 0.05), 101)
    )
else:
    raise ValueError(f"Invalid repeat_on: {repeat_on}")
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)


zigzag_sweep_exp = ze.twotone.ZigZagSweepExp()
_ = zigzag_sweep_exp.run(soc, soccfg, cfg, repeat_on=repeat_on)
```

```python
%matplotlib inline
best_x, fig = zigzag_sweep_exp.analyze()
```

```python
filename = f"{qub_name}_zigzag_sweep_{repeat_on}"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
zigzag_sweep_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

```python
if repeat_on == "X90_pulse":
    md.pi2_gain = best_x
    ml.update_module(name="pi2_amp", override_cfg={"gain": md.pi2_gain})
elif repeat_on == "X180_pulse":
    md.pi_gain = best_x
    ml.update_module(name="pi_amp", override_cfg={"gain": md.pi_gain})
else:
    raise ValueError("Invalid repeat_on value")
```

# Rabi

```python

```

## Length Rabi

```python
jpa_sgs.output_off()
```

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    "qub_pulse": {
        "waveform": ml.get_waveform("qub_flat"),
        "ch": qub_1_4_ch,
        "nqz": 2,
        "freq": md.q_f,
        "gain": 1.0,
        # "gain": md.pi_gain,
        # "mixer_freq": md.q_f,
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "relax_delay": 30.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.03, 1.5, 101),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

qub_lenrabi_exp = ze.twotone.rabi.LenRabiExp()
Ts, signals = qub_lenrabi_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
md.pi_len, md.pi2_len, md.rabi_f, fig = qub_lenrabi_exp.analyze(decay=True)
md.pi_len, md.pi2_len, md.rabi_f
```

```python
filename = f"{qub_name}_rabi_length"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
qub_lenrabi_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"pi len = {md.pi_len}us\npi/2 len = {md.pi2_len}us"),
)
```

```python
# pi_len = 1.0
# pi2_len = 0.5
ml.register_module(
    pi_len={
        **exp_cfg["qub_pulse"],
        "waveform": {**exp_cfg["qub_pulse"]["waveform"], "length": md.pi_len},
        # "pre_delay": 0.005,
        # "post_delay": 0.005,
        "desc": "len pi pulse",
    },
    pi2_len={
        **exp_cfg["qub_pulse"],
        "waveform": {**exp_cfg["qub_pulse"]["waveform"], "length": md.pi2_len},
        # "pre_delay": 0.005,
        # "post_delay": 0.005,
        "desc": "len pi/2 pulse",
    },
)
```

## Amplitude Rabi

```python
%matplotlib widget
max_gain = min(5 * ml.get_module("pi_len")["gain"], 1.0)
exp_cfg = {
    # "reset": "reset_10",
    "qub_pulse": {
        "waveform": ml.get_waveform(
            "qub_flat",
            override_cfg={"length": 1.1 * md.pi_len},
        ),
        "ch": qub_1_4_ch,
        "nqz": 2,
        "freq": md.q_f,
        # "mixer_freq": md.q_f,
        # "mixer_freq": 0.5 * (reset_f + md.q_f),
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "relax_delay": 30.5,  # us
    # "relax_delay": 5 * t1,
    "sweep": make_sweep(0.01, 1.0, 51),
    # "sweep": make_sweep(0.0, max_gain, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

qub_amprabi_exp = ze.twotone.rabi.AmpRabiExp()
pdrs, signals = qub_amprabi_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
md.pi_gain, md.pi2_gain, fig = qub_amprabi_exp.analyze(skip=2)
md.pi_gain, md.pi2_gain
```

```python
filename = f"{qub_name}_rabi_amplitude"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
qub_amprabi_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"pi gain = {md.pi_gain}\npi/2 gain = {md.pi2_gain}"),
)
```

```python
# pi_gain = 1.0
# pi2_gain = 0.5
ml.register_module(
    pi_amp={
        **cfg["qub_pulse"],
        "gain": md.pi_gain,
        # "pre_delay": 0.005,
        # "post_delay": 0.005,
        "desc": "amp pi pulse",
    },
    pi2_amp={
        **cfg["qub_pulse"],
        "gain": md.pi2_gain,
        # "pre_delay": 0.005,
        # "post_delay": 0.005,
        "desc": "amp pi/2 pulse",
    },
)
```

# Optimize Readout

```python
# jpa_sgs.output_off()
jpa_sgs.output_on()
jpa_sgs.get_info()
```

## Frequency tuning

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_bath",
    "qub_pulse": "pi_amp",
    "readout": ml.get_module(
        "readout_rf",
        override_cfg={
            # "pulse_cfg": {
            #     "waveform": {"length": 1.5},
            # "gain": 0.02,
            # },
            # "ro_cfg": {
            #     "ro_length": 1.4,
            # },
        },
    ),
    "relax_delay": 30.5,  # us
    # "relax_delay": 3 * t1,  # us
    "sweep": make_sweep(md.r_f - 10, md.r_f + 10, step=0.15),
    # "sweep": make_sweep(5920, 5940, step=0.25),
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=50)

opt_ro_freq_exp = ze.twotone.ro_optimize.FreqExp()
fpts, snrs = opt_ro_freq_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
md.best_ro_freq, fig = opt_ro_freq_exp.analyze(smooth=1)
md.best_ro_freq
```

```python
filename = f"{qub_name}_ro_opt_freq"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
opt_ro_freq_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"optimal frequency = {md.best_ro_freq:.1f}MHz"),
)
```

```python
md.best_ro_freq = md.r_f
```

## Power tuning

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_bath",
    "qub_pulse": "pi_amp",
    "readout": ml.get_module(
        "readout_rf",
        override_cfg={
            "pulse_cfg": {
                # "waveform": {"length": 1.5},
                "freq": md.best_ro_freq,
            },
            "ro_cfg": {
                # "ro_length": 1.5,
            },
        },
    ),
    "relax_delay": 30.5,  # us
    # "relax_delay": 3 * t1,  # us
    "sweep": make_sweep(0.01, 0.2, 151),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

opt_ro_pdr_exp = ze.twotone.ro_optimize.PowerExp()
pdrs, snrs = opt_ro_pdr_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
md.best_ro_gain, fig = opt_ro_pdr_exp.analyze(penalty_ratio=1.0)
md.best_ro_gain
```

```python
filename = f"{qub_name}_ro_opt_pdr"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
opt_ro_pdr_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"optimal power = {md.best_ro_gain:.2f}"),
)
```

## Readout Length tuning

```python
%matplotlib widget
# pdr_max = 0.6
exp_cfg = {
    # "reset": "reset_bath",
    "qub_pulse": "pi_amp",
    "readout": ml.get_module(
        "readout_rf",
        override_cfg={
            "pulse_cfg": {
                "freq": md.best_ro_freq,
                "gain": md.best_ro_gain,
            }
        },
    ),
    "relax_delay": 30.5,  # us
    # "relax_delay": 3 * t1,  # us
    "sweep": make_sweep(0.01, 1.5, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=10000, rounds=1)

opt_ro_len_exp = ze.twotone.ro_optimize.LengthExp()
ro_lens, snrs = opt_ro_len_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
md.best_ro_length, fig = opt_ro_len_exp.analyze(t0=10.0)
md.best_ro_length
```

```python
filename = f"{qub_name}_ro_opt_len"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
opt_ro_len_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"optimal readout length = {md.best_ro_length:.2f}us"),
)
```

```python
# ro_max = 1.5
ml.register_module(
    readout_dpm=ml.get_module(
        "readout_rf",
        {
            "pulse_cfg": {
                "freq": md.best_ro_freq,
                "gain": md.best_ro_gain,
                "waveform": {
                    "length": md.best_ro_length + 0.1,
                },
            },
            "ro_cfg": {
                "ro_length": md.best_ro_length,
            },
            "desc": "Readout with largest dispersive shift",
        },
    )
)
```

## Auto Optimize

```python
import gc

plt.close("all")
gc.collect()
```

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_bath",
    "qub_pulse": "pi_amp",
    "readout": "readout_rf",
    "relax_delay": 30.5,  # us
    "sweep": {
        "freq": make_sweep(md.r_f - md.rf_w, md.r_f + md.rf_w, 51),
        "gain": make_sweep(0.01, 0.2, 51),
        "length": make_sweep(0.01, 2.0, 51),
    },
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

auto_opt_ro_exp = ze.twotone.ro_optimize.AutoExp()
_ = auto_opt_ro_exp.run(soc, soccfg, cfg, num_points=1001)
```

```python
%matplotlib inline
md.best_ro_freq, md.best_ro_gain, md.best_ro_length, fig = auto_opt_ro_exp.analyze()
md.best_ro_freq, md.best_ro_gain, md.best_ro_length
```

```python
filename = f"{qub_name}_auto_opt"
auto_opt_ro_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

```python
# md.best_ro_freq = md.r_f
ml.update_module(
    "readout_dpm",
    {
        "pulse_cfg": {
            "freq": md.best_ro_freq,
            "gain": md.best_ro_gain,
            "waveform": {
                # "length": md.best_ro_length + 0.1,
                "length": 0.25 * md.t1_with_tone + 0.1,
            },
        },
        "ro_cfg": {
            # "ro_length": md.best_ro_length,
            "ro_length": 0.25 * md.t1_with_tone + 0.1,
        },
    },
)
```

# T1 & T2

```python
# t1 = 5.0
```

## T2Ramsey

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    "pi2_pulse": "pi2_amp",
    "readout": "readout_dpm",
    "relax_delay": 30.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0, 1.0, 101),  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

# activate_detune = 1.5
activate_detune = 0.1 / cfg["sweep"]["step"]
print(f"activate_detune: {activate_detune:.2f}")

t2ramsey_exp = ze.twotone.time_domain.T2RamseyExp()
_ = t2ramsey_exp.run(soc, soccfg, cfg, detune=activate_detune)
```

```python
%matplotlib inline
t2r, _, detune, _, fig = t2ramsey_exp.analyze()
print(f"real detune: {(detune - activate_detune) * 1e3:.1f}kHz")
```

```python
filename = f"{qub_name}_t2ramsey"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
t2ramsey_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"detune = {detune:.3f}MHz\nt2r = {t2r:.3f}us"),
)
```

```python
md.q_f = ml.get_module("pi2_amp")["freq"] + activate_detune - detune
md.q_f
```

## T1

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_bath",
    "pi_pulse": "pi_amp",
    # "pi_pulse": {
    #     "waveform": ml.get_waveform("qub_flat", override_cfg={"length": 15.0}),
    #     "ch": qub_1_4_ch,
    #     "nqz": 1,
    #     "gain": 0.5,
    #     "freq": q_f,
    #     "mixer_freq": q_f,
    # },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "relax_delay": 50.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.0, 50.1, 101),
    # "sweep": make_sweep(0.01*t1, 5 * t1, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

t1_exp = ze.twotone.time_domain.T1Exp()
Ts, signals = t1_exp.run(soc, soccfg, cfg, uniform=True)
```

```python
%matplotlib inline
md.t1, md.t1err, fig = t1_exp.analyze(dual_exp=False)
md.t1
```

```python
filename = f"{qub_name}_t1"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
t1_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"t1 = {md.t1:.3f}us"),
)
```

```python
from datetime import datetime

sample_table.add_sample(
    **{
        "calibrated mA": md.cur_A,
        "Freq (MHz)": md.q_f,
        "T1 (us)": md.t1,
        "T1err (us)": md.t1err,
        "comment": "Manual Added",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
)
md.dump_json(f"../result/{qub_name}/exp_image/{md.cur_A * 1e3:.3f}mA/metadata.json")
```

### With Tone

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_bath",
    "pi_pulse": "pi_amp",
    "test_pulse": ml.get_module(
        "readout_dpm",
        {
            "pulse_cfg": {
                "post_delay": 5.0 / (2 * np.pi * md.rf_w),
            }
        },
    )["pulse_cfg"],
    "readout": "readout_dpm",
    "relax_delay": 50.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(1.0, 20, 101),
    # "sweep": make_sweep(0.01*t1, 5 * t1, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

t1_with_tone_exp = ze.twotone.time_domain.T1WithToneExp()
_ = t1_with_tone_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
md.t1_with_tone, _, fig = t1_with_tone_exp.analyze(dual_exp=False)
md.t1_with_tone
```

```python
filename = f"{qub_name}_t1_with_tone_{cfg['test_pulse']['gain']:.2f}"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
t1_with_tone_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"t1 = {md.t1_with_tone:.3f}us"),
)
```

### With Sweep Tone

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_120",
    "pi_pulse": "pi_amp",
    "test_pulse": {
        "waveform": "mist_waveform",
        "ch": res_ch,
        "nqz": 2,
        "freq": md.r_f,
        "post_delay": 3.0 / (2 * np.pi * md.rf_w),
    },
    "readout": "readout_dpm",
    "relax_delay": 30.0,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": {
        "gain": make_sweep(0.0, 1.0, 301),
        "length": make_sweep(1.0, 30, 501),
    },
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

t1_with_tone_sweep_exp = ze.twotone.time_domain.T1WithToneSweepExp()
_ = t1_with_tone_sweep_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
*_, fig = t1_with_tone_sweep_exp.analyze()
```

```python
filename = f"{qub_name}_t1_with_tone_sweep"
savefig(
    fig,
    f"../result/{qub_name}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png",
)
plt.close(fig)
t1_with_tone_sweep_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## T2Echo

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_120",
    "pi_pulse": "pi_amp",
    "pi2_pulse": "pi2_amp",
    "readout": "readout_dpm",
    "relax_delay": 40.0,  # us
    # "relax_delay": 5 * t1,  # us
    # "sweep": make_sweep(0.0, 5 * t2r, 51),
    # "sweep": make_sweep(0.0, 1.5 * t2e, 101),
    "sweep": make_sweep(0.01, 5.0, 101),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

activate_detune = 0.1 / cfg["sweep"]["step"]
print(f"activate_detune: {activate_detune:.2f}")

t2echo_exp = ze.twotone.time_domain.T2EchoExp()
Ts, signals = t2echo_exp.run(soc, soccfg, cfg, detune=activate_detune)
```

```python
%matplotlib inline
md.t2e, _, detune, _, fig = t2echo_exp.analyze(fit_method="fringe")
```

```python
filename = f"{qub_name}_t2echo"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
t2echo_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"detune = {detune:.3f}MHz\nt2echo = {md.t2e:.3f}us"),
)
```

## CPMG

```python
ml.get_module("pi_len")["waveform"]["length"]
```

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_120",
    "pi_pulse": "pi_amp",
    "pi2_pulse": {
        **ml.get_module("pi2_amp"),
        "phase": 90,  # Y/2 gate
    },
    "readout": "readout_dpm",
    "relax_delay": 40.0,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": {
        "times": list(range(1, 101, 10)),
        "length": make_sweep(0.0, 15, 31),
    },
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

cpmg_exp = ze.twotone.time_domain.CPMGExp()
times, Ts, signals = cpmg_exp.run(soc, soccfg, cfg)
```

```python
_, _, fig = cpmg_exp.analyze()
```

```python
cpmg_exp.save(
    # filepath=os.path.join(database_path, f"{qub_name}_t2echo@{md.cur_A * 1e3:.3f}mA"),
    filepath=os.path.join(database_path, f"{qub_name}_cpmg@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

# Single shot

```python
# jpa_sgs.output_off()
jpa_sgs.output_on()
# jpa_yoko.set_current(md.best_jpa_flux)
# md.cur_jpa_A = jpa_yoko.set_current(md.best_jpa_flux)
# md.cur_jpa_A
jpa_sgs.get_info()
```

## Ground state & Excited state

```python
exp_cfg = {
    # "reset": "reset_10",
    "probe_pulse": "pi_amp",
    # "probe_pulse": ml.get_module("readout_dpm")["pulse_cfg"],
    "readout": ml.get_module(
        # "readout_rf",
        "readout_dpm",
        {
            "pulse_cfg": {
                "waveform": {
                    # "length": 0.1*md.t1_with_tone + 0.1,
                    # "length": 1.0 + 0.1,
                },
                # "gain": 0.05,
                # "freq": 5350.64,
            },
            "ro_cfg": {
                # "ro_length": 0.1*md.t1_with_tone,
                # "ro_length": 1.0,
            },
        },
    ),
    "relax_delay": 70.5,  # us
}
cfg = ml.make_cfg(exp_cfg, shots=100000)
print("readout length: ", cfg["readout"]["ro_cfg"]["ro_length"])

ge_sh_exp = ze.singleshot.GE_Exp()
_ = ge_sh_exp.run(soc, soccfg, cfg)
```

```python
import gc

plt.close("all")
gc.collect()
```

```python
%matplotlib inline
md.fid, pops, result_dict, fig = ge_sh_exp.analyze(
    backend="center",
    # init_p0=0.0,
    length_ratio=cfg["readout"]["ro_cfg"]["ro_length"] / md.t1_with_tone,
    logscale=True,
    align_t1=True,
)
print(f"Optimal fidelity after rotation = {md.fid:.1%}")
```

```python
import time

filename = f"{qub_name}_singleshot_w_jpa_wo_reset_log_{time.strftime('%Y%m%d_%H%M%S')}"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
ge_sh_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"fide: {md.fid:.1%}"),
)
```

```python
from zcu_tools.simulate.temp import effective_temperature

n_g = pops[0][0]  # n_gg
n_e = pops[0][1]  # n_ge

n_g, n_e = (n_g, n_e) if n_g > n_e else (n_e, n_g)  # ensure n_g >= n_e
n_g, n_e = n_g / (n_g + n_e), n_e / (n_g + n_e)  # normalize

eff_T, err_T = effective_temperature(population=[(n_g, 0.0), (n_e, md.q_f)])
eff_T, err_T
```

```python
md.g_center = result_dict["g_center"]
md.e_center = result_dict["e_center"]
md.ge_s = result_dict["s"]
md.g_center, md.e_center, md.ge_s
```

## Confusion matrix

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    "probe_pulse": "pi_amp",
    # "probe_pulse": {
    #     "waveform": ml.get_waveform(
    #         "mist_waveform",
    #         {
    #             # "length": 5.0 / (2 * np.pi * md.rf_w) + 50.0,
    #             "length": 5.0 / (2 * np.pi * md.rf_w) + 0.0,
    #         },
    #     ),
    #     "ch": res_ch,
    #     "nqz": 2,
    #     "freq": md.r_f,
    #     "post_delay": 10 / (2 * np.pi * md.rf_w),
    # },
    "readout": "readout_dpm",
    "relax_delay": 70.5,  # us
}
cfg = ml.make_cfg(exp_cfg, shots=10000)

check_sh_exp = ze.singleshot.CheckExp()
_ = check_sh_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
md.ge_radius = 2.5 * md.ge_s
fig = check_sh_exp.analyze(md.g_center, md.e_center, md.ge_radius, max_point=10000)
```

```python
filename = f"{qub_name}_singleshot_e"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
ge_sh_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg, f"ge_radius: {md.ge_radius:.3}"),
)
```

```python
%matplotlib inline
md.confusion_matrix, fig = ge_sh_exp.calc_confusion_matrix(
    md.g_center, md.e_center, md.ge_radius, init_pops=pops
)
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
        "freq": md.q_f,
        "gain": 1.0,
        # "gain": md.pi_gain,
        # "mixer_freq": md.q_f,
    },
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "relax_delay": 50.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.03, 0.5, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

qub_lenrabi_sh_exp = ze.singleshot.LenRabiExp()
_ = qub_lenrabi_sh_exp.run(soc, soccfg, cfg, md.g_center, md.e_center, md.ge_radius)
```

```python
%matplotlib inline
fig = qub_lenrabi_sh_exp.analyze(
    confusion_matrix=md.confusion_matrix,
)
```

```python
filename = f"{qub_name}_rabi_length_pop"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
qub_lenrabi_sh_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## T1

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_bath",
    "pi_pulse": "pi_amp",
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "relax_delay": 50.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.01, 50.1, 101),
    # "sweep": make_sweep(0.01*t1, 5 * t1, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

t1_sh_exp = ze.singleshot.t1.T1Exp()
_ = t1_sh_exp.run(
    soc, soccfg, cfg, md.g_center, md.e_center, md.ge_radius, uniform=True
)
```

```python
%matplotlib inline
fig = t1_sh_exp.analyze(
    confusion_matrix=md.confusion_matrix,
)
```

```python
filename = f"{qub_name}_t1_pop"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
t1_sh_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

### T1 with Tone

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    "pi_pulse": "pi_amp",
    "probe_pulse": ml.get_module(
        "readout_dpm",
        {
            "pulse_cfg": {
                "post_delay": 5.0 / (2 * np.pi * md.rf_w),
            }
        },
    )["pulse_cfg"],
    "readout": "readout_dpm",
    "relax_delay": 50.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.03, 20, 101),
    # "sweep": make_sweep(0.01*t1, 5 * t1, 51),
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=10)

t1_with_tone_sh_exp = ze.singleshot.t1.T1WithToneExp()
_ = t1_with_tone_sh_exp.run(
    soc, soccfg, cfg, md.g_center, md.e_center, md.ge_radius, uniform=True
)
```

```python
%matplotlib inline
fig = t1_with_tone_sh_exp.analyze(
    confusion_matrix=md.confusion_matrix,
)
```

```python
filename = f"{qub_name}_t1_with_tone_gain{cfg['probe_pulse']['gain']:.3f}_pop"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
t1_with_tone_sh_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

### T1 with sweep tone

```python
import gc

plt.close("all")
gc.collect()
```

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    "pi_pulse": "pi_amp",
    "probe_pulse": {
        "waveform": ml.get_waveform("mist_waveform"),
        "ch": res_ch,
        "nqz": 2,
        "freq": md.r_f,
        "post_delay": 10 / (2 * np.pi * md.rf_w),
    },
    "readout": "readout_dpm",
    "relax_delay": 30.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": {
        "gain": np.linspace(0.0**2, 0.2**2, 151) ** 0.5,
        "length": make_sweep(0.01, 15, 501),
    },
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=5)

t1_with_tone_sweep_sh_exp = ze.singleshot.t1.T1WithToneSweepExp()
_ = t1_with_tone_sweep_sh_exp.run(
    soc, soccfg, cfg, md.g_center, md.e_center, md.ge_radius
)
```

```python
%matplotlib inline
fig = t1_with_tone_sweep_sh_exp.analyze(
    ac_coeff=md.ac_stark_coeff, confusion_matrix=md.confusion_matrix
)
```

```python
filename = f"{qub_name}_t1_with_tone_sweep_pop"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
t1_with_tone_sweep_sh_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## MIST

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_10",
    "init_pulse": "pi_amp",
    "probe_pulse": {
        "waveform": ml.get_waveform(
            "mist_waveform",
            {
                # "length": 5.0 / (2 * np.pi * md.rf_w) + 50.0,
                "length": 5.0 / (2 * np.pi * md.rf_w) + 0.3,
            },
        ),
        "ch": res_ch,
        "nqz": 2,
        "freq": md.r_f,
        "post_delay": 10 / (2 * np.pi * md.rf_w),
    },
    "readout": "readout_dpm",
    "sweep": {
        "gain": make_sweep(0.0, 0.03, 301),
    },
    "relax_delay": 50.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

mist_sh_exp = ze.singleshot.mist.PowerDepExp()
_ = mist_sh_exp.run(soc, soccfg, cfg, md.g_center, md.e_center, md.ge_radius)
```

```python
%matplotlib inline
fig = mist_sh_exp.analyze(
    ac_coeff=md.ac_stark_coeff, confusion_matrix=md.confusion_matrix
)
```

```python
filename = f"{qub_name}_mist_e_singleshot_short_{time.strftime('%Y%m%d_%H%M%S')}"
# filename = f"{qub_name}_mist_singleshot_steady_{time.strftime('%Y%m%d_%H%M%S')}"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
mist_sh_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## MIST Shots

```python
%matplotlib widget
exp_cfg = {
    "reset": "reset_10",
    "init_pulse": "pi_amp",
    "probe_pulse": {
        "waveform": ml.get_waveform(
            "mist_waveform",
            {
                "length": 5.0 / (2 * np.pi * md.rf_w) + 50.0,
                # "length": 5.0 / (2 * np.pi * md.rf_w) + 0.3,
            },
        ),
        "ch": res_ch,
        "nqz": 2,
        "freq": md.r_f,
        "gain": 0.112,
        "post_delay": 10 / (2 * np.pi * md.rf_w),
    },
    "readout": "readout_dpm",
    "relax_delay": 10.5,  # us
}
cfg = ml.make_cfg(exp_cfg, shots=1000000)

mist_sh_exp = ze.singleshot.CheckExp()
_ = mist_sh_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
fig = mist_sh_exp.analyze(md.g_center, md.e_center, 3.0 * md.ge_s)
```

```python
filename = f"{qub_name}_mist_gain{cfg['probe_pulse']['gain']:.4f}_singleshot_steady"
# filename = f"{qub_name}_e_mist_gain{cfg['probe_pulse']['gain']:.4f}_singleshot_short"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
mist_sh_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

## AC Stark shift

```python
import gc

plt.close("all")
gc.collect()
```

```python
%matplotlib widget
pi_amp_len = ml.get_module("pi_amp")["waveform"]["length"]
exp_cfg = {
    # "reset": "reset_bath",
    "stark_pulse1": {
        "waveform": ml.get_waveform(
            "mist_waveform", {"length": 5.1 / (2 * np.pi * md.rf_w) + pi_amp_len}
        ),
        "ch": res_ch,
        "nqz": 2,
        "freq": md.r_f,
        "block_mode": False,
    },
    "stark_pulse2": ml.get_module(
        "pi_amp",
        {
            "pre_delay": 5.0 / (2 * np.pi * md.rf_w),
            "post_delay": 3.1 / (2 * np.pi * md.rf_w),
        },
    ),
    # "readout": "readout_rf",
    "readout": "readout_dpm",
    "sweep": {
        "gain": make_sweep(0.0, 0.05, 301),
        "freq": make_sweep(md.q_f - 150.0, md.q_f + 50.0, step=0.025),
    },
    "relax_delay": 0.5,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=3000, rounds=1)

ac_stark_sh_exp = ze.singleshot.AcStarkExp()
_ = ac_stark_sh_exp.run(soc, soccfg, cfg, md.g_center, md.e_center, md.ge_radius)
```

```python
%matplotlib inline
fig = ac_stark_sh_exp.analyze(
    ac_coeff=md.ac_stark_coeff, confusion_matrix=md.confusion_matrix
)
```

```python
filename = f"{qub_name}_ac_stark_pop"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
ac_stark_sh_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

# MIST

```python
mist_pulse_len = 1.0  # us
ml.register_waveform(
    mist_waveform={
        "style": "const",
        "length": mist_pulse_len,  # us
    },
    # mist_padding_waveform={
    #     "style": "padding",
    #     "length": mist_pulse_len,
    #     "pre_length": 0.02,
    #     "post_length": 0.01,
    #     "pre_gain": 1.0,
    #     "post_gain": -1.0,
    # },
)
```

## Flux dependence

```python
md.cur_A = -2.8e-3
1e3 * flux_yoko.set_current(md.cur_A)
```

```python
jpa_sgs.output_off()
```

```python
%matplotlib widget
exp_cfg = {
    # "init_pulse": "pi_amp",
    "probe_pulse": {
        "waveform": ml.get_waveform("mist_waveform"),
        "ch": res_ch,
        "nqz": 2,
        "freq": md.r_f,
        "post_delay": 10 / (2 * np.pi * md.rf_w),
    },
    "readout": "readout_rf",
    "dev": {
        "flux_yoko": {
            "label": "flux_dev",
            "mode": "current",
        },
    },
    "sweep": {
        "flux": make_sweep(-1.9e-3, 2.4e-3, 71),
        "gain": make_sweep(0.0, 0.1, 151),
    },
    "relax_delay": 30.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=1000, rounds=100)

mist_flux_exp = ze.twotone.mist.MistFluxDepExperiment()
values, gains, signals = mist_flux_exp.run(soc, soccfg, cfg)
```

```python
fig = mist_flux_exp.analyze()
fig.show()
```

```python
filename = f"{qub_name}_mist_over_flux"
# fig.write_image(f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
# plt.close(fig)
mist_flux_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

```python
flux_yoko.set_current(0.0e-3)
jpa_yoko.set_current(0.0e-3)
jpa_sgs.output_off()
```

## Single Trace

```python
# md.cur_A = 1.162e-3
1e3 * flux_yoko.set_current(current=md.cur_A)
# md.cur_V = 0.0
# flux_yoko.set_voltage(voltage=md.cur_V)
```

```python
preditor.A_to_flx(md.cur_A)
```

```python
%matplotlib widget
exp_cfg = {
    # "reset": "reset_bath",
    "init_pulse": "pi_amp",
    "probe_pulse": {
        "waveform": ml.get_waveform(
            "mist_waveform",
            {
                # "length": 5.0 / (2 * np.pi * md.rf_w) + 30.0,
                "length": 5.0 / (2 * np.pi * md.rf_w) + 0.3,
            },
        ),
        "ch": res_ch,
        "nqz": 2,
        # "freq": r_f,
        # "post_delay": 10 / (2 * np.pi * rf_w),
        # "gain": 0.3,
        "freq": md.r_f,
        "post_delay": 3 / (2 * np.pi * md.rf_w),
    },
    "readout": "readout_dpm",
    "sweep": {
        "gain": make_sweep(0.0, 0.5, 301),
    },
    "relax_delay": 50.0,  # us
}
cfg = ml.make_cfg(exp_cfg, reps=100, rounds=100)

mist_exp = ze.mist.MISTPowerDepExp()
_ = mist_exp.run(soc, soccfg, cfg)
```

```python
%matplotlib inline
fig = mist_exp.analyze(
    ac_coeff=md.ac_stark_coeff,
)
```

```python
filename = f"{qub_name}_mist_e"
savefig(fig, f"{result_dir}/exp_image/{md.cur_A * 1e3:.3f}mA/{filename}.png")
plt.close(fig)
mist_exp.save(
    filepath=os.path.join(database_path, f"{filename}@{md.cur_A * 1e3:.3f}mA"),
    # filepath=os.path.join(database_path, f"{filename}@{md.cur_V:.3f}V"),
    comment=make_comment(cfg),
)
```

```python

```
