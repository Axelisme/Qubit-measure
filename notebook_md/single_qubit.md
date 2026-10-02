---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
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
    version: 3.13.11
---

# Import Module

```python
%load_ext autoreload
import os
import time
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

%autoreload 2
import zcu_tools.experiment.v2 as ze
import zcu_tools.program.v2 as zp
from zcu_tools.resources.context import (
    ContextManager,
    MetaDict,
    ModuleLibrary,
)
from zcu_tools.resources.sample_table import (
    SampleTable,
    validate_sample_table_v2,
)
from zcu_tools.notebook.utils import dump_device_info, gc_collect, make_sweep, savefig
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.datafile import create_datafolder
from zcu_tools.experiment.cfg_assembler import CfgEnv, make_cfg
from zcu_tools.notebook import NotebookAdapter
```

# Create data/result folder

```python
chip_name = "Q5_2D"
res_name = "R1"
qub_name = "Q1"
# 4 2 5 1 3

result_dir = os.path.join("..", "result", chip_name, qub_name)
database_path = create_datafolder(
    database_dir=os.path.join("..", "Database"),
    name=os.path.join(chip_name, qub_name),
)

em = ContextManager(os.path.join(result_dir, "exps"))
ml = ModuleLibrary()
md = MetaDict()
```

# Connect to zcu216

```python
from zcu_tools.qick_remote import make_soc_proxy

soc, soccfg = make_soc_proxy("192.168.10.179", 8887)
print(soccfg)
```

```python
soc.get_sample_rates()
# soc.valid_sample_rates(tiletype='dac', tile=2)
```

# Predefine parameters

```python
md.res_ch = 0
# md.qub_0_1_ch = 11
# md.qub_1_4_ch = 2
md.qub_4_5_ch = 1
# md.qub_5_6_ch = 5
# md.lo_flux_ch = 15

md.ro_ch = 0
```

# Initialize devices

重新初始化前，先執行下方 Disconnect，關閉目前連線。

```python
import pyvisa
from zcu_tools.device import DeviceManager

device_manager = DeviceManager()
resource_manager = pyvisa.ResourceManager()
nb_adapter = NotebookAdapter(
    soc=soc, soccfg=soccfg, device_manager=device_manager
)
```

## YOKOGS200

```python

```

### Qubit Flux

```python
from zcu_tools.device.yoko import YOKOGS200

device_manager.close_device("flux_yoko", ignore_missing=True)

flux_yoko = YOKOGS200(
    address="USB0::0x0B21::0x0039::91WB18859::INSTR", rm=resource_manager
)
device_manager.register_device("flux_yoko", flux_yoko)

flux_yoko.set_mode("current", rampstep=1e-6)
# flux_yoko.set_mode("voltage", rampstep=1e-3)
```

```python
cur_value = flux_yoko.get_current()
cur_value * 1e3
```

```python
cur_value = 0.0e-3
cur_value = flux_yoko.set_current(cur_value)
cur_value * 1e3
```

### JPA Flux

```python
from zcu_tools.device.yoko import YOKOGS200

device_manager.close_device("jpa_yoko", ignore_missing=True)

jpa_yoko = YOKOGS200(
    address="USB0::0x0B21::0x0039::91T810992::INSTR", rm=resource_manager
)
device_manager.register_device("jpa_yoko", jpa_yoko)

jpa_yoko.set_mode("current", rampstep=1e-6)
```

```python
md.cur_jpa_A = jpa_yoko.get_current()
md.cur_jpa_A
```

```python
md.cur_jpa_A = jpa_yoko.set_current(0.0e-3)
md.cur_jpa_A * 1e3
```

## RF Source

```python

```

### JPA Pump

```python
from zcu_tools.device.sgs100a import RohdeSchwarzSGS100A

device_manager.close_device("jpa_sgs", ignore_missing=True)

jpa_sgs = RohdeSchwarzSGS100A(
    address="TCPIP0::192.168.10.89::inst0::INSTR", rm=resource_manager
)
device_manager.register_device("jpa_sgs", jpa_sgs)
```

```python
# jpa_sgs.set_power(-20)  # dBm
# jpa_sgs.IQ_off()
jpa_sgs.output_on()
# jpa_sgs.output_off()
```

```python
jpa_sgs.get_info()
```

# Initial Experiment Folder

```python
# ml, md = em.new_flux(label="20260504", clone_from=(ml, md))
# ml, md = em.new_flux(cur_value, unit="A")
# ml, md = em.new_flux(cur_value, clone_from=(ml, md), unit="A")
# ml, md = em.new_flux(cur_value, clone_from="032600_1.800mA", unit="A")
ml, md = em.use_flux(label="051115_2.000mA")
env = CfgEnv(md=md, ml=ml, device_manager=device_manager)
ml, md
```

切換 flux folder 後，重新執行本節。`make_cfg` 會讀取當下的裝置設定；更新裝置或 `md` 參數後，重新執行所需量測的 cfg cell。

# Lookback

```python
md.timeFly = 0.5
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        "readout": {
            "type": "readout/pulse",
            "pulse_cfg": {
                "waveform": {"style": "const", "length": 1.0},
                "ch": md.res_ch,
                "nqz": 2,
                "gain": 1.0,
                "freq": 6717,
                # "freq": md.r_f,
            },
            "ro_cfg": {
                "ro_ch": md.ro_ch,
                "ro_length": 1.5,  # us
                "trig_offset": md.timeFly - 0.1,  # us
            },
        },
    },
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, ze.LookbackCfg, env, overrides={'rounds': 500})

from zcu_tools.experiment.v2.lookback import LookbackAnalyzeOptions

lookback_exp = nb_adapter(ze.LookbackExp())
_ = lookback_exp.run(cfg)
```

```python
lookback_analysis = lookback_exp.analyze(
    LookbackAnalyzeOptions(ratio=0.1, smooth=1.0)
)
predict_offset = lookback_analysis.result.predict_offset
fig = lookback_analysis.figures["fit"]
predict_offset
```

```python
md.timeFly = float(predict_offset)
md.timeFly
```

```python
filename = f"lookback_{time.strftime('%H%M')}"
savefig(lookback_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
lookback_filepath = lookback_exp.save(
    Path(database_path) / f"{filename}@{em.label}.hdf5",
    unique=True,
    comment=f"timeFly = {md.timeFly}us",
)
```

# OneTone

```python
md.res_probe_len = 5.0  # us
ml.register_waveform(
    ro_waveform={
        "style": "flat_top",
        "raise_waveform": {"style": "cosine", "length": 0.1},
        "length": md.res_probe_len,  # us
    }
)
```

## Resonator Frequency

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        "readout": {
            "type": "readout/pulse",
            "pulse_cfg": {
                "waveform": "ro_waveform",
                "ch": md.res_ch,
                "nqz": 2,
                "gain": 0.05,
                # "gain": 1.0,
                "freq": 0.0,  # not used
            },
            "ro_cfg": {
                "ro_ch": md.ro_ch,
                "ro_length": md.res_probe_len - 0.1,  # us
                "trig_offset": md.timeFly + 0.05,  # us
            },
        }
    },
    # "sweep": make_sweep(5000, 6000, 1001),
    # "sweep": make_sweep(5540, 5562, 301),
    "sweep": make_sweep(md.r_f - 1.5 * md.rf_w, md.r_f + 1.5 * md.rf_w, 301),
    "relax_delay": 1.0,  # us
}
cfg = make_cfg(exp_cfg, ze.onetone.FreqCfg, env, overrides={'reps': 100, 'rounds': 100})

from zcu_tools.experiment.v2.onetone.freq import FreqAnalyzeOptions

res_freq_exp = nb_adapter(ze.onetone.FreqExp())
_ = res_freq_exp.run(cfg)
```

```python
res_freq_analysis = res_freq_exp.analyze(
    FreqAnalyzeOptions(model_type="hm", fit_bg_amp_slope=True)
)
f = res_freq_analysis.result.freq
kappa = res_freq_analysis.result.fwhm
params = res_freq_analysis.result.params
fig = res_freq_analysis.figures["fit"]
```

```python
md.r_f = f
md.rf_w = kappa
```

```python
filename = f"{res_name}_freq_{time.strftime('%m%d')}"
savefig(res_freq_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
res_freq_filepath = res_freq_exp.save(
    Path(database_path) / f"{filename}@{em.label}.hdf5",
    unique=True,
    comment=str(params),
)
```

## Power Dependence

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "init_pulse": "pi_amp",
        "readout": {
            "type": "readout/pulse",
            "pulse_cfg": {
                "waveform": ml.get_waveform("ro_waveform").with_updates(length=1.0),
                "ch": md.res_ch,
                "nqz": 2,
                "freq": 0.0,  # not used
                "gain": 0.0,  # not used
            },
            "ro_cfg": {
                "ro_ch": md.ro_ch,
                "ro_length": 1.0 - 0.1,  # us
                "trig_offset": md.timeFly + 0.05,  # us
            },
        },
    },
    "sweep": {
        "gain": make_sweep(0.001, 0.5, 101),
        "freq": make_sweep(md.r_f - 1.5 * md.rf_w, md.r_f + 1.5 * md.rf_w, 201),
        # "freq": make_sweep(5450, 5480, 201),
    },
    "relax_delay": 10.0,  # us
}
cfg = make_cfg(
    exp_cfg, ze.onetone.PowerDepCfg, env,
    overrides={"reps": 100, "rounds": 10, "earlystop_snr": 100.0},
)

res_gain_exp = nb_adapter(ze.onetone.PowerDepExp())
_ = res_gain_exp.run(cfg)
```

```python
filename = f"{res_name}_gain_{time.strftime('%H%M')}"
res_gain_filepath = res_gain_exp.save(
    Path(database_path) / f"{filename}@{em.label}.hdf5",
    unique=True,
)
```

## Flux dependence

量測完成後，執行選線 cell。拖曳兩條譜線，確認選點後按 Done，再執行下一格。Cancel 不會產生本次選線結果。

按 Done 後，可由下方 cell 查看並保存選線圖。若取消本次操作，先重新執行選線 cell。

```python
cur_value = flux_yoko.set_current(-5e-3)
cur_value * 1e3
```

```python
flux_yoko.get_current()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        "readout": {
            "type": "readout/pulse",
            "pulse_cfg": {
                "waveform": "ro_waveform",
                "ch": md.res_ch,
                "nqz": 2,
                "gain": 0.005,
                "freq": 0.0,  # not used
            },
            "ro_cfg": {
                "ro_ch": md.ro_ch,
                "ro_length": md.res_probe_len - 0.1,  # us
                "trig_offset": md.timeFly + 0.05,  # us
            },
        },
    },
    "dev": {"flux_yoko": {"label": "flux_dev", "mode": "current"}},
    "sweep": {
        "flux": make_sweep(3.57e-3, 3.61e-3, 101),
        # "flux": make_sweep(5.0, -5.0, 101),
        "freq": make_sweep(md.r_f - 1.0 * md.rf_w, md.r_f + 1.0 * md.rf_w, 101),
        # "freq": make_sweep(5755, 5775, 101),
    },
    "relax_delay": 1.0,  # us
}
cfg = make_cfg(exp_cfg, ze.onetone.FluxDepCfg, env, overrides={'reps': 1000, 'rounds': 1})

from zcu_tools.experiment.v2.onetone.flux_dep import FluxDepExp
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.notebook.experiments import FluxDepAnalyzer, FluxDepPickerOptions

res_flux_exp = nb_adapter(FluxDepExp())
flux_run = res_flux_exp.run(cfg)
```

```python
flux_filepath = res_flux_exp.save(
    Path(database_path) / f"{res_name}_flux",
    unique=True,
)
```

```python
flux_pick = FluxDepAnalyzer().start(flux_run, FluxDepPickerOptions())
# Select the two lines, then click Done before running the next cell.
```

```python
flux_record = flux_pick.record
if flux_record is None:
    raise RuntimeError("Select the two flux lines and click Done first")
flux_fig = flux_record.figures["pick"]
md.flx_half = flux_record.result.flux_half
md.flx_int = flux_record.result.flux_int
md.flx_period = flux_record.result.flux_period
md.flx_half, md.flx_int, md.flx_period
```

```python
flx_value = (1.0 - 0.5) * (md.flx_int - md.flx_half) / 0.5 + md.flx_half
cur_value = flux_yoko.set_current(flx_value)
cur_value * 1e3
```

## Set readout pulse

```python
ro_pulse_len = 1.0  # us
ml.register_module(
    readout_rf={
        "type": "readout/pulse",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": ro_pulse_len},
            "ch": md.res_ch,
            "nqz": 2,
            "freq": md.r_f,
            "gain": 0.2,
        },
        "ro_cfg": {
            "ro_ch": md.ro_ch,
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
    "modules": {
        "readout": "readout_rf",
    },
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
cfg = make_cfg(exp_cfg, ze.jpa.OneToneFluxCfg, env, overrides={'reps': 100, 'rounds': 10})


jpa_flux_onetone_exp = nb_adapter(ze.jpa.OneToneFluxExp())
_ = jpa_flux_onetone_exp.run(cfg)
```

```python
jpa_flux_onetone_exp.save(
    Path(os.path.join(database_path, "JPA_flux_onetone")),
    unique=True,
)
```

## JPA frequency

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "pi_pulse": "pi_len",
        "readout": "readout_rf",
    },
    "dev": {"jpa_sgs": {"label": "jpa_rf_dev"}},
    # "sweep": make_sweep(2 * md.r_f + 100, 2 * md.r_f + 500, step=0.25),
    "sweep": make_sweep(11750, 11800, 501),
    "relax_delay": 0.5,  # us
}
cfg = make_cfg(exp_cfg, ze.jpa.FreqCfg, env, overrides={'reps': 10000, 'rounds': 1})

jpa_freq_exp = nb_adapter(ze.jpa.FreqExp())
_ = jpa_freq_exp.run(cfg)
```

```python
%matplotlib inline
jpa_freq_analysis = jpa_freq_exp.analyze(None)
md.best_jpa_freq = jpa_freq_analysis.result.best_freq
fig = jpa_freq_analysis.figures["fit"]
md.best_jpa_freq
```

```python
filename = f"JPA_freq_{time.strftime('%m%d')}"
savefig(jpa_freq_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
jpa_freq_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
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
    "modules": {
        "reset": "reset_bath",
        "pi_pulse": "pi_amp",
        "readout": "readout_rf",
    },
    "dev": {
        "jpa_yoko": {
            "label": "jpa_flux_dev",
            "mode": "current",
        },
    },
    "sweep": make_sweep(-5.0e-3, 5.0e-3, 1001),
    "relax_delay": 0.5,  # us
}
cfg = make_cfg(exp_cfg, ze.jpa.FluxCfg, env, overrides={'reps': 10000, 'rounds': 1})

jpa_flux_exp = nb_adapter(ze.jpa.FluxExp())
_ = jpa_flux_exp.run(cfg)
```

```python
%matplotlib inline
jpa_flux_analysis = jpa_flux_exp.analyze(None)
md.best_jpa_flux = jpa_flux_analysis.result.best_flux
fig = jpa_flux_analysis.figures["fit"]
md.best_jpa_flux * 1e3
```

```python
filename = f"JPA_flux_{time.strftime('%m%d')}"
savefig(jpa_flux_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
jpa_flux_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

```python
md.cur_jpa_A = jpa_yoko.set_current(md.best_jpa_flux)
md.cur_jpa_A * 1e3
```

## JPA Power

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        "reset": "reset_bath",
        "pi_pulse": "pi_len",
        "readout": "readout_rf",
    },
    "dev": {"jpa_sgs": {"label": "jpa_rf_dev"}},
    "sweep": make_sweep(-20, 1, 501),
    "relax_delay": 0.5,  # us
}
cfg = make_cfg(exp_cfg, ze.jpa.PowerCfg, env, overrides={'reps': 10000, 'rounds': 1})

jpa_pdr_exp = nb_adapter(ze.jpa.PowerExp())
_ = jpa_pdr_exp.run(cfg)
```

```python
%matplotlib inline
jpa_pdr_analysis = jpa_pdr_exp.analyze(None)
md.best_jpa_power = jpa_pdr_analysis.result.best_power
fig = jpa_pdr_analysis.figures["fit"]
md.best_jpa_power
```

```python
filename = f"JPA_power_{time.strftime('%m%d')}"
savefig(jpa_pdr_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
jpa_pdr_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

```python
jpa_sgs.set_power(md.best_jpa_power)  # dBm
```

## Auto Optimize

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "pi_pulse": "pi_amp",
        "readout": "readout_dpm",
    },
    "dev": {
        "jpa_sgs": {"label": "jpa_rf_dev", "output": "on"},
        "jpa_yoko": {"label": "jpa_flux_dev", "mode": "current"},
    },
    "sweep": {
        "jpa_flux": make_sweep(-2.75e-3, -2e-3, 100),
        "jpa_freq": make_sweep(2 * md.r_f + 50, 2 * md.r_f + 500, 100),
        # "jpa_freq": make_sweep(11750, 11850, 100),
        "jpa_power": make_sweep(-25, -5, 50),
    },
    "relax_delay": 30.5,  # us
}
cfg = make_cfg(exp_cfg, ze.jpa.JPAOptCfg, env, overrides={'reps': 1000, 'rounds': 1, 'num_points': 10000})

jpa_opt_exp = nb_adapter(ze.jpa.AutoOptimizeExp())
_ = jpa_opt_exp.run(cfg)
```

```python
%matplotlib inline
jpa_opt_analysis = jpa_opt_exp.analyze(None)
md.best_jpa_flux = jpa_opt_analysis.result.best_flux
md.best_jpa_freq = jpa_opt_analysis.result.best_freq
md.best_jpa_power = jpa_opt_analysis.result.best_power
fig = jpa_opt_analysis.figures["fit"]
1e3 * md.best_jpa_flux, 1e-3 * md.best_jpa_freq, md.best_jpa_power
```

```python
filename = f"JPA_opt_{time.strftime('%m%d')}"
savefig(jpa_opt_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
jpa_opt_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

```python
jpa_sgs.set_frequency(1e6 * md.best_jpa_freq)  # Hz
jpa_sgs.set_power(md.best_jpa_power)  # dBm
md.cur_jpa_A = jpa_yoko.set_current(md.best_jpa_flux)
md.cur_jpa_A * 1e3
```

```python
jpa_sgs.output_on()
# jpa_sgs.output_off()
```

## JPA Check

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        "readout": "readout_dpm",
    },
    "dev": {"jpa_sgs": {"label": "jpa_rf_dev"}},
    "sweep": make_sweep(md.r_f - 20, md.r_f + 20, 101),
    "relax_delay": 0.5,  # us
}
cfg = make_cfg(exp_cfg, ze.jpa.CheckCfg, env, overrides={'reps': 1000, 'rounds': 5})

jpa_check_exp = nb_adapter(ze.jpa.CheckExp())
_ = jpa_check_exp.run(cfg)
```

```python
%matplotlib inline
jpa_check_analysis = jpa_check_exp.analyze(None)
fig = jpa_check_analysis.figures["fit"]
```

```python
filename = f"JPA_check_{time.strftime('%m%d')}"
savefig(jpa_check_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
jpa_check_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
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
    qub_const={"style": "const", "length": 2.0},
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
sample_table = SampleTable(os.path.join(result_dir, "samples.csv"))
validate_sample_table_v2(sample_table.samples, allow_empty=True)
```

```python
preditor = FluxoniumPredictor.from_file(os.path.join(result_dir, "params.json"))
preditor.flux_half = md.flx_half
preditor.flux_period = md.flx_period
# preditor.update_bias(md.flx_bias)
```

# Twotone Frequency

```python
cur_value = flux_yoko.set_current(2e-3)
print(f"{cur_value * 1e3:.2f} mA")

# ml, md = em.use_flux(label="031620_0.000mA")
ml, md = em.new_flux(value=cur_value, clone_from=(ml, md), unit="A")
```

```python
md.q_f = preditor.predict_freq(cur_value, transition=(0, 1))
# q_f = preditor.predict_freq(cur_value, transition=(0, 1))
md.q_f
```

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        # "init_pulse": "pi_len",
        "qub_pulse": {
            "type": "pulse",
            "waveform": ml.get_waveform("qub_flat", dict(length=0.1)),
            "ch": md.qub_4_5_ch,
            "nqz": 2,
            "gain": 0.05,
            # "mixer_freq": md.q_f,
            "freq": 0.0,  # not used
        },
        # "readout": "readout_rf",
        "readout": "readout_dpm",
    },
    # "sweep": make_sweep(md.q_f - 200, md.q_f + 200, step=0.25),
    "sweep": make_sweep(md.q_f - 20, md.q_f + 20, step=0.2),
    # "sweep": make_sweep(4000, 6000, step=1.00),
    "relax_delay": 0.5,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.FreqCfg, env, overrides={'reps': 1000, 'rounds': 100})

qub_freq_exp = nb_adapter(ze.twotone.FreqExp())
_ = qub_freq_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.freq import FreqAnalyzeOptions

qub_freq_analysis = qub_freq_exp.analyze(FreqAnalyzeOptions())
f = qub_freq_analysis.result.freq
kappa = qub_freq_analysis.result.fwhm
fig = qub_freq_analysis.figures["fit"]
f
```

```python
md.qf_w
```

```python
md.q_f = f
md.qf_w = kappa
```

```python
filename = f"{qub_name}_freq_{time.strftime('%m%d')}"
savefig(qub_freq_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
qub_freq_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"frequency = {f}MHz",
    unique=True,
)
```

```python
flx_bias = preditor.calculate_bias(cur_value, md.q_f)
flx_bias * 1e3
```

```python
md.flx_bias = flx_bias
preditor.update_bias(md.flx_bias)
```

# Rabi

```python
gc_collect()
```

## Length Rabi

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_10",
        "qub_pulse": {
            "type": "pulse",
            "waveform": "qub_flat",
            "ch": md.qub_4_5_ch,
            "nqz": 2,
            "freq": md.q_f,
            "gain": 0.3,
            # "gain": md.pi_gain,
            # "mixer_freq": md.q_f,
        },
        # "readout": "readout_rf",
        "readout": "readout_dpm",
    },
    "relax_delay": 10.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.03, 0.3, 101),
}
cfg = make_cfg(exp_cfg, ze.twotone.rabi.LenRabiCfg, env, overrides={'reps': 1000, 'rounds': 100})

qub_lenrabi_exp = nb_adapter(ze.twotone.rabi.LenRabiExp())
_ = qub_lenrabi_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.rabi.len_rabi import LenRabiAnalyzeOptions

qub_lenrabi_analysis = qub_lenrabi_exp.analyze(LenRabiAnalyzeOptions(decay=True))
md.pi_len = qub_lenrabi_analysis.result.pi_len
md.pi2_len = qub_lenrabi_analysis.result.pi2_len
md.rabi_f = qub_lenrabi_analysis.result.rabi_f
fig = qub_lenrabi_analysis.figures["fit"]
md.pi_len, md.pi2_len, md.rabi_f
```

```python
filename = f"{qub_name}_rabi_length_{time.strftime('%m%d')}"
savefig(qub_lenrabi_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
qub_lenrabi_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"pi len = {md.pi_len}us\npi/2 len = {md.pi2_len}us",
    unique=True,
)
```

```python
# pi_len = 1.0
# pi2_len = 0.5
qub_pulse = cfg.modules.qub_pulse
ml.register_module(
    pi_len=qub_pulse.with_updates(
        waveform={"length": md.pi_len},
        # pre_delay=0.005,
        # post_delay=0.005,
        desc="len pi pulse",
    ),
    pi2_len=qub_pulse.with_updates(
        waveform={"length": md.pi2_len},
        # pre_delay=0.005,
        # post_delay=0.005,
        desc="len pi/2 pulse",
    ),
)
```

## Amplitude Rabi

```python
%matplotlib widget
max_gain = min(5 * ml.get_module("pi_len", type=zp.PulseCfg).gain, 1.0)
exp_cfg = {
    "modules": {
        # "reset": "reset_10",
        "qub_pulse": {
            "type": "pulse",
            "waveform": ml.get_waveform("qub_cos", dict(length=1.01 * md.pi_len)),
            "ch": md.qub_4_5_ch,
            "nqz": 2,
            "freq": md.q_f,
            # "mixer_freq": md.q_f,
            # "mixer_freq": 0.5 * (reset_f + md.q_f),
            "gain": 0.0,  # not used
        },
        # "readout": "readout_rf",
        "readout": "readout_dpm",
    },
    "relax_delay": 10.5,  # us
    # "relax_delay": 5 * t1,
    "sweep": make_sweep(-0.3, 0.6, 51),
    # "sweep": make_sweep(0.0, max_gain, 51),
}
cfg = make_cfg(exp_cfg, ze.twotone.rabi.AmpRabiCfg, env, overrides={'reps': 1000, 'rounds': 100})

qub_amprabi_exp = nb_adapter(ze.twotone.rabi.AmpRabiExp())
_ = qub_amprabi_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.rabi.amp_rabi import AmpRabiAnalyzeOptions

qub_amprabi_analysis = qub_amprabi_exp.analyze(AmpRabiAnalyzeOptions(skip=1))
md.pi_gain = qub_amprabi_analysis.result.pi_amp
md.pi2_gain = qub_amprabi_analysis.result.pi2_amp
fig = qub_amprabi_analysis.figures["fit"]
md.pi_gain, md.pi2_gain
```

```python
filename = f"{qub_name}_rabi_amplitude_{time.strftime('%m%d')}"
savefig(qub_amprabi_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
qub_amprabi_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"pi gain = {md.pi_gain}\npi/2 gain = {md.pi2_gain}",
    unique=True,
)
```

```python
# pi_gain = 1.0
# pi2_gain = 0.5
qub_pulse = exp_cfg["modules"]["qub_pulse"]
ml.register_module(
    pi_amp={
        **qub_pulse,
        "gain": md.pi_gain,
        "desc": "amp pi pulse",
    },
    pi2_amp={
        **qub_pulse,
        "gain": md.pi2_gain,
        "desc": "amp pi/2 pulse",
    },
)
```

# Reset

```python
cur_value = flux_yoko.set_current(0.0e-3)
cur_value * 1e3
```

## One Pulse

```python
md.reset_f = md.r_f - md.q_f
md.reset_f
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "init_pulse": "pi_amp",
        "init_pulse": ml.get_module(
            "pi_amp",
            # dict(mixer_freq=0.5 * (md.reset_f + md.q_f)),
            type=zp.PulseCfg,
        ),
        "tested_reset": {
            "type": "reset/pulse",
            "pulse_cfg": {
                "waveform": ml.get_waveform("qub_flat").with_updates(length=5.0),
                "ch": md.qub_1_4_ch,
                "nqz": 2,
                "gain": 0.3,
                # "mixer_freq": md.reset_f,
                # "mixer_freq": 0.5 * (md.reset_f + md.q_f),
                "post_delay": 5 / (2 * np.pi * md.rf_w),
                "freq": 0.0,  # not used
            },
        },
        "readout": "readout_rf",
    },
    "sweep": make_sweep(md.reset_f - 50, md.reset_f + 50, step=0.5),
    # "sweep": make_sweep(50, 1500, 1001),
    # "relax_delay": 0.1,  # us
    "relax_delay": 1.0 * md.t1,
}
cfg = make_cfg(exp_cfg, ze.twotone.reset.single_tone.FreqCfg, env, overrides={'reps': 1000, 'rounds': 100})

single_reset_freq_exp = nb_adapter(ze.twotone.reset.single_tone.FreqExp())
_ = single_reset_freq_exp.run(cfg)
```

```python
%matplotlib inline
single_reset_freq_analysis = single_reset_freq_exp.analyze(None)
f = single_reset_freq_analysis.result.freq
kappa = single_reset_freq_analysis.result.fwhm
fig = single_reset_freq_analysis.figures["fit"]
f
```

```python
md.reset_f = f
```

```python
filename = f"{qub_name}_sidereset_freq_{time.strftime('%m%d')}"
savefig(single_reset_freq_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
single_reset_freq_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"frequency = {f}MHz",
    unique=True,
)
```

### Reset Length

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "init_pulse": "pi_amp",
        "tested_reset": {
            "type": "reset/pulse",
            "pulse_cfg": {
                "waveform": "qub_flat",
                "ch": md.qub_1_4_ch,
                "nqz": 1,
                "gain": 1.0,
                "freq": md.reset_f,
                # "mixer_freq": reset_f,
                # "mixer_freq": 0.5 * (reset_f + q_f),
                "post_delay": 5 / (2 * np.pi * md.rf_w),
            },
        },
        "readout": "readout_rf",
    },
    "sweep": make_sweep(0.1, 20.0, 50),
    "relax_delay": 30.5,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.reset.single_tone.LengthCfg, env, overrides={'reps': 1000, 'rounds': 100})

single_reset_length_exp = nb_adapter(ze.twotone.reset.single_tone.LengthExp())
_ = single_reset_length_exp.run(cfg)
```

```python
%matplotlib inline
single_reset_length_analysis = single_reset_length_exp.analyze(None)
fig = single_reset_length_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_sidereset_length_{time.strftime('%m%d')}"
savefig(single_reset_length_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
single_reset_length_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

### Set Reset Pulse

```python
tested_reset = cfg.modules.tested_reset.with_updates(
    pulse_cfg={"waveform": {"length": 25.0}}
)  # us
ml.register_module(
    reset_10=tested_reset.with_updates(desc="Reset with one pulse from 1 to 0"),
)
```

### Check Reset

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_10",
        "rabi_pulse": ml.get_module("pi_amp", type=zp.PulseCfg),
        # "rabi_pulse": {
        #     **ml.get_module("pi_amp"),
        #     # "mixer_freq": 0.5 * (md.reset_f + md.q_f),
        # },
        "tested_reset": "reset_10",
        # "tested_reset": ml.get_module(
        #     "reset_10",
        #     {
        #         "pulse_cfg": {
        #             # "mixer_freq": 0.5 * (md.reset_f + md.q_f),
        #         }
        #     },
        # ),
        "readout": "readout_rf",
    },
    "sweep": make_sweep(0.0, 1.0, 51),
    "relax_delay": 70.0,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.reset.RabiCheckCfg, env, overrides={'reps': 1000, 'rounds': 10})

single_reset_check_exp = nb_adapter(ze.twotone.reset.RabiCheckExp())
_ = single_reset_check_exp.run(cfg)
```

```python
%matplotlib inline
single_reset_check_analysis = single_reset_check_exp.analyze(None)
reset_check_fit = single_reset_check_analysis.result
fig = single_reset_check_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_sidereset_check_{time.strftime('%m%d')}"
savefig(single_reset_check_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
single_reset_check_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

## Two pulse

```python
jpa_sgs.output_off()
```

### Reset Freq 1

```python
reset1_trans = (1, 2)
md.reset_f1 = preditor.predict_freq(cur_value, transition=reset1_trans)
# md.reset_f1 = preditor.predict_freq(cur_value, transition=reset1_trans)
md.reset_f1
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_120",
        "init_pulse": "pi_amp",
        "qub_pulse": {
            "type": "pulse",
            "waveform": ml.get_waveform("qub_flat", dict(length=5.0)),
            "ch": md.res_ch,
            "nqz": 2,
            "gain": 1.0,
            # "mixer_freq": md.reset_f1,
            # "mixer_freq": md.q_f,
            "freq": 0.0,  # not used
        },
        "readout": "readout_rf",
        # "readout": "readout_dpm",
    },
    "sweep": make_sweep(md.reset_f1 - 55, md.reset_f1 + 55, step=0.1),
    # "sweep": make_sweep(4680, 4710, step=0.1),
    "relax_delay": 30.5,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.FreqCfg, env, overrides={'reps': 1000, 'rounds': 1000})

dualreset_freq1_exp = nb_adapter(ze.twotone.FreqExp())
_ = dualreset_freq1_exp.run(cfg)
```

```python
%matplotlib inline
dualreset_freq1_analysis = dualreset_freq1_exp.analyze(FreqAnalyzeOptions())
f = dualreset_freq1_analysis.result.freq
kappa = dualreset_freq1_analysis.result.fwhm
fig = dualreset_freq1_analysis.figures["fit"]
f
```

```python
md.reset_f1 = f
md.resetf1_w = kappa
```

```python
filename = f"{qub_name}_dualreset_freq1_{time.strftime('%m%d')}"
savefig(dualreset_freq1_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
dualreset_freq1_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"frequency = {f}MHz",
    unique=True,
)
```

```python
bias = preditor.calculate_bias(cur_value, md.reset_f1, transition=reset1_trans)
bias * 1e3
# bias = preditor.calculate_bias(cur_value, md.reset_f1, transition=reset1_trans)
# bias
```

```python
preditor.update_bias(bias)
```

### Reset Freq 2

```python
reset2_trans = (3, 1)
md.reset_f2 = abs(md.r_f + preditor.predict_freq(cur_value, transition=reset2_trans))
# md.reset_f2 = abs(md.r_f + preditor.predict_freq(cur_value, transition=reset2_trans))
md.reset_f2
```

```python
%matplotlib widget
dualreset_len = 5.0  # us
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        # "init_pulse": "pi_amp",
        "tested_reset": {
            "type": "reset/two_pulse",
            "pulse1_cfg": {
                "waveform": ml.get_waveform("qub_flat", dict(length=dualreset_len)),
                "ch": md.res_ch,
                "nqz": 1,
                "gain": 1.0,
                # "mixer_freq": md.reset_f1,
                # "mixer_freq": md.q_f,
                "freq": 0.0,  # not used
            },
            "pulse2_cfg": {
                "waveform": ml.get_waveform("qub_flat", dict(length=dualreset_len)),
                "ch": md.qub_1_4_ch,
                "nqz": 2,
                "gain": 1.0,
                # "mixer_freq": md.reset_f2,
                "post_delay": 5.0 / (2 * np.pi * md.rf_w),
                "freq": 0.0,  # not used
            },
        },
        "readout": "readout_rf",
    },
    "sweep": {
        "freq1": make_sweep(md.reset_f1 - 1.5, md.reset_f1 + 1.5, step=0.05),
        "freq2": make_sweep(md.reset_f2 - 15, md.reset_f2 - 3, step=0.05),
        # "freq1": make_sweep(1438, 1453, step=0.5),
        # "freq2": make_sweep(2610, 2620, step=0.1),
    },
    # "relax_delay": 5 / rf_w,  # us
    "relax_delay": 0.5,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.reset.dual_tone.FreqCfg, env, overrides={'reps': 100, 'rounds': 1000, 'method': 'hard'})

dualreset_freq2_exp = nb_adapter(ze.twotone.reset.dual_tone.FreqExp())
_ = dualreset_freq2_exp.run(cfg)
```

```python
%matplotlib inline
xlabal = f"|{reset1_trans[0]}, 0> - |{reset1_trans[1]}, 0>"
ylabal = f"|{reset2_trans[0]}, 0> - |{reset2_trans[1]}, 1>"
from zcu_tools.experiment.v2.twotone.reset.dual_tone.freq import FreqAnalyzeOptions as ResetFreqAnalyzeOptions

dualreset_freq2_analysis = dualreset_freq2_exp.analyze(ResetFreqAnalyzeOptions(smooth=0.5, xname=xlabal, yname=ylabal))
f1 = dualreset_freq2_analysis.result.freq1
f2 = dualreset_freq2_analysis.result.freq2
fig = dualreset_freq2_analysis.figures["fit"]
f1, f2
```

```python
reset_f1 = f1
reset_f2 = f2
```

```python
filename = f"{qub_name}_dualreset_both_freq_{time.strftime('%m%d')}"
savefig(dualreset_freq2_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
dualreset_freq2_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"frequency = ({reset_f1:.1f}, {reset_f2:.1f})MHz",
    unique=True,
)
```

### Set Dual Reset Pulse

```python
reset_f1 = 100
reset_f2 = 200
reset1_trans = (0, 3)
reset2_trans = (3, 1)
```

```python
dualreset_len = 30.0
tested_reset = cfg.modules.tested_reset
ml.register_module(
    reset_120=tested_reset.with_updates(
        pulse1_cfg={
            "waveform": {"length": dualreset_len},
            "freq": reset_f1,
        },
        pulse2_cfg={
            "waveform": {"length": dualreset_len},
            "freq": reset_f2,
        },
        desc=f"Reset with two pulse: {reset1_trans} and {reset2_trans}",
    ),
)
```

### Reset Gain

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        # "init_pulse": {
        #     **ml.get_waveform("qub_waveform"),
        #     "ch": qub_all_ch,
        #     "nqz": 2,
        #     "gain": 0.01,
        #     "mixer_freq": q_f,
        # },
        # "init_pulse": "pi_amp",
        "tested_reset": "reset_120",
        "readout": "readout_rf",
    },
    "sweep": {
        "gain1": make_sweep(0.0, 1.0, 51),
        "gain2": make_sweep(0.5, 1.0, 51),
    },
    "relax_delay": 0.5,  # us
    # "relax_delay": 3 * t1,
}
cfg = make_cfg(exp_cfg, ze.twotone.reset.dual_tone.PowerCfg, env, overrides={'reps': 100, 'rounds': 100})

dualreset_gain_exp = nb_adapter(ze.twotone.reset.dual_tone.PowerExp())
_ = dualreset_gain_exp.run(cfg)
```

```python
%matplotlib inline
xlabal = f"|{reset1_trans[0]}, 0> - |{reset1_trans[1]}, 0>"
ylabal = f"|{reset2_trans[0]}, 0> - |{reset2_trans[1]}, 1>"
from zcu_tools.experiment.v2.twotone.reset.dual_tone.power import PowerAnalyzeOptions as ResetPowerAnalyzeOptions

dualreset_gain_analysis = dualreset_gain_exp.analyze(ResetPowerAnalyzeOptions(xname=xlabal, yname=ylabal))
gain1 = dualreset_gain_analysis.result.gain1
gain2 = dualreset_gain_analysis.result.gain2
fig = dualreset_gain_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_dualreset_gain_{time.strftime('%m%d')}"
savefig(dualreset_gain_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
dualreset_gain_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"best gain = ({gain1:.1f}, {gain2:.1f})",
    unique=True,
)
```

```python
# gain1 = 0.5
# gain2 = 0.5
ml.update_module(
    "reset_120",
    override_cfg={
        "pulse1_cfg": {"gain": gain1},
        "pulse2_cfg": {"gain": gain2},
    },
)
```

### Reset Time

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        # "init_pulse": "pi_amp",
        "tested_reset": "reset_120",
        "readout": "readout_rf",
    },
    "sweep": make_sweep(0.05, 40.0, 51),
    "relax_delay": 0.5,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.reset.dual_tone.LengthCfg, env, overrides={'reps': 100, 'rounds': 100})

dualreset_len_exp = nb_adapter(ze.twotone.reset.dual_tone.LengthExp())
_ = dualreset_len_exp.run(cfg)
```

```python
filename = f"{qub_name}_dualreset_time_{time.strftime('%H%M')}"
dualreset_len_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

```python
dualreset_len = 4.0  # us
ml.update_module(
    "reset_120",
    override_cfg={
        "pulse1_cfg": {"waveform": {"length": dualreset_len}},
        "pulse2_cfg": {"waveform": {"length": dualreset_len}},
    },
)
```

### Check Reset

```python
exp_cfg = {
    "modules": {
        "reset": "reset_120",
        "rabi_pulse": "pi_amp",
        "tested_reset": "reset_120",
        "readout": "readout_rf",
    },
    "sweep": make_sweep(0.0, 1.0, 51),
    "relax_delay": 0.0,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.reset.RabiCheckCfg, env, overrides={'reps': 1000, 'rounds': 10})

dualreset_check_exp = nb_adapter(ze.twotone.reset.RabiCheckExp())
_ = dualreset_check_exp.run(cfg)
```

```python
filename = f"{qub_name}_dualreset_check_{time.strftime('%H%M')}"
dualreset_check_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

## Bath Reset

```python
gc_collect()
```

### Rabi frequency

```python
%matplotlib widget
pi_pulse = ml.get_module("pi_amp", type=zp.PulseCfg)
bathreset_qub_gain = min(1.0, 10 * pi_pulse.gain * pi_pulse.waveform.length * md.rf_w)

exp_cfg = {
    "modules": {
        "reset": "reset_bath",
        "qub_pulse": pi_pulse.with_updates(
            waveform="qub_flat",
            # gain=0.7,
            gain=bathreset_qub_gain,
        ),
        "readout": "readout_rf",
    },
    "relax_delay": 0.5,  # us
    # "relax_delay": 3 * md.t1,  # us
    "sweep": make_sweep(0.03, 2 / md.rf_w, 151),
}
cfg = make_cfg(exp_cfg, ze.twotone.rabi.LenRabiCfg, env, overrides={'reps': 100, 'rounds': 100})

rabifreq_exp = nb_adapter(ze.twotone.rabi.LenRabiExp())
_ = rabifreq_exp.run(cfg)
```

```python
%matplotlib inline
rabifreq_analysis = rabifreq_exp.analyze(LenRabiAnalyzeOptions(decay=True))
md.rabi_f = rabifreq_analysis.result.rabi_f
fig = rabifreq_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_rabi_freq_{time.strftime('%m%d')}"
savefig(rabifreq_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
rabifreq_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"pi len = {md.pi_len}us\npi/2 len = {md.pi2_len}us",
    unique=True,
)
```

### bath frequency

```python
gc_collect()
```

```python
%matplotlib widget
probe_len = 10.0
exp_cfg = {
    "modules": {
        "reset": "reset_bath",
        "init_pulse": "pi_amp",
        "tested_reset": {
            "type": "reset/bath",
            "cavity_tone_cfg": {
                "waveform": ml.get_waveform(
                    "qub_flat", dict(length=probe_len + 5.0 / (2 * np.pi * md.rf_w))
                ),
                "ch": md.res_ch,
                "nqz": 2,
                "post_delay": 5.0 / (2 * np.pi * md.rf_w),
                "freq": 0.0,  # not used
                "gain": 0.0,  # not used
            },
            "qubit_tone_cfg": ml.get_module(
                "pi_amp",
                {
                    "waveform": ml.get_waveform("qub_flat", {"length": probe_len}),
                    "gain": bathreset_qub_gain,
                    "pre_delay": 5.0 / (2 * np.pi * md.rf_w),
                },
            ),
            "pi2_cfg": ml.get_module(
                "pi2_amp",
                {
                    "phase": 90,
                },
            ),
        },
        "readout": "readout_rf",
        # "readout": "readout_dpm",
    },
    "sweep": {
        "freq": make_sweep(md.r_f - 1.2 * md.rabi_f, md.r_f - 0.8 * md.rabi_f, 51),
        "gain": make_sweep(0.4, 1.0, 51),
    },
    "relax_delay": 10.5,  # us
    # "relax_delay": 3 * md.t1,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.reset.bath.FreqGainCfg, env, overrides={'reps': 1000, 'rounds': 100})

bathreset_freq_exp = nb_adapter(ze.twotone.reset.bath.FreqGainExp())
_ = bathreset_freq_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.reset.bath.freq import FreqGainAnalyzeOptions

bathreset_freq_analysis = bathreset_freq_exp.analyze(FreqGainAnalyzeOptions(smooth=1))
md.bathreset_gain = bathreset_freq_analysis.result.gain
md.bathreset_freq = bathreset_freq_analysis.result.freq
fig = bathreset_freq_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_bathreset_freqgain_{time.strftime('%m%d')}"
savefig(bathreset_freq_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
bathreset_freq_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

### Length

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        "reset": "reset_bath",
        "init_pulse": "pi_amp",
        "tested_reset": {
            "type": "reset/bath",
            "cavity_tone_cfg": {
                "waveform": ml.get_waveform(
                    "qub_flat", {"length": 5.0 / (2 * np.pi * md.rf_w)}
                ),
                "ch": md.res_ch,
                "nqz": 2,
                "gain": md.bathreset_gain,
                "freq": md.bathreset_freq,
                "post_delay": 5.0 / (2 * np.pi * md.rf_w),
            },
            "qubit_tone_cfg": ml.get_module("pi_amp").with_updates(
                waveform=ml.get_waveform("qub_flat", {"length": 0.0}),
                gain=bathreset_qub_gain,
                pre_delay=5.0 / (2 * np.pi * md.rf_w),
            ),
            "pi2_cfg": ml.get_module(
                "pi2_amp",
                {
                    "phase": 90,
                },
            ),
        },
        "readout": "readout_rf",
        # "readout": "readout_dpm",
    },
    "sweep": make_sweep(0.05, 15.0, 201),
    "relax_delay": 10.5,  # us
    # "relax_delay": 3 * md.t1,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.reset.bath.LengthCfg, env, overrides={'reps': 100, 'rounds': 1000})

bathreset_len_exp = nb_adapter(ze.twotone.reset.bath.LengthExp())
_ = bathreset_len_exp.run(cfg)
```

```python
%matplotlib inline
bathreset_len_analysis = bathreset_len_exp.analyze(None)
fig = bathreset_len_analysis.figures["fit"]
```

```python
bath_reset_len = 10.0  # us
```

```python
filename = f"{qub_name}_bathreset_len_{time.strftime('%m%d')}"
savefig(bathreset_len_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
bathreset_len_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

### Phase

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        # "init_pulse": "pi_amp",
        "tested_reset": {
            "type": "reset/bath",
            "cavity_tone_cfg": {
                "waveform": ml.get_waveform(
                    "qub_flat", {"length": bath_reset_len + 5.0 / (2 * np.pi * md.rf_w)}
                ),
                "ch": md.res_ch,
                "nqz": 2,
                "gain": md.bathreset_gain,
                "freq": md.bathreset_freq,
                "post_delay": 5.0 / (2 * np.pi * md.rf_w),
            },
            "qubit_tone_cfg": ml.get_module("pi_amp").with_updates(
                waveform=ml.get_waveform("qub_flat", {"length": bath_reset_len}),
                gain=bathreset_qub_gain,
                pre_delay=5.0 / (2 * np.pi * md.rf_w),
            ),
            "pi2_cfg": "pi2_amp",
        },
        "readout": "readout_rf",
        # "readout": "readout_dpm",
    },
    "sweep": make_sweep(-360.0, 360.0, 201),
    # "relax_delay": 10.5,  # us
    "relax_delay": 3 * md.t1,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.reset.bath.PhaseCfg, env, overrides={'reps': 100, 'rounds': 1000})

bathreset_phase_exp = nb_adapter(ze.twotone.reset.bath.PhaseExp())
_ = bathreset_phase_exp.run(cfg)
```

```python
%matplotlib inline
bathreset_phase_analysis = bathreset_phase_exp.analyze(None)
max_phase = bathreset_phase_analysis.result.max_phase
min_phase = bathreset_phase_analysis.result.min_phase
fig = bathreset_phase_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_bathreset_phase_{time.strftime('%m%d')}"
savefig(bathreset_phase_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
bathreset_phase_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

### Set Bath reset pulse

```python
# bath_reset_len = 15.0  # us
tested_reset: zp.BathResetCfg = cfg.modules.tested_reset
ml.register_module(
    reset_bath=tested_reset.with_updates(
        pi2_cfg=dict(
            # "phase": 90,
            phase=max_phase,
        ),
        desc="Reset to Ground with cavity-assisted bath reset",
    ),
    reset_bath_e=tested_reset.with_updates(
        pi2_cfg=dict(
            # "phase": -90,
            phase=min_phase,
        ),
        desc="Reset to Excited with cavity-assisted bath reset",
    ),
)
```

### Check reset

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "rabi_pulse": "pi_amp",
        "tested_reset": "reset_bath",
        "pi_pulse": "pi_amp",
        "readout": "readout_rf",
        # "readout": "readout_dpm",
    },
    "sweep": make_sweep(0.0, 0.4, 51),
    # "relax_delay": 0.5,  # us
    "relax_delay": 5 * md.t1,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.reset.RabiCheckCfg, env, overrides={'reps': 100, 'rounds': 100})

bathreset_rabicheck_exp = nb_adapter(ze.twotone.reset.RabiCheckExp())
_ = bathreset_rabicheck_exp.run(cfg)
```

```python
%matplotlib inline
bathreset_rabicheck_analysis = bathreset_rabicheck_exp.analyze(None)
reset_check_fit = bathreset_rabicheck_analysis.result
fig = bathreset_rabicheck_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_bathreset_check_{time.strftime('%m%d')}"
savefig(bathreset_rabicheck_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
bathreset_rabicheck_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

# TwoTone Flux Dependence

```python
cur_value = flux_yoko.set_current(-5e-3)
# cur_value = flux_yoko.set_current(md.flx_int)
cur_value * 1e3
```

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_120",
        "qub_pulse": {
            "type": "pulse",
            "waveform": ml.get_waveform("qub_flat", dict(length=1.0)),
            "ch": md.qub_4_5_ch,
            "nqz": 2,
            "gain": 0.1,
            # "mixer_freq": md.q_f,
            "freq": 0.0,  # not used
        },
        # "readout": "readout_rf",
        "readout": "readout_dpm",
    },
    "dev": {
        "flux_yoko": {"label": "flux_dev", "mode": "current"},
    },
    "sweep": {
        "flux": make_sweep(-5.0e-3, 5e-3, 151),
        # "flux": make_sweep(md.flx_int, -2e-3, 81),
        "freq": make_sweep(4000, 6000, step=1.0),
    },
    "relax_delay": 0.5,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.FreqFluxCfg, env, overrides={'reps': 2000, 'rounds': 40, 'fail_retry': 3})

qub_flux_exp = nb_adapter(ze.twotone.FreqFluxExp())
qub_flux_run = qub_flux_exp.run(cfg)
```

```python
filename = f"{qub_name}_flux_{time.strftime('%H%M')}"
qub_flux_exp.save(Path(database_path) / filename, unique=True)
```

```python
%matplotlib widget
from zcu_tools.experiment.v2.twotone.fluxdep import FreqFluxCfg, FreqFluxResult

qub_flux_analyzer = FluxDepAnalyzer[FreqFluxCfg, FreqFluxResult]()
qub_flux_picker = qub_flux_analyzer.start(
    qub_flux_run,
    FluxDepPickerOptions(
        # flux_half=md.flx_half,
        # flux_int=md.flx_int,
    ),
)
```

```python
# Click Done in the picker before reading the committed selection.
qub_flux_selection = qub_flux_picker.record
if qub_flux_selection is None:
    raise RuntimeError("Complete the flux picker with Done first")
md.flx_half = qub_flux_selection.result.flux_half
md.flx_int = qub_flux_selection.result.flux_int
md.flx_period = qub_flux_selection.result.flux_period
md.flx_half, md.flx_int
```

# Other TwoTone

```python
gc_collect()
```

## Power dependence

```python
exp_cfg = {
    "modules": {
        # "reset": "reset_120",
        "qub_pulse": {
            "type": "pulse",
            "waveform": ml.get_waveform("qub_flat", {"length": 0.3}),
            "ch": md.qub_1_4_ch,
            "nqz": 2,
            # "mixer_freq": md.q_f,
            "gain": 0.0,  # not used
            "freq": 0.0,  # not used
        },
        # "qub_pulse": "pi_amp",
        "readout": "readout_rf",
        # "readout": "readout_dpm",
    },
    "sweep": {
        "gain": make_sweep(0.1, 1.0, 30),
        # "freq": make_sweep(1700, 2000, 30),
        "freq": make_sweep(1000, 5000, step=1.5),
    },
    "relax_delay": 0.5,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.PowerCfg, env, overrides={'reps': 100, 'rounds': 100})

qub_pdr_exp = nb_adapter(ze.twotone.PowerExp())
_ = qub_pdr_exp.run(cfg)
```

```python
qub_pdr_exp.save(
    Path(
        os.path.join(database_path, f"{qub_name}_pdr@{em.label}")
    ),
    unique=True,
)
```

## CKP

```python
%matplotlib widget
ckp_res_gain = 0.015
ckp_qub_len = 1.5
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "pi_pulse": "pi_amp",
        "res_pulse": ml.get_module(
            "readout_rf", type=zp.PulseReadoutCfg
        ).pulse_cfg.with_updates(
            waveform=dict(length=5.1 / (2 * np.pi * md.rf_w) + ckp_qub_len),
            gain=ckp_res_gain,
        ),
        "qub_pulse": {
            "type": "pulse",
            "waveform": ml.get_waveform("qub_flat", {"length": ckp_qub_len}),
            "ch": md.qub_1_4_ch,
            "nqz": 2,
            "gain": 0.01,
            "pre_delay": 5.0 / (2 * np.pi * md.rf_w),
            "post_delay": 3.1 / (2 * np.pi * md.rf_w),
            "freq": 0.0,  # not used
        },
        "readout": "readout_rf",
        # "readout": "readout_dpm",
    },
    "sweep": {
        "res_freq": make_sweep(md.r_f - 1.5 * md.rf_w, md.r_f + 1.5 * md.rf_w, 101),
        "qub_freq": make_sweep(md.q_f - 10, md.q_f + 5, 101),
    },
    "relax_delay": 10.1,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.CKP_Cfg, env, overrides={'reps': 100, 'rounds': 100})

ckp_exp = nb_adapter(ze.twotone.CKP_Exp())
_ = ckp_exp.run(cfg)
```

```python
%matplotlib inline
ckp_analysis = ckp_exp.analyze(None)
chi = ckp_analysis.result.chi
kappa = ckp_analysis.result.kappa
center_freq = ckp_analysis.result.res_freq
fig = ckp_analysis.figures["fit"]
```

```python
md.chi = chi
md.rf_w = kappa
md.readout_f = center_freq
```

```python
filename = f"{qub_name}_ckp_{time.strftime('%m%d')}"
savefig(ckp_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
ckp_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

## Dispersive Shift

```python
jpa_sgs.output_off()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_10",
        "qub_pulse": "pi_amp",
        "readout": ml.get_module(
            "readout_rf",
            {
                "pulse_cfg": {
                    "gain": 0.01,
                }
            },
        ),
    },
    "sweep": make_sweep(md.r_f - 2.25 * md.rf_w, md.r_f + 2.0 * md.rf_w, step=0.1),
    "relax_delay": 30.5,  # us
    # "relax_delay": 2 * t1, # us
}
cfg = make_cfg(exp_cfg, ze.twotone.DispersiveCfg, env, overrides={'reps': 1000, 'rounds': 1000})

dispersive_shift_exp = nb_adapter(ze.twotone.DispersiveExp())
_ = dispersive_shift_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.dispersive import DispersiveAnalyzeOptions

dispersive_shift_analysis = dispersive_shift_exp.analyze(DispersiveAnalyzeOptions())
md.chi = dispersive_shift_analysis.result.chi
rf_w = dispersive_shift_analysis.result.avg_fwhm
fig = dispersive_shift_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_dispersive_gain{cfg.modules.readout.pulse_cfg.gain:.3f}_{time.strftime('%m%d')}"
savefig(dispersive_shift_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
dispersive_shift_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"chi = {md.chi:.3g} MHz, kappa = {rf_w:.3g} MHz",
    unique=True,
)
```

## AC Stark Shift

```python
%matplotlib widget
ac_qub_len = 5.0  # us
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "stark_pulse1": {
            "type": "pulse",
            "waveform": {
                "style": "const",
                "length": 5.1 / (2 * np.pi * md.rf_w) + ac_qub_len,
            },
            "ch": md.res_ch,
            "nqz": 2,
            "freq": md.r_f,
            "gain": 0.0,  # not used
        },
        "stark_pulse2": {
            "type": "pulse",
            "waveform": ml.get_waveform("qub_flat", {"length": ac_qub_len}),
            "ch": md.qub_1_4_ch,
            "nqz": 2,
            "gain": 0.01,
            # "mixer_freq": md.q_f,
            # "mixer_freq": 0.5 * (reset_f + q_f),
            "pre_delay": 5.0 / (2 * np.pi * md.rf_w),
            "post_delay": 3.1 / (2 * np.pi * md.rf_w),
            "freq": 0.0,  # not used
        },
        "readout": "readout_rf",
        # "readout": "readout_dpm",
    },
    "sweep": {
        "gain": make_sweep(0.0, 0.01, 101),
        "freq": make_sweep(md.q_f - 5.0, md.q_f + 1.0, step=0.05),
    },
    "relax_delay": 0.5,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.AcStarkCfg, env, overrides={'reps': 1000, 'rounds': 10, 'earlystop_snr': 50})

ac_stark_exp = nb_adapter(ze.twotone.AcStarkExp())
_ = ac_stark_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.ac_stark import AcStarkAnalyzeOptions

ac_stark_analysis = ac_stark_exp.analyze(AcStarkAnalyzeOptions(chi=md.chi, kappa=md.rf_w, deg=1, cutoff=0.01))
md.ac_stark_coeff = ac_stark_analysis.result.ac_coeff
fig = ac_stark_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_ac_stark_freq{cfg.modules.stark_pulse1.freq:.3f}MHz_{time.strftime('%m%d')}"
savefig(ac_stark_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
ac_stark_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    # comment=f"ac_stark_coeff = {md.ac_stark_coeff:.3g} MHz",
    unique=True,
)
```

## All XY

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        "reset": "reset_bath",
        "X180_pulse": "pi_amp",
        "X90_pulse": "pi2_amp",
        "Y180_pulse": ml.get_module("pi_amp", {"phase": 90}),
        "Y90_pulse": ml.get_module("pi2_amp", {"phase": 90}),
        # "readout": "readout_rf",
        "readout": "readout_dpm",
    },
    "relax_delay": 10.5,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.AllXYCfg, env, overrides={'reps': 1000, 'rounds': 1000})

allxy_exp = nb_adapter(ze.twotone.AllXY_Exp())
_ = allxy_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.allxy import AllXYAnalyzeOptions

allxy_analysis = allxy_exp.analyze(AllXYAnalyzeOptions())
fig = allxy_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_allxy_{time.strftime('%m%d')}"
savefig(allxy_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
allxy_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

## RB

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        "reset": "reset_bath",
        "X90_pulse": "pi2_amp",
        "X180_pulse": "pi_amp",
        "readout": "readout_dpm",
    },
    "sweep": make_sweep(0, 500, 50, force_int=True),
    "seed": 0,
    "n_seeds": 100,
    "relax_delay": 10.5,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.RBCfg, env, overrides={'reps': 100, 'rounds': 100})

rb_exp = nb_adapter(ze.twotone.RB_Exp())
_ = rb_exp.run(cfg)
```

```python
%matplotlib inline
rb_analysis = rb_exp.analyze(None)
fig = rb_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_rb_{time.strftime('%m%d')}"
savefig(rb_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
rb_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

## Zig-Zag

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_10",
        "X90_pulse": "pi2_amp",
        # "X180_pulse": "pi_amp",
        "readout": "readout_rf",
        # "readout": "readout_dpm",
    },
    "n_times": 10,
    "relax_delay": 30.5,  # us
}
repeat_on = "X90_pulse"
cfg = make_cfg(exp_cfg, ze.twotone.ZigZagCfg, env, overrides={'reps': 1000, 'rounds': 100, 'repeat_on': repeat_on})

zigzag_exp = nb_adapter(ze.twotone.ZigZagExp())
_ = zigzag_exp.run(cfg)
```

```python
filename = f"{qub_name}_zigzag_{repeat_on}_{time.strftime('%m%d')}"
zigzag_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

### Sweep parameters

```python
%matplotlib widget
repeat_on = "X180_pulse"

exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "X90_pulse": ml.get_module(
            "pi2_amp",
            {
                "freq": md.q_f,
                # "mixer_freq": md.q_f,
            },
        ),
        "X180_pulse": ml.get_module(
            "pi_amp",
            {
                "freq": md.q_f,
                # "mixer_freq": md.q_f,
            },
        ),
        "readout": "readout_rf",
        # "readout": "readout_dpm",
    },
    "sweep": {},
    "n_times": 6,
    "relax_delay": 100.5,  # us
}
if repeat_on == "X90_pulse":
    exp_cfg["sweep"].update(
        gain=make_sweep(md.pi2_gain * 0.8, md.pi2_gain * 1.2, 101)
        # gain=make_sweep(0.2, 0.7, 101)
    )
elif repeat_on == "X180_pulse":
    exp_cfg["sweep"].update(gain=make_sweep(md.pi_gain * 0.8, md.pi_gain * 1.2, 101))
else:
    raise ValueError(f"Invalid repeat_on: {repeat_on}")
cfg = make_cfg(exp_cfg, ze.twotone.ZigZagScanCfg, env, overrides={'reps': 100, 'rounds': 100, 'repeat_on': repeat_on})


zigzag_scan_exp = nb_adapter(ze.twotone.ZigZagScanExp())
_ = zigzag_scan_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.zigzag_sweep import ZigZagScanAnalyzeOptions

zigzag_scan_analysis = zigzag_scan_exp.analyze(ZigZagScanAnalyzeOptions(find_range=(None, None)))
best_x = zigzag_scan_analysis.result.min_value
fig = zigzag_scan_analysis.figures["fit"]
```

```python
gc_collect()
```

```python
filename = f"{qub_name}_zigzag_sweep_{repeat_on}_{time.strftime('%m%d')}"
savefig(zigzag_scan_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
zigzag_scan_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

```python
if repeat_on == "X90_pulse":
    md.pi2_gain = best_x
    ml.update_module(
        "pi2_amp",
        override_cfg={
            "gain": md.pi2_gain,
            "freq": md.q_f,
            # "mixer_freq": md.q_f,
        },
    )
elif repeat_on == "X180_pulse":
    md.pi_gain = best_x
    ml.update_module(
        "pi_amp",
        override_cfg={
            "gain": md.pi_gain,
            "freq": md.q_f,
            # "mixer_freq": md.q_f
        },
    )
else:
    raise ValueError("Invalid repeat_on value")
```

# Optimize Readout

```python
jpa_sgs.output_off()
# jpa_sgs.output_on()
jpa_sgs.get_info()
```

## Frequency tuning

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "qub_pulse": "pi_amp",
        "readout": ml.get_module(
            "readout_rf",
            {
                "pulse_cfg": {
                    "waveform": {"length": 1.1},
                    "gain": 0.02,
                },
                "ro_cfg": {
                    "ro_length": 1.0,
                },
            },
        ),
    },
    # "relax_delay": 10.5,  # us
    "relax_delay": 5 * md.t1,  # us
    "sweep": make_sweep(md.r_f - 1.5 * md.rf_w, md.r_f + 1.5 * md.rf_w, step=0.1),
    # "sweep": make_sweep(5450, 5460, step=0.1),
}
cfg = make_cfg(exp_cfg, ze.twotone.ro_optimize.FreqCfg, env, overrides={'reps': 1000, 'rounds': 100})

opt_ro_freq_exp = nb_adapter(ze.twotone.ro_optimize.FreqExp())
_ = opt_ro_freq_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.ro_optimize.freq import FreqAnalyzeOptions as ROFreqAnalyzeOptions

opt_ro_freq_analysis = opt_ro_freq_exp.analyze(ROFreqAnalyzeOptions(smooth=2))
best_freq = opt_ro_freq_analysis.result.best_freq
fig = opt_ro_freq_analysis.figures["fit"]
best_freq
```

```python
md.best_ro_freq = best_freq
```

```python
filename = f"{qub_name}_ro_opt_freq_{time.strftime('%m%d')}"
savefig(opt_ro_freq_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
opt_ro_freq_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"optimal frequency = {best_freq:.1f}MHz",
    unique=True,
)
```

```python
# md.best_ro_freq = md.r_f
```

## Power tuning

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "qub_pulse": "pi_amp",
        "readout": ml.get_module(
            "readout_rf",
            {
                "pulse_cfg": {
                    "waveform": {"length": 1.1},
                    "freq": md.best_ro_freq,
                },
                "ro_cfg": {
                    "ro_length": 1.0,
                    "ro_freq": md.best_ro_freq,
                },
            },
        ),
    },
    # "relax_delay": 10.5,  # us
    "relax_delay": 5 * md.t1,  # us
    "sweep": make_sweep(0.001, 0.2, 101),
}
cfg = make_cfg(exp_cfg, ze.twotone.ro_optimize.PowerCfg, env, overrides={'reps': 1000, 'rounds': 100})

opt_ro_pdr_exp = nb_adapter(ze.twotone.ro_optimize.PowerExp())
_ = opt_ro_pdr_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.ro_optimize.power import PowerAnalyzeOptions as ROPowerAnalyzeOptions

opt_ro_pdr_analysis = opt_ro_pdr_exp.analyze(ROPowerAnalyzeOptions(penalty_ratio=0.5))
best_gain = opt_ro_pdr_analysis.result.best_gain
fig = opt_ro_pdr_analysis.figures["fit"]
best_gain
```

```python
md.best_ro_gain = best_gain
```

```python
filename = f"{qub_name}_ro_opt_gain_{time.strftime('%m%d')}"
savefig(opt_ro_pdr_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
opt_ro_pdr_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"optimal power = {best_gain:.2f}",
    unique=True,
)
```

```python
# md.best_ro_gain = 0.2
```

## Freq-Power tuning

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "qub_pulse": "pi_amp",
        "readout": ml.get_module(
            "readout_rf",
            {
                "pulse_cfg": {
                    "waveform": {"length": 1.1},
                },
                "ro_cfg": {
                    "ro_length": 1.0,
                },
            },
        ),
    },
    # "relax_delay": 10.5,  # us
    "relax_delay": 5 * md.t1,  # us
    "sweep": {
        "freq": make_sweep(
            md.best_ro_freq - 0.5 * md.rf_w, md.best_ro_freq + 0.5 * md.rf_w, 31
        ),
        "gain": make_sweep(0.0, 0.2, 31),
    },
}
cfg = make_cfg(exp_cfg, ze.twotone.ro_optimize.FreqGainCfg, env, overrides={'reps': 100, 'rounds': 1000})

opt_ro_freq_pdr_exp = nb_adapter(ze.twotone.ro_optimize.FreqGainExp())
_ = opt_ro_freq_pdr_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.ro_optimize.freq_gain import FreqGainAnalyzeOptions as ROFreqGainAnalyzeOptions

opt_ro_freq_pdr_analysis = opt_ro_freq_pdr_exp.analyze(ROFreqGainAnalyzeOptions())
best_freq = opt_ro_freq_pdr_analysis.result.best_freq
best_gain = opt_ro_freq_pdr_analysis.result.best_gain
fig = opt_ro_freq_pdr_analysis.figures["fit"]
best_freq, best_gain
```

```python
md.best_ro_freq = best_freq
md.best_ro_gain = best_gain
```

```python
filename = f"{qub_name}_ro_opt_gain_{time.strftime('%m%d')}"
savefig(opt_ro_freq_pdr_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
opt_ro_freq_pdr_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"optimal freq = {best_freq:.2f}, power = {best_gain:.2f}",
    unique=True,
)
```

## Readout Length tuning

```python
%matplotlib widget
# pdr_max = 0.6
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "qub_pulse": "pi_amp",
        "readout": ml.get_module(
            "readout_rf",
            {
                "pulse_cfg": {
                    "freq": md.best_ro_freq,
                    "gain": md.best_ro_gain,
                },
                "ro_cfg": {
                    "ro_freq": md.best_ro_freq,
                },
            },
        ),
    },
    # "relax_delay": 10.5,  # us
    "relax_delay": 5 * md.t1,  # us
    "sweep": make_sweep(0.01, 3.5, 51),
}
cfg = make_cfg(exp_cfg, ze.twotone.ro_optimize.LengthCfg, env, overrides={'reps': 10000, 'rounds': 1})

opt_ro_len_exp = nb_adapter(ze.twotone.ro_optimize.LengthExp())
_ = opt_ro_len_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.ro_optimize.length import LengthAnalyzeOptions as ROLengthAnalyzeOptions

opt_ro_len_analysis = opt_ro_len_exp.analyze(ROLengthAnalyzeOptions(t0=5.0))
best_length = opt_ro_len_analysis.result.best_length
fig = opt_ro_len_analysis.figures["fit"]
best_length
```

```python
md.best_ro_length = best_length
```

```python
filename = f"{qub_name}_ro_opt_length_{time.strftime('%m%d')}"
savefig(opt_ro_len_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
opt_ro_len_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"optimal readout length = {best_length:.2f}us",
    unique=True,
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
                "ro_freq": md.best_ro_freq,
            },
            "desc": "Readout with largest dispersive shift",
        },
        type=zp.PulseReadoutCfg,
    )
)
```

## Auto Optimize

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "qub_pulse": "pi_amp",
        "readout": "readout_rf",
    },
    # "relax_delay": 30.5,  # us
    "relax_delay": 3 * md.t1,  # us
    "sweep": {
        "freq": make_sweep(md.r_f - 0.2 * md.rf_w, md.r_f + 0.2 * md.rf_w, 51),
        "gain": make_sweep(0.1, 0.25, 51),
        "length": make_sweep(5.0, 10.0, 51),
    },
}
cfg = make_cfg(exp_cfg, ze.twotone.ro_optimize.AutoOptCfg, env, overrides={'reps': 1000, 'rounds': 10, 'num_points': 1001})

auto_opt_ro_exp = nb_adapter(ze.twotone.ro_optimize.AutoOptExp())
_ = auto_opt_ro_exp.run(cfg)
```

```python
%matplotlib inline
auto_opt_ro_analysis = auto_opt_ro_exp.analyze(None)
md.best_ro_freq = auto_opt_ro_analysis.result.best_freq
md.best_ro_gain = auto_opt_ro_analysis.result.best_gain
md.best_ro_length = auto_opt_ro_analysis.result.best_length
fig = auto_opt_ro_analysis.figures["fit"]
md.best_ro_freq, md.best_ro_gain, md.best_ro_length
```

```python
filename = f"{qub_name}_ro_opt_auto_{time.strftime('%m%d')}"
auto_opt_ro_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
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
                "length": md.best_ro_length + 0.1,
                # "length": 0.25 * md.t1_with_tone + 0.1,
            },
        },
        "ro_cfg": {
            "ro_length": md.best_ro_length,
            # "ro_length": 0.25 * md.t1_with_tone + 0.1,
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
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "pi2_pulse": "pi2_amp",
        "readout": "readout_dpm",
    },
    # "relax_delay": 10.5,  # us
    "relax_delay": 5 * md.t1,  # us
    "sweep": make_sweep(0.0, 0.4, 101),  # us
    # "sweep": make_sweep(0.0, 1.5 * md.t2r, 101),  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.time_domain.T2RamseyCfg, env, overrides={'reps': 1000, 'rounds': 100})

activate_detune = 0.05 / cfg.sweep.length.step

cfg = cfg.with_updates(detune=activate_detune)

t2ramsey_exp = nb_adapter(ze.twotone.time_domain.T2RamseyExp())
t2ramsey_run = t2ramsey_exp.run(cfg)
true_detune = t2ramsey_run.result.true_activate_detune
if true_detune is None:
    raise ValueError("The run did not return its applied detuning")
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.time_domain.t2ramsey import T2RamseyAnalyzeOptions

t2ramsey_analysis = t2ramsey_exp.analyze(T2RamseyAnalyzeOptions(fit_fringe=True))
md.t2r = t2ramsey_analysis.result.t2r
md.t2r_err = t2ramsey_analysis.result.t2r_err
detune = t2ramsey_analysis.result.detune
fig = t2ramsey_analysis.figures["fit"]
print(f"real detune: {(detune - true_detune) * 1e3:.1f}kHz")
```

```python
filename = f"{qub_name}_t2ramsey_{time.strftime('%m%d')}"
savefig(t2ramsey_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
t2ramsey_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"activate detune = {true_detune:.3f}MHz\nt2r = {md.t2r:.3f}us",
    unique=True,
)
```

```python
md.q_f = cfg.modules.pi2_pulse.freq + true_detune - detune
md.q_f
```

## T1

調整等待時間的範圍與點數後執行量測，再選擇擬合要略過的資料點。以下分別示範一般 T1、With Tone 與 With Sweep Tone。

```python
from zcu_tools.experiment.v2.twotone.time_domain.t1 import T1AnalyzeOptions, T1Exp
from zcu_tools.notebook import NotebookAdapter

exp_cfg = {
    "modules": {
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
    },
    "relax_delay": 30.5,  # us
    # "relax_delay": 5 * md.t1,  # us
    # "sweep": make_sweep(0.01, 10.1, 101),
    "sweep": make_sweep(0.01, 5 * md.t1, 51),
    "uniform": False,
}
cfg = make_cfg(exp_cfg, ze.twotone.time_domain.T1Cfg, env, overrides={'reps': 1000, 'rounds': 100})

t1_exp = nb_adapter(T1Exp())
_ = t1_exp.run(cfg)
```

```python
t1_record = t1_exp.analyze(T1AnalyzeOptions(dual_exp=False, skip=1))
analysis = t1_record.result
md.t1, md.t1err = analysis.t1, analysis.t1_err
fig = t1_record.figures["fit"]
md.t1
```

```python
filename = f"{qub_name}_t1_{time.strftime('%m%d')}"
savefig(fig, os.path.join(em.flux_dir, "image", f"{filename}.png"))
t1_filepath = t1_exp.save(
    Path(database_path) / f"{filename}@{em.label}",
    unique=True,
    comment=f"t1 = {md.t1:.3f}us",
)
```

### With Tone

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "pi_pulse": "pi_amp",
        "test_pulse": ml.get_module(
            "readout_dpm", type=zp.PulseReadoutCfg
        ).pulse_cfg.with_updates(post_delay=5.0 / (2 * np.pi * md.rf_w)),
        "readout": "readout_dpm",
    },
    "relax_delay": 50.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(1.0, 20, 101),
    # "sweep": make_sweep(0.01*t1, 5 * t1, 51),
}
cfg = make_cfg(exp_cfg, ze.twotone.time_domain.T1WithToneCfg, env, overrides={'reps': 1000, 'rounds': 10})

t1_with_tone_exp = nb_adapter(ze.twotone.time_domain.T1WithToneExp())
_ = t1_with_tone_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.time_domain.t1 import T1WithToneAnalyzeOptions

t1_with_tone_analysis = t1_with_tone_exp.analyze(T1WithToneAnalyzeOptions(dual_exp=False))
md.t1_with_tone = t1_with_tone_analysis.result.t1
fig = t1_with_tone_analysis.figures["fit"]
md.t1_with_tone
```

```python
filename = f"{qub_name}_t1_with_tone_gain{cfg.modules.test_pulse.gain:.2f}_{time.strftime('%m%d')}"
savefig(t1_with_tone_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
t1_with_tone_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"t1 = {md.t1_with_tone:.3f}us",
    unique=True,
)
```

### With Sweep Tone

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_120",
        "pi_pulse": "pi_amp",
        "test_pulse": {
            "type": "pulse",
            "waveform": "mist_waveform",
            "ch": md.res_ch,
            "nqz": 2,
            "freq": md.r_f,
            "post_delay": 3.0 / (2 * np.pi * md.rf_w),
            "gain": 0.0,  # not used
        },
        "readout": "readout_dpm",
    },
    "relax_delay": 30.0,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": {
        "gain": make_sweep(0.0, 1.0, 301),
        "length": make_sweep(1.0, 30, 501),
    },
}
cfg = make_cfg(exp_cfg, ze.twotone.time_domain.ScanT1WithToneCfg, env, overrides={'reps': 100, 'rounds': 100})

t1_with_tone_sweep_exp = nb_adapter(ze.twotone.time_domain.ScanT1WithToneExp())
_ = t1_with_tone_sweep_exp.run(cfg)
```

```python
%matplotlib inline
t1_with_tone_sweep_analysis = t1_with_tone_sweep_exp.analyze(None)
fig = t1_with_tone_sweep_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_t1_with_tone_sweep_{time.strftime('%m%d')}"
savefig(t1_with_tone_sweep_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
t1_with_tone_sweep_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

## T2Echo

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_120",
        "pi_pulse": "pi_amp",
        "pi2_pulse": "pi2_amp",
        "readout": "readout_dpm",
    },
    # "relax_delay": 40.0,  # us
    "relax_delay": 5 * md.t1,  # us
    # "sweep": make_sweep(0.0, 5 * md.t2r, 51),
    "sweep": make_sweep(0.0, 1.5 * md.t2e, 101),
    # "sweep": make_sweep(0.01, 5.0, 101),
}
cfg = make_cfg(exp_cfg, ze.twotone.time_domain.T2EchoCfg, env, overrides={'reps': 1000, 'rounds': 100})

activate_detune = 0.1 / cfg.sweep.length.step

cfg = cfg.with_updates(detune=activate_detune)

t2echo_exp = nb_adapter(ze.twotone.time_domain.T2EchoExp())
t2echo_run = t2echo_exp.run(cfg)
true_detune = t2echo_run.result.true_activate_detune
if true_detune is None:
    raise ValueError("The run did not return its applied detuning")
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.time_domain.t2echo import T2EchoAnalyzeOptions

t2echo_analysis = t2echo_exp.analyze(T2EchoAnalyzeOptions(fit_method="fringe"))
md.t2e = t2echo_analysis.result.t2e
md.t2e_err = t2echo_analysis.result.t2e_err
detune = t2echo_analysis.result.detune
fig = t2echo_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_t2echo_{time.strftime('%m%d')}"
savefig(t2echo_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
t2echo_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"activate detune = {true_detune:.3f}MHz\nt2echo = {md.t2e:.3f}us",
    unique=True,
)
```

## CPMG

```python
gc_collect()
```

```python
%matplotlib widget
times = list(range(20, 0, -1))

exp_cfg = {
    "modules": {
        # "reset": "reset_120",
        "pi_pulse": "pi_len",
        "pi2_pulse": ml.get_module("pi2_len", {"phase": 90}),  # Y/2 gate
        "readout": "readout_dpm",
    },
    "sweep": {"times": times},
    "length_expts": 151,
    "length_range": [(0.1 * t, 5.0 * t) for t in times],
    # "relax_delay": 30.0,  # us
    "relax_delay": 5 * md.t1,  # us
}
detune_ratio = 0.1
cfg = make_cfg(exp_cfg, ze.twotone.time_domain.CPMG_Cfg, env, overrides={'reps': 1000, 'rounds': 100, 'detune_ratio': detune_ratio})

cpmg_exp = nb_adapter(ze.twotone.time_domain.CPMG_Exp())
_ = cpmg_exp.run(cfg)
```

```python
from zcu_tools.experiment.v2.twotone.time_domain.cpmg import CPMGAnalyzeOptions

cpmg_analysis = cpmg_exp.analyze(CPMGAnalyzeOptions(fit_fringe=True))
fig = cpmg_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_cpmg_{time.strftime('%m%d')}"
savefig(cpmg_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
cpmg_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

# Save Sample

```python
from datetime import datetime

validate_sample_table_v2(sample_table.samples, allow_empty=True)

flx_int = md.get("flx_int")
flx_period = md.get("flx_period")
frame_fields = {}
if (
    isinstance(flx_int, (int, float))
    and isinstance(flx_period, (int, float))
    and np.isfinite(flx_int)
    and np.isfinite(flx_period)
    and flx_period > 0
):
    frame_fields = {"flux_int": flx_int, "flux_period": flx_period}

sample_table.add_sample(
    **{
        "dev_value": cur_value,
        "dev_unit": "A",
        **frame_fields,
        "Freq (MHz)": md.q_f,
        "T1 (us)": md.t1,
        "T1err (us)": md.t1err,
        "T2r (us)": md.t2r,
        "T2r err (us)": md.t2r_err,
        "T2e (us)": md.t2e,
        "T2e err (us)": md.t2e_err,
        "Tcomment": "Manual Added",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
)
```

```python
device_cfg_path = os.path.join(em.flux_dir, "device_info.json")

dump_device_info(device_cfg_path, device_manager)
```

# Single shot

```python
jpa_sgs.get_info()
```

## Ground state & Excited state

先執行 GE 量測與 FIT，再執行 post analysis。Post analysis 使用前一格得到的校準結果，不需要重跑量測。

```python
from zcu_tools.experiment.v2.singleshot.ge import (
    GE_Exp,
    GEAnalyzeOptions,
    GEPostAnalyzeOptions,
)
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.notebook.experiments import GEPostAnalyzer

exp_cfg = {
    "modules": {
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
    },
    # "relax_delay": 70.5,  # us
    "relax_delay": 5 * md.t1,  # us
}
cfg = make_cfg(exp_cfg, ze.singleshot.GE_Cfg, env, overrides={'shots': 100000})
print("readout length: ", cfg.modules.readout.ro_cfg.ro_length)

ge_core = GE_Exp()
sh_ge_exp = nb_adapter(ge_core)
_ = sh_ge_exp.run(cfg)
ge_post_analyzer = GEPostAnalyzer(ge_core)
```

```python
ge_primary = sh_ge_exp.analyze(
    GEAnalyzeOptions(
        initial_state="ground",
        backend="center",
        # length_ratio=cfg.modules.readout.ro_cfg.ro_length / md.t1_with_tone,
        logscale=True,
        align_t1=True,
    ),
)
ge_analysis = ge_primary.result
md.fid = ge_analysis.fidelity
fig = ge_primary.figures["fit"]
print(f"Optimal fidelity after rotation = {md.fid:.1%}")
```

```python
filename = f"{qub_name}_sh_ge_{time.strftime('%H%M')}"
savefig(fig, os.path.join(em.flux_dir, "image", f"{filename}.png"))
ge_filepath = sh_ge_exp.save(
    Path(database_path) / f"{filename}@{em.label}",
    unique=True,
    comment=str(ge_analysis),
)
```

```python
from zcu_tools.simulate.temp import effective_temperature

n_g = ge_analysis.init_pops[0][0]  # n_gg
n_e = ge_analysis.init_pops[0][1]  # n_ge

n_g, n_e = (n_g, n_e) if n_g > n_e else (n_e, n_g)  # ensure n_g >= n_e
n_g, n_e = n_g / (n_g + n_e), n_e / (n_g + n_e)  # normalize

eff_T, err_T = effective_temperature(population=[(n_g, 0.0), (n_e, md.q_f)])
eff_T, err_T
```

```python
md.g_center = ge_analysis.g_center
md.e_center = ge_analysis.e_center
md.ge_s = ge_analysis.ge_s
md.g_center, md.e_center, md.ge_s
```

```python
ge_post_record = ge_post_analyzer.analyze(
    ge_primary,
    GEPostAnalyzeOptions(consider_other=False),
)
ge_post = ge_post_record.result
md.confusion_matrix = ge_post.confusion.matrix
md.ge_radius = ge_post.confusion.radius
post_fig = ge_post_record.figures["post"]
md.ge_radius / md.ge_s
```

## Confusion matrix

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_10",
        # "probe_pulse": "pi_amp",
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
    },
    "relax_delay": 70.5,  # us
}
cfg = make_cfg(exp_cfg, ze.singleshot.CheckCfg, env, overrides={'shots': 10000})

sh_exp = nb_adapter(ze.singleshot.CheckExp())
_ = sh_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.check import CheckAnalyzeOptions

sh_analysis = sh_exp.analyze(CheckAnalyzeOptions(md.g_center, md.e_center, md.ge_radius, max_point=10000))
fig = sh_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_sh_g_{time.strftime('%H%M')}"
savefig(sh_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
sh_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"g: {md.g_center:.3}, e: {md.e_center:.3}, radius: {md.ge_radius:.3}, ",
    unique=True,
)
```

## Length Rabi

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_120",
        "qub_pulse": {
            "type": "pulse",
            "waveform": "qub_flat",
            "ch": md.qub_1_4_ch,
            "nqz": 2,
            "freq": md.q_f,
            "gain": 1.0,
            # "gain": md.pi_gain,
            # "mixer_freq": md.q_f,
        },
        # "readout": "readout_rf",
        "readout": "readout_dpm",
    },
    "relax_delay": 50.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.03, 0.2, 51),
}
# Retain 1000 * 100 acquisitions as raw IQ shots for the joint fit.
cfg = make_cfg(exp_cfg, ze.singleshot.LenRabiCfg, env, overrides={'shots': 100000, 'reps': 100000, 'rounds': 1, 'g_center': md.g_center, 'e_center': md.e_center, 'radius': md.ge_radius})

sh_lenrabi_exp = nb_adapter(ze.singleshot.LenRabiExp())
_ = sh_lenrabi_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.len_rabi import LenRabiAnalyzeOptions as SSLenRabiAnalyzeOptions

sh_lenrabi_analysis = sh_lenrabi_exp.analyze(SSLenRabiAnalyzeOptions())
fig = sh_lenrabi_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_sh_rabi_length_{time.strftime('%H%M')}"
savefig(sh_lenrabi_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
sh_lenrabi_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=(
        f"g: {md.g_center:.3}, "
        f"e: {md.e_center:.3}, "
        f"radius: {md.ge_radius:.3}, "
        f"confusion:{md.confusion_matrix}"
    ),
    unique=True,
)
```

## T1

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "pi_pulse": "pi_amp",
        # "readout": "readout_rf",
        "readout": "readout_dpm",
    },
    "relax_delay": 50.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.01, 50.1, 101),
    # "sweep": make_sweep(0.01*t1, 5 * t1, 51),
}
cfg = make_cfg(exp_cfg, ze.singleshot.t1.T1Cfg, env, overrides={'reps': 1000, 'rounds': 10, 'g_center': md.g_center, 'e_center': md.e_center, 'radius': md.ge_radius, 'uniform': True})

sh_t1_exp = nb_adapter(ze.singleshot.t1.T1Exp())
_ = sh_t1_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.t1.t1 import T1AnalyzeOptions as SST1AnalyzeOptions

sh_t1_analysis = sh_t1_exp.analyze(SST1AnalyzeOptions(confusion_matrix=md.confusion_matrix, skip=1))
fig = sh_t1_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_sh_t1_{time.strftime('%H%M')}"
savefig(sh_t1_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
sh_t1_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=(
        f"g: {md.g_center:.3}, "
        f"e: {md.e_center:.3}, "
        f"radius: {md.ge_radius:.3}, "
        f"confusion:{md.confusion_matrix}"
    ),
    unique=True,
)
```

### T1 with Tone

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_10",
        "pi_pulse": "pi_amp",
        "probe_pulse": ml.get_module(
            "readout_dpm", type=zp.PulseReadoutCfg
        ).pulse_cfg.with_updates(post_delay=5.0 / (2 * np.pi * md.rf_w)),
        "readout": "readout_dpm",
    },
    "relax_delay": 50.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": make_sweep(0.03, 20, 101),
    # "sweep": make_sweep(0.01*t1, 5 * t1, 51),
}
cfg = make_cfg(exp_cfg, ze.singleshot.t1.T1WithToneCfg, env, overrides={'reps': 1000, 'rounds': 10, 'g_center': md.g_center, 'e_center': md.e_center, 'radius': md.ge_radius, 'uniform': True})

sh_t1_with_tone_exp = nb_adapter(ze.singleshot.t1.T1WithToneExp())
_ = sh_t1_with_tone_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.t1.t1_with_tone import T1WithToneAnalyzeOptions as SST1WithToneAnalyzeOptions

sh_t1_with_tone_analysis = sh_t1_with_tone_exp.analyze(SST1WithToneAnalyzeOptions(confusion_matrix=md.confusion_matrix, skip=2))
t1 = sh_t1_with_tone_analysis.result.t1
t1_b = sh_t1_with_tone_analysis.result.t1_b
fig = sh_t1_with_tone_analysis.figures["fit"]
```

```python
md.t1_with_tone = t1
```

```python
filename = f"{qub_name}_sh_t1_with_tone_gain{cfg.modules.probe_pulse.gain:.3f}_{time.strftime('%H%M')}"
savefig(sh_t1_with_tone_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
sh_t1_with_tone_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=(
        f"g: {md.g_center:.3}, "
        f"e: {md.e_center:.3}, "
        f"radius: {md.ge_radius:.3}, "
        f"confusion:{md.confusion_matrix}"
    ),
    unique=True,
)
```

### T1 with sweep tone

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_10",
        "pi_pulse": "pi_amp",
        "probe_pulse": {
            "type": "pulse",
            "waveform": "mist_waveform",
            "ch": md.res_ch,
            "nqz": 2,
            "freq": md.readout_f,
            "post_delay": 10 / (2 * np.pi * md.rf_w),
            "gain": 0.0,  # not used
        },
        "readout": "readout_dpm",
    },
    "relax_delay": 40.5,  # us
    # "relax_delay": 5 * t1,  # us
    "sweep": {
        "gain": np.sqrt(np.linspace(0.0**2, 0.22**2, 201)),
        "length": make_sweep(0.01, 15, 501),
    },
}
cfg = make_cfg(exp_cfg, ze.singleshot.t1.T1WithToneSweepCfg, env, overrides={'reps': 1000, 'rounds': 1, 'g_center': md.g_center, 'e_center': md.e_center, 'radius': md.ge_radius})

sh_t1_with_tone_sweep_exp = nb_adapter(ze.singleshot.t1.T1WithToneSweepExp())
_ = sh_t1_with_tone_sweep_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.t1.t1_with_tone_sweep import T1WithToneSweepAnalyzeOptions

sh_t1_with_tone_sweep_analysis = sh_t1_with_tone_sweep_exp.analyze(T1WithToneSweepAnalyzeOptions(ac_coeff=md.ac_stark_coeff, confusion_matrix=md.confusion_matrix))
fig = sh_t1_with_tone_sweep_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_sh_t1_with_tone_sweep_{time.strftime('%H%M')}"
savefig(sh_t1_with_tone_sweep_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
sh_t1_with_tone_sweep_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=(
        f"g: {md.g_center:.3}, "
        f"e: {md.e_center:.3}, "
        f"radius: {md.ge_radius:.3}, "
        f"ac_stark_coeff: {md.ac_stark_coeff:.3}, "
        f"confusion:{md.confusion_matrix}"
    ),
    unique=True,
)
```

## MIST

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_10",
        # "init_pulse": "pi_amp",
        "probe_pulse": {
            "type": "pulse",
            "waveform": ml.get_waveform(
                "mist_waveform",
                {
                    # "length": 5.0 / (2 * np.pi * md.rf_w) + 50.0,
                    "length": 5.0 / (2 * np.pi * md.rf_w) + 0.3,
                },
            ),
            "ch": md.res_ch,
            "nqz": 2,
            "freq": md.readout_f,
            "post_delay": 10 / (2 * np.pi * md.rf_w),
            "gain": 0.0,  # not used
        },
        "readout": "readout_dpm",
    },
    "sweep": {
        "gain": make_sweep(0.0, 0.2, 301),
    },
    "relax_delay": 20.5,  # us
}
cfg = make_cfg(exp_cfg, ze.singleshot.mist.PowerCfg, env, overrides={'reps': 1000, 'rounds': 100, 'g_center': md.g_center, 'e_center': md.e_center, 'radius': md.ge_radius})

sh_mist_exp = nb_adapter(ze.singleshot.mist.PowerExp())
_ = sh_mist_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.mist.power import PowerAnalyzeOptions as SSMistPowerAnalyzeOptions

sh_mist_analysis = sh_mist_exp.analyze(
    SSMistPowerAnalyzeOptions(ac_coeff=md.ac_stark_coeff, confusion_matrix=md.confusion_matrix),
)
fig = sh_mist_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_sh_mist_g_short_{time.strftime('%H%M')}"
# filename = f"{qub_name}_sh_mist_steady_{time.strftime('%H%M')}"
savefig(sh_mist_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
sh_mist_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=(
        f"g: {md.g_center:.3}, "
        f"e: {md.e_center:.3}, "
        f"radius: {md.ge_radius:.3}, "
        f"ac_stark_coeff: {md.ac_stark_coeff:.3}, "
        f"confusion:{md.confusion_matrix}"
    ),
    unique=True,
)
```

## MIST Shots

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_10",
        "init_pulse": "pi_amp",
        "probe_pulse": {
            "type": "pulse",
            "waveform": ml.get_waveform(
                "mist_waveform",
                {
                    # "length": 5.0 / (2 * np.pi * md.rf_w) + 50.0,
                    "length": 5.0 / (2 * np.pi * md.rf_w) + 0.3,
                },
            ),
            "ch": md.res_ch,
            "nqz": 2,
            "freq": md.readout_f,
            "gain": 0.132,
            "post_delay": 10 / (2 * np.pi * md.rf_w),
        },
        "readout": "readout_dpm",
    },
    "relax_delay": 50.5,  # us
}
cfg = make_cfg(exp_cfg, ze.singleshot.CheckCfg, env, overrides={'shots': 1000000})

sh_mist_check_exp = nb_adapter(ze.singleshot.CheckExp())
_ = sh_mist_check_exp.run(cfg)
```

```python
%matplotlib inline
sh_mist_check_analysis = sh_mist_check_exp.analyze(
    CheckAnalyzeOptions(md.g_center, md.e_center, md.ge_radius),
)
fig = sh_mist_check_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_sh_mist_steady_gain{cfg.modules.probe_pulse.gain:.4f}_{time.strftime('%H%M')}"
# filename = f"{qub_name}_sh_e_mist_short_gain{cfg.modules.probe_pulse.gain:.4f}_{time.strftime('%H%M')}"
savefig(sh_mist_check_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
sh_mist_check_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"g: {md.g_center:.3}, e: {md.e_center:.3}, radius: {md.ge_radius:.3}, ",
    unique=True,
)
```

## AC Stark shift

```python
%matplotlib widget
probe_len = 1 * ml.get_module("pi_amp", type=zp.PulseCfg).waveform.length
exp_cfg = {
    "modules": {
        # "reset": "reset_bath_e",
        # "init_pulse": "pi_amp",
        "stark_pulse1": {
            "type": "pulse",
            "waveform": ml.get_waveform(
                "mist_waveform", {"length": 5.1 / (2 * np.pi * md.rf_w) + probe_len}
            ),
            "ch": md.res_ch,
            "nqz": 2,
            "freq": md.readout_f,
            # "freq": md.r_f,
            "gain": 0.0,  # not used
        },
        "stark_pulse2": ml.get_module(
            "pi_amp",
            {
                "waveform": {"length": probe_len},
                # "gain": 0.1,
                "pre_delay": 5.0 / (2 * np.pi * md.rf_w),
                "post_delay": 3.1 / (2 * np.pi * md.rf_w),
            },
        ),
        # "readout": "readout_rf",
        "readout": "readout_dpm",
    },
    "sweep": {
        "gain": make_sweep(0.0, 0.22, 301),
        # "gain": make_sweep(0.0, 0.05, 51),
        "freq": make_sweep(md.q_f - 700.0, md.q_f + 100.0, step=0.1),
    },
    "relax_delay": 5.5,  # us
}
cfg = make_cfg(exp_cfg, ze.singleshot.AcStarkCfg, env, overrides={'reps': 1000, 'rounds': 2, 'g_center': md.g_center, 'e_center': md.e_center, 'radius': md.ge_radius})

sh_ac_stark_exp = nb_adapter(ze.singleshot.AcStarkExp())
_ = sh_ac_stark_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.ac_stark import AcStarkAnalyzeOptions as SSAcStarkAnalyzeOptions

sh_ac_stark_analysis = sh_ac_stark_exp.analyze(SSAcStarkAnalyzeOptions(chi=md.chi, kappa=md.rf_w, confusion_matrix=md.confusion_matrix, cutoff=0.05))
ac_stark_coeff = sh_ac_stark_analysis.result.ac_stark_coeff
fig = sh_ac_stark_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_sh_ac_stark_rf{cfg.modules.stark_pulse1.freq:.1f}MHz_{time.strftime('%H%M')}"
savefig(sh_ac_stark_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
sh_ac_stark_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=(
        f"g: {md.g_center:.3}, "
        f"e: {md.e_center:.3}, "
        f"radius: {md.ge_radius:.3}, "
        f"confusion:{md.confusion_matrix}"
    ),
    unique=True,
)
```

```python
md.ac_stark_coeff = ac_stark_coeff
```

# MIST

```python
mist_pulse_len = 1.0  # us
ml.register_waveform(
    mist_waveform={
        "style": "const",
        "length": mist_pulse_len,  # us
    },
)
```

## Single Trace

```python
cur_value = flux_yoko.set_current(-7e-3)
cur_value * 1e3
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "init_pulse": "pi_amp",
        "probe_pulse": {
            "type": "pulse",
            "waveform": ml.get_waveform(
                "mist_waveform",
                {
                    # "length": 5.0 / (2 * np.pi * md.rf_w) + 30.0,
                    "length": 5.0 / (2 * np.pi * md.rf_w) + 0.3,
                },
            ),
            "ch": md.res_ch,
            "nqz": 2,
            # "freq": r_f,
            # "post_delay": 10 / (2 * np.pi * rf_w),
            # "gain": 0.3,
            "freq": md.r_f,
            "post_delay": 3 / (2 * np.pi * md.rf_w),
            "gain": 0.0,  # not used
        },
        "readout": "readout_dpm",
    },
    "sweep": {
        "gain": make_sweep(0.0, 0.5, 301),
    },
    "relax_delay": 50.0,  # us
}
cfg = make_cfg(exp_cfg, ze.mist.PowerDepCfg, env, overrides={'reps': 100, 'rounds': 100})

mist_exp = nb_adapter(ze.mist.PowerDepExp())
_ = mist_exp.run(cfg)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.mist.power_dep.single_trace import PowerDepAnalyzeOptions

mist_analysis = mist_exp.analyze(PowerDepAnalyzeOptions(ac_coeff=md.ac_stark_coeff))
fig = mist_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_mist_e_{time.strftime('%H%M')}"
savefig(mist_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
mist_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    comment=f"ac_stark_coeff: {md.ac_stark_coeff:.3}",
    unique=True,
)
```

# FastFlux

```python

```

## Twotone

```python
%matplotlib widget
lf_qub_len = 0.2  # us
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "flux_pulse": {
            "type": "pulse",
            "waveform": {"style": "const", "length": 0.01 + lf_qub_len},
            "ch": md.lo_flux_ch,
            "nqz": 1,
            "freq": 0.0,  # not used
            "mixer_freq": 0.0,  # not used
            "gain": 0.0,  # not used
        },
        "qub_pulse": {
            "type": "pulse",
            "waveform": ml.get_waveform("qub_flat", {"length": lf_qub_len}),
            "ch": md.qub_1_4_ch,
            "nqz": 2,
            "gain": 0.3,
            "pre_delay": 0.005,
            "post_delay": 0.006,
            "freq": 0.0,  # not used
        },
        # "readout": "readout_rf",
        "readout": "readout_dpm",
    },
    "sweep": {
        "gain": make_sweep(-1.0, 1.0, 101),
        "freq": make_sweep(md.q_f - 70.0, md.q_f + 70.0, step=0.2),
        # "freq": make_sweep(5650, 5750, step=0.5),
    },
    "relax_delay": 0.1,  # us
}
cfg = make_cfg(exp_cfg, ze.fastflux.TwotoneCfg, env, overrides={'reps': 100, 'rounds': 1000})

lf_twotone_exp = nb_adapter(ze.fastflux.TwoToneExp())
_ = lf_twotone_exp.run(cfg)
```

```python
%matplotlib inline
lf_twotone_analysis = lf_twotone_exp.analyze(None)
fig = lf_twotone_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_fastflux_twotone_{time.strftime('%H%M')}"
savefig(lf_twotone_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
lf_twotone_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

## Distortion

```python

```

### Acc Phase

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "flux_pulse": {
            "type": "pulse",
            "waveform": {"style": "const", "length": 0.3},
            "ch": md.lo_flux_ch,
            "nqz": 1,
            "gain": 0.3,
            "pre_delay": 0.4,
            "freq": 0.0,  # not used
            "mixer_freq": 0.0,  # not used
        },
        "pi2_pulse": "pi2_len",
        # "readout": "readout_rf",
        "readout": "readout_dpm",
    },
    "sweep": {
        "length": make_sweep(0.05, 1.0, 351),
        "phase": make_sweep(-360, 360, 51),
    },
    "readout_t": 1.05,
    "relax_delay": 10.1,  # us
}
cfg = make_cfg(exp_cfg, ze.fastflux.distortion.AccPhaseCfg, env, overrides={'reps': 100, 'rounds': 500})

lf_dt_ap_exp = nb_adapter(ze.fastflux.distortion.AccPhaseExp())
_ = lf_dt_ap_exp.run(cfg)
```

```python
%matplotlib inline
lf_dt_ap_analysis = lf_dt_ap_exp.analyze(None)
fig = lf_dt_ap_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_flux_distortion_accphase_{time.strftime('%H%M')}"
savefig(lf_dt_ap_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
lf_dt_ap_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

### Phase

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "flux_pulse": {
            "type": "pulse",
            "waveform": {"style": "const", "length": 0.3},
            "ch": md.lo_flux_ch,
            "nqz": 1,
            "gain": 0.25,
            "pre_delay": 0.2,
            "freq": 0.0,  # not used
            "mixer_freq": 0.0,  # not used
        },
        "pi2_pulse": "pi2_len",
        "readout": "readout_dpm",
    },
    "sweep": {
        "length": make_sweep(0.0, 0.9, 151),
        "phase": make_sweep(-360, 360, 51),
    },
    "readout_t": 0.95,
    "relax_delay": 0.1,  # us
}
cfg = make_cfg(exp_cfg, ze.fastflux.distortion.PhaseCfg, env, overrides={'reps': 1000, 'rounds': 1000})

lf_dt_p_exp = nb_adapter(ze.fastflux.distortion.PhaseExp())
_ = lf_dt_p_exp.run(cfg)
```

```python
%matplotlib inline
lf_dt_p_analysis = lf_dt_p_exp.analyze(None)
fig = lf_dt_p_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_flux_distortion_phase_{time.strftime('%H%M')}"
savefig(lf_dt_p_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
lf_dt_p_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

### Freq

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "flux_pulse": {
            "type": "pulse",
            "waveform": {"style": "const", "length": 1.0},
            "ch": md.lo_flux_ch,
            "nqz": 1,
            "gain": 0.3,
            "pre_delay": 0.1,
            "freq": 0.0,  # not used
            "mixer_freq": 0.0,  # not used
        },
        "qub_pulse": {
            "type": "pulse",
            "waveform": {"style": "const", "length": 0.05},
            "ch": md.qub_1_4_ch,
            "nqz": 2,
            "gain": 0.3,
            # "mixer_freq": md.q_f,
            "freq": 0.0,  # not used
        },
        "readout": "readout_dpm",
    },
    "sweep": {
        "length": make_sweep(0.05, 1.2, 41),
        "freq": make_sweep(md.q_f - 80, md.q_f + 20, step=1.0),
    },
    "readout_t": 1.25,
    "relax_delay": 0.1,  # us
}
cfg = make_cfg(exp_cfg, ze.fastflux.distortion.FreqCfg, env, overrides={'reps': 100, 'rounds': 1000})

lf_dt_freq_exp = nb_adapter(ze.fastflux.distortion.FreqExp())
_ = lf_dt_freq_exp.run(cfg)
```

```python
%matplotlib inline
lf_dt_freq_analysis = lf_dt_freq_exp.analyze(None)
fig = lf_dt_freq_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_flux_distortion_freq_{time.strftime('%H%M')}"
savefig(lf_dt_freq_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
lf_dt_freq_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

## T1

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        # "reset": "reset_bath",
        "flux_pulse": {
            "type": "pulse",
            "waveform": {"style": "const", "length": 0.0},  # not used
            "ch": md.lo_flux_ch,
            "nqz": 1,
            "pre_delay": 0.05,
            "post_delay": 0.05,
            "freq": 0.0,  # not used
            "mixer_freq": 0.0,  # not used
            "gain": 0.0,  # not used
        },
        "pi_pulse": ml.get_module("pi_amp"),
        # "readout": "readout_rf",
        "readout": "readout_dpm",
    },
    "sweep": {
        "gain": make_sweep(-1.0, 1.0, 101),
        "length": make_sweep(0.05, 5.0, 101),
    },
    "relax_delay": 10.1,  # us
}
cfg = make_cfg(exp_cfg, ze.fastflux.T1Cfg, env, overrides={'reps': 100, 'rounds': 1000})

lf_t1_exp = nb_adapter(ze.fastflux.T1Exp())
_ = lf_t1_exp.run(cfg)
```

```python
%matplotlib inline
lf_t1_analysis = lf_t1_exp.analyze(None)
fig = lf_t1_analysis.figures["fit"]
```

```python
filename = f"{qub_name}_fastflux_t1_{time.strftime('%H%M')}"
savefig(lf_t1_analysis.figures["fit"], os.path.join(em.flux_dir, "image", f"{filename}.png"))
lf_t1_exp.save(
    Path(
        os.path.join(database_path, f"{filename}@{em.label}")
    ),
    unique=True,
)
```

# Disconnect

```python
if resource_manager is not None:
    device_manager.close_all_devices()
    resource_manager.close()
    resource_manager = None
```
