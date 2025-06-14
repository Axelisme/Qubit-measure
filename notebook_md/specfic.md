---
jupyter:
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

```python
%load_ext autoreload
import os

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

%autoreload 2
import zcu_tools.notebook.single_qubit as zf  # noqa: E402
import zcu_tools.notebook.schedule.v2 as zs  # noqa: E402

# ruff: noqa: I001
from zcu_tools import (  # noqa: E402
    DefaultCfg,
    create_datafolder,
    make_cfg,
    make_sweep,
    save_data,
    make_comment,
)
```

```python
import zcu_tools.config as zc

# zc.config.DATA_DRY_RUN = True
# zc.config.YOKO_DRY_RUN = True
```

# Create data folder

```python
chip_name = r"Q12_2D[2]\Q4"

# data_host = "192.168.10.232"  # cmd-> ipconfig -> ipv4 #controling computer
data_host = None

database_path = create_datafolder(os.path.join(os.getcwd(), ".."), prefix=chip_name)
```

# Connect to zcu216
```python
from zcu_tools.remote import make_proxy
from zcu_tools.program.base import MyProgram  # noqa: F401
from zcu_tools.tools import get_ip_address  # noqa: F401

# zc.config.LOCAL_IP = get_ip_address("tailscale0")
zc.config.LOCAL_IP = "192.168.10.232"
zc.config.LOCAL_PORT = 8887

soc, soccfg, rm_prog = make_proxy("192.168.10.113", 8887, proxy_prog=True)
MyProgram.init_proxy(rm_prog, test=True)
print(soccfg)
```


```python
# from qick import QickSoc  # noqa: E402

# soc = QickSoc()
# soccfg = soc
# print(soc)
```

# Predefine parameters

```python
res_ch = 0
qub_ch = 11
reset_ch = 5
reset_ch2 = 2

DefaultCfg.set_adc(ro_chs=[0])
# DefaultCfg.set_dev(flux_dev="none", flux=0.0)
```

```python
DefaultCfg.load("Q12_2D[2]-Q4_default_cfg_-0.42mA_0613.yaml")
```

## Initialize the flux

```python
from zcu_tools.device import YokoDevControl  # noqa: E402

YokoDevControl.connect_server(
    {
        "host_ip": data_host,
        # "host_ip": "127.0.0.1",
        "dComCfg": {"address": "0x0B21::0x0039::91T810992", "interface": "USB"},
        "outputCfg": {"Current - Sweep rate": 10e-6},
    },
    reinit=True,
)
DefaultCfg.set_dev(flux_dev="yoko")
cur_A = YokoDevControl.get_current()
cur_A
```

```python
# cur_A = 0.0e-3
YokoDevControl.set_current(cur_A)
DefaultCfg.set_dev(flux=cur_A)
```


# MIST

```python
qub_name = "Q4"
```

## Power depedence

```python
# DefaultCfg.set_pulse(readout_rf = {**DefaultCfg.get_pulse("readout_rf"), "gain": 0.1})
```

```python
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        # "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse("readout_rf"),
            "post_delay": 0.5,
            "freq": 5796,
        },
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
        "reset": "two_pulse",
        "reset_pulse1": "mux_reset1",
        "reset_pulse2": "mux_reset2",
        "reset_pi_pulse": "pi_amp",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}
```

```python
exp_cfg["sweep"] =  make_sweep(0.0, 1.0, 101)
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

# zs.mist.visualize_mist_pdr_dep(soccfg, cfg, time_fly=0.6)
pdrs, signals = zs.mist.measure_mist_pdr_dep(soc, soccfg, cfg)
```

```python
filename = f"{qub_name}_mist_pdr@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg),
    tag="MIST/pdr/single",
    server_ip=data_host,
)
```

### Abnormal

```python
exp_cfg = {
    "dac": {
        "res_pulse": {
            **DefaultCfg.get_pulse("readout_rf"),
            "freq": 5796,
        },
        # "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse("readout_rf"),
            # **DefaultCfg.get_pulse("pi_amp"),
            # "length": 0.1,
            "post_delay": 0.5,
        },

        # "reset": "pulse",
        # "reset_pulse": "reset_red",
        "reset": "two_pulse",
        "reset_pulse1": {
            **DefaultCfg.get_pulse("mux_reset1"),
            # "gain": 0.0,
        },
        "reset_pulse2": "mux_reset2",
        # "reset_pi_pulse": "pi_amp",

        "readout": "two_pulse",
        "pre_res_pulse": "pi_amp",
    },
    "adc": {
        "relax_delay": 1.0,  # us
    },
}
```

```python
exp_cfg["sweep"] =  make_sweep(0.0, 1.0, 101)
cfg = make_cfg(exp_cfg, reps=1000, rounds=100)

# zs.mist.visualize_abnormal_pdr_dep(soccfg, cfg, time_fly=0.6)
pdrs, signals = zs.mist.measure_abnormal_pdr_dep(soc, soccfg, cfg)
```

```python
zf.mist.analyze_abnormal_pdr_dep(pdrs, signals)
```

```python
filename = f"{qub_name}_abnormal_pdr@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
    y_info={"name": "GE", "unit": "None", "values": np.array([0,1])},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg),
    tag="MIST/pdr/abnormal",
    server_ip=data_host,
)
```

### Overnight

```python
from IPython.display import display, clear_output
import time

exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        # "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse("readout_rf"),
            "post_delay": 0.5,
        },
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
        "reset": "two_pulse",
        "reset_pulse1": "mux_reset1",
        "reset_pulse2": "mux_reset2",
        "reset_pi_pulse": "pi_amp",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}
exp_cfg["sweep"] =  make_sweep(0.0, 1.0, 51)
cfg = make_cfg(exp_cfg, reps=500, rounds=10)

# Create two subplots: one for current scan, one for historical scans
%matplotlib inline
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
plt.tight_layout()

dh = display(fig, display_id=True)

total_time = 1 * 60 * 60  # 1 hours in seconds
interval = 5 * 60  # 5 minutes in seconds

overnight_signals = []
try:
    for i in tqdm(range(total_time // interval), desc="Overnight Scans", unit="iteration"):
        start_t = time.time()

        pdrs, signals = zs.measure_mist_pdr_dep(soc, soccfg, cfg, backend_mode=True)
        overnight_signals.append(signals)

        signals_array = np.array(overnight_signals)
        g0 = np.mean(signals_array[:, 0])

        # Left plot: Current scan
        ax_left.clear()
        ax_left.plot(pdrs, np.abs(signals - g0), linestyle="-", marker=".")
        ax_left.set_xlabel("Drive Power (a.u.)")
        ax_left.set_ylabel("Signal (a.u.)")
        ax_left.set_title(f"Current Scan (Iteration {i+1})")

        # Right plot: Historical scans
        ax_right.clear()
        ax_right.plot(pdrs, np.abs(signals_array- g0).T, linestyle="--")
        ax_right.set_xlabel("Drive Power (a.u.)")
        ax_right.set_ylabel("Signal (a.u.)")
        ax_right.set_title("Historical Scans")

        dh.update(fig)

        while time.time() - start_t < interval:
            plt.pause(0.5)  # Pause to allow the plot to update
    plt.close(fig)
    clear_output(wait=True)


    overnight_signals = np.array(overnight_signals)

    g0 = np.mean(overnight_signals[:, 0])

    # plot overnight_signals in one plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pdrs, np.abs(overnight_signals-g0).T, linestyle="--")
    ax.set_xlabel(r"$\bar n$")
    ax.set_ylabel("Signal difference")
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    print("Overnight scans interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")


filename = f"{qub_name}_mist_pdr_overnight@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
    y_info={"name": "Iteration", "unit": "None", "values": np.arange(len(overnight_signals))},
    z_info={"name": "Signal", "unit": "a.u.", "values": np.array(overnight_signals)},
    comment=make_comment(cfg),
    tag="MIST/pdr/overnight",
    server_ip=data_host,
)
```

### Two Pulse Reset

```python
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        # "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse("readout_rf"),
            "post_delay": 0.5,
        },
        "reset_test_pulse1": {
            **DefaultCfg.get_pulse("mux_reset1"),
            # "length": 10.0,
            # "gain": 0.0,
        },
        "reset_test_pulse2": {
            **DefaultCfg.get_pulse("mux_reset2"),
            # "length": 10.0,
            # "gain": 0.0
        },
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
        "reset": "two_pulse",
        "reset_pulse1": "mux_reset1",
        "reset_pulse2": "mux_reset2",
        # "reset_pi_pulse": "pi_amp",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}
```

```python
exp_cfg["sweep"] =  make_sweep(0.0, 1.0, 101)
cfg = make_cfg(exp_cfg, reps=1000, rounds=10)

# zs.visualize_mist_pdr_mux_reset(soccfg, cfg, time_fly=0.6)
pdrs, signals = zs.measure_mist_pdr_mux_reset(soc, soccfg, cfg)
```

```python
filename = f"{qub_name}_mist_pdr_mux_reset@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
    y_info={"name": "W/O Reset", "unit": "None", "values": np.array([0, 1])},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg),
    tag="MIST/pdr_reset",
    server_ip=data_host,
)
```

#### Abnormal

```python
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        # "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse("readout_rf"),
            "post_delay": 0.5,
        },
        "reset_test_pulse1": {
            **DefaultCfg.get_pulse("mux_reset1"),
            # "length": 5.0,
            "gain": 0.0,
        },
        "reset_test_pulse2": {
            **DefaultCfg.get_pulse("mux_reset2"),
            # "length": 5.0,
            # "gain": 0.0
        },

        # "reset": "pulse",
        # "reset_pulse": "reset_red",
        "reset": "two_pulse",
        "reset_pulse1": "mux_reset1",
        "reset_pulse2": "mux_reset2",
        # "reset_pi_pulse": "pi_amp",

        "readout": "two_pulse",
        "pre_res_pulse": "pi_amp",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}
```

```python
exp_cfg["sweep"] =  make_sweep(0.0, 1.0, 61)
cfg = make_cfg(exp_cfg, reps=1000, rounds=100)

zs.mist.visualize_abnormal_pdr_mux_reset(soccfg, cfg, time_fly=0.6)
# pdrs, signals = zs.mist.measure_abnormal_pdr_mux_reset(soc, soccfg, cfg)
```

```python
zf.mist.analyze_abnormal_pdr_dep(pdrs, signals)
```

```python
filename = f"{qub_name}_abnormal_pdr_mux_reset@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
    y_info={"name": "GE", "unit": "None", "values": np.array([0,1])},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg),
    tag="MIST/pdr_reset_abnormal",
    server_ip=data_host,
)
```

## Flux Power depedence

```python
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        "qub_pulse": {
            **DefaultCfg.get_pulse("readout_rf"),
            "post_delay": 0.5,
        },
    },
    "adc": {
        "relax_delay": 5.0,  # us
    },
}
```

```python
exp_cfg["sweep"] = {
    "flux": make_sweep(-0.25e-3, -0.55e-3, 101),
    "gain": make_sweep(0.0, 1.0, 51),
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=100)

flxs, pdrs, signals2D = zs.measure_mist_flx_pdr_dep2D(soc, soccfg, cfg)
```

```python
filename = f"{qub_name}_mist_flx_pdr@{cur_A * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Flux", "unit": "A", "values": flxs},
    y_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
    comment=make_comment(cfg),
    tag="MIST/flx_pdr",
    server_ip=data_host,
)
```

# Lookback2D

```python
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": "const",
            # "style": "cosine",
            # "style": "gauss",
            # "sigma": 9.5/4,  # us
            # "style": "flat_top",
            # "raise_pulse": {"style": "gauss", "length": 5.0, "sigma": 0.2},
            # "raise_pulse": {"style": "cosine", "length": 3.0},
            "freq": 6028,  # MHz
            "gain": 30000,
            "length": 1.0,  # us
        },
    },
    "adc": {
        "ro_length": 2.0,  # us
        "trig_offset": 0.48,  # us
    },
    "relax_delay": 0.0,  # us
}
```


```python
cfg = make_cfg(exp_cfg, rounds=1000)

Ts, signals = zs.measure_lookback(soc, soccfg, cfg)
```

```python
_ = zf.lookback_show(
    Ts,
    signals,
    ratio=0.15,
    plot_fit=True,
    smooth=1.0,
    pulse_cfg=cfg["dac"]["res_pulse"],
)
```


```python
cfg = make_cfg(exp_cfg, rounds=1000)

freqs = np.linspace(6020, 6030, 501)
signals2D = []
for f in tqdm(freqs):
    cfg["dac"]["res_pulse"]["freq"] = f
    Ts, signals = zs.measure_lookback(soc, soccfg, cfg, progress=False)
    signals2D.append(signals)
signals2D = np.array(signals2D)
```

```python
cfg = make_cfg(exp_cfg, rounds=10000)
cfg["dac"]["res_pulse"]["freq"] = 6020

pdrs = np.arange(3000, 30000, 1000)
signals = []
for p in tqdm(pdrs):
    cfg["dac"]["res_pulse"]["gain"] = p.item()
    Ts, Is, Qs = zs.measure_lookback(soc, soccfg, cfg, progress=False)
    signals.append(Is + 1j * Qs)
signals = np.array(signals)
```

```python
filename = "lookback2D"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    y_info={"name": "freq", "unit": "", "values": freqs},
    comment=make_comment(cfg),
    tag="Lookback",
    server_ip=data_host,
)
```

# Circle fit

```python
res_name = "Resonator_test"
```

```python
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": "flat_top",
            "raise_pulse": {"style": "gauss", "length": 0.6, "sigma": 0.1},
            "gain": 300,
            "nqz": 2,
            "length": 5.0,  # us
            "trig_offset": 2.5,
            "ro_length": 2.5,
        },
    },
    "relax_delay": 0.0,  # us
}
```

```python
exp_cfg["sweep"] = make_sweep(5900, 6100, 101)
cfg = make_cfg(exp_cfg, reps=100, rounds=10)

fpts, signals = zs.measure_res_freq(soc, soccfg, cfg)
```

```python
num1, num2 = 5, 5
slope1, _ = zf.phase_analyze(fpts[:num1], signals[:num1])
slope2, _ = zf.phase_analyze(fpts[-num2:], signals[-num2:])
slope = (slope1 + slope2) / 2
```

```python
c_signals = zf.rotate_phase(fpts, signals, -slope)
plt.plot(c_signals.real, c_signals.imag, marker="o")
```

```python
r_f = zf.freq_analyze(fpts, c_signals, asym=True, max_contrast=True)
r_f
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
