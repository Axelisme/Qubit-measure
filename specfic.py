# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
import os
import sys

# # %cd /home/xilinx/jupyter_notebooks/nthu/sinica-5q/Axel/Qubit-measure
print(os.getcwd())
sys.path.append(os.getcwd())

import numpy as np
from tqdm.auto import tqdm
from pprint import pprint  # noqa: F401
import matplotlib.pyplot as plt

# %autoreload 2
import zcu_tools.analysis as zf  # noqa: E402
import zcu_tools.schedule.v2 as zs  # noqa: E402

# ruff: noqa: I001
from zcu_tools import (  # noqa: E402
    DefaultCfg,
    create_datafolder,
    make_cfg,
    make_sweep,
    save_data,
    make_comment,
)

# %%
import zcu_tools.config as zc

zc.config.DATA_DRY_RUN = True
zc.config.YOKO_DRY_RUN = True

# %% [markdown]
# # Connect to zcu216
# %%
# from zcu_tools.remote import make_proxy
# from zcu_tools.program.base import MyProgram  # noqa: E402
# from zcu_tools.tools import get_ip_address

# ns_host = "pynq"
# ns_port = 8887
# zc.config.LOCAL_IP = get_ip_address("tailscale0")
# zc.config.LOCAL_PORT = 8887

# soc, soccfg, rm_prog = make_proxy(ns_host, ns_port)
# MyProgram.init_proxy(rm_prog, test=True)
# print(soccfg)


# %%
from qick import QickSoc  # noqa: E402

soc = QickSoc()
soccfg = soc
print(soc)

# %% [markdown]
# # Create data folder

# %%
database_path = create_datafolder(os.getcwd(), prefix="")

# data_host = "192.168.10.232"  # cmd-> ipconfig -> ipv4 #controling computer
# data_host = "100.76.229.37"  # tailscale
data_host = None

# %% [markdown]
# # Predefine parameters

# %%
DefaultCfg.set_dac(res_ch=0, qub_ch=1)
DefaultCfg.set_adc(ro_chs=[0])
DefaultCfg.set_dev(flux_dev="none", flux=0.0)

# %%
# DefaultCfg.load("cfg.yaml")

# %% [markdown]
# ## Initialize the flux

# %%
from zcu_tools.device import YokoDevControl  # noqa: E402

YokoDevControl.connect_server(
    {
        "host_ip": data_host,
        # "host_ip": "127.0.0.1",
        "dComCfg": {"address": "0x0B21::0x0039::90ZB35281", "interface": "USB"},
        "outputCfg": {"Current - Sweep rate": 10e-6},
    },
    reinit=True,
)
DefaultCfg.set_dev(flux_dev="yoko")

# %%
cur_flux = 6.0292e-3
YokoDevControl.set_current(cur_flux)
DefaultCfg.set_default(flux=cur_flux)


# %% [markdown]
# # Lookback2D

# %%
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


# %%
cfg = make_cfg(exp_cfg, rounds=1000)

Ts, signals = zs.measure_lookback(soc, soccfg, cfg)

# %%
predict_offset = zf.lookback_show(Ts, signals, ratio=0.5)


# %%
cfg = make_cfg(exp_cfg, rounds=1000)

freqs = np.linspace(6020, 6030, 501)
signals = []
for f in tqdm(freqs):
    cfg["dac"]["res_pulse"]["freq"] = f
    Ts, Is, Qs = zs.measure_lookback(soc, soccfg, cfg, progress=False)
    signals.append(Is + 1j * Qs)
signals = np.array(signals)

# %%
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

# %%
cfg = make_cfg(exp_cfg, rounds=10000)
cfg["dac"]["res_pulse"]["freq"] = 6020

pdrs = np.arange(3000, 30000, 1000)
signals = []
for p in tqdm(pdrs):
    cfg["dac"]["res_pulse"]["gain"] = p.item()
    Ts, Is, Qs = zs.measure_lookback(soc, soccfg, cfg, progress=False)
    signals.append(Is + 1j * Qs)
signals = np.array(signals)

# %%
filename = "lookback2D"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    y_info={"name": "pdr", "unit": "", "values": pdrs},
    comment=make_comment(cfg),
    tag="Lookback",
    server_ip=data_host,
)

# %% [markdown]
# # Circle fit

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": "flat_top",
            "raise_pulse": {"style": "gauss", "length": 0.6, "sigma": 0.1},
            "gain": 300,
            "length": 5.0,  # us
            "trig_offset": 2.5,
            "ro_length": 2.5,
        },
    },
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(6010, 6040, 101)
cfg = make_cfg(exp_cfg, reps=200, rounds=200)

fpts, signals = zs.measure_res_freq(soc, soccfg, cfg, instant_show=False)

# %%
num1, num2 = 5, 5
slope1, _ = zf.phase_analyze(fpts[:num1], signals[:num1])
slope2, _ = zf.phase_analyze(fpts[-num2:], signals[-num2:])
slope = (slope1 + slope2) / 2

# %%
c_signals = zf.rotate_phase(fpts, signals, -slope)
plt.plot(c_signals.real, c_signals.imag, marker="o")

# %%
r_f = zf.freq_analyze(fpts, c_signals, asym=True, max_contrast=True)
r_f

# %%
filename = "res_freq"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"resonator frequency = {r_f}MHz"),
    # comment=make_comment(cfg),
    tag="OneTone/freq",
    server_ip=data_host,
)
