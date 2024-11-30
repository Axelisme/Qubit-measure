# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import sys

import numpy as np

# # %cd /home/xilinx/jupyter_notebooks/nthu/sinica-5q/Axel/Qubit-measure
print(os.getcwd())
sys.path.append(os.getcwd())

import zcu_tools.analysis as zf  # noqa: E402
import zcu_tools.schedule.qubit.ef as zs  # noqa: E402

# ruff: noqa: I001
from zcu_tools import (  # noqa: E402
    # reload_zcutools,
    DefaultCfg,
    create_datafolder,
    make_cfg,
    make_sweep,
    save_data,
    save_cfg,
    make_comment,
)

# %% [markdown]
# # Connect to zcu216
# %%
from qick.pyro import make_proxy  # noqa: E402

# ns_host = "140.114.82.71"
ns_host = "100.101.250.4"  # tailscale
ns_port = 8887
proxy_name = "myqick"

soc, soccfg = make_proxy(ns_host=ns_host, ns_port=ns_port, proxy_name=proxy_name)
print(soccfg)


# %%
# from qick import QickSoc  # noqa: E402

# soc = QickSoc()
# soccfg = soc
# print(soc)

# %% [markdown]
# # Utility functions


# %% [markdown]
# # Create data folder

# %%
database_path = create_datafolder(os.getcwd(), "Axel")

# data_host = "192.168.10.252"  # cmd-> ipconfig -> ipv4 #controling computer
data_host = "100.76.229.37"  # tailscale
# data_host = None

# %% [markdown]
# # Predefine parameters

# %%
res_name = "r1"
qubit_name = "q1"
flux_dev = "zcu216"

defaultcfg_path = os.path.join(database_path, "default_cfg.yaml")
DefaultCfg.load(defaultcfg_path)

assert res_name in DefaultCfg.res_cfgs, f"{res_name} not in DefaultCfg.res_cfgs"
assert qubit_name in DefaultCfg.qub_cfgs, f"{qubit_name} not in DefaultCfg.qub_cfgs"

# %%
# set default parameters
DefaultCfg.set_default(
    resonator=res_name,
    qubit=qubit_name,
    ge_pulse="pi",
    flux_dev=flux_dev,
    flux="sw_spot",
)

ro_pulse = "readout_dp1"
r_f = DefaultCfg.res_cfgs[res_name]["freq"]
q_f = DefaultCfg.qub_cfgs[qubit_name]["freq"]


# %% [markdown]
# # Qubit Frequency

# %%
ef_style = "cosine"
exp_cfg = {
    "res_pulse": ro_pulse,
    "ef_pulse": {
        "style": ef_style,
        "gain": 5000,
        "phase": 0,
        "length": 4,
    },
    "relax_delay": 5.0,  # us
}


# %%
quess_q = q_f - 200
exp_cfg["sweep"] = make_sweep(quess_q - 150, quess_q + 150, 5)
cfg = make_cfg(exp_cfg, reps=8, rounds=1)

fpts, signals = zs.measure_ef_freq(soc, soccfg, cfg)

# %%
f_amp, f_pha = zf.freq_analyze(fpts, signals)
f_amp, f_pha

# %%
ef_freq = f_amp
# ef_freq = f_pha
ef_freq

# %%
# ef_freq = 4900

# %%
filename = "ef_freq"
save_cfg(os.path.join(database_path, filename), cfg)
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"qubit ef frequency = {ef_freq}MHz"),
    tag="EF TwoTone",
    server_ip=data_host,
)

# %% [markdown]
# # Amplitude Rabi

# %%
ef_pulse_len = 0.07
exp_cfg = {
    "res_pulse": ro_pulse,
    "ef_pulse": {
        "style": ef_style,
        "freq": ef_freq,
        "length": ef_pulse_len,
    },
    "relax_delay": 70.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 30000, step=5000)
cfg = make_cfg(exp_cfg, reps=500, rounds=1)

pdrs, signals = zs.measure_ef_amprabi(soc, soccfg, cfg)

# %%
pi_gain, pi2_gain, _ = zf.amprabi_analyze(pdrs, signals)
pi_gain = int(pi_gain + 0.5)
pi2_gain = int(pi2_gain + 0.5)
pi_gain, pi2_gain

# %%
# pi_gain = 10000
# pi2_gain = 5000

# %%
filename = "ef_amp_rabi"
save_cfg(os.path.join(database_path, filename), cfg)
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Amplitude", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"pi gain = {pi_gain}\npi2 gain = {pi2_gain}"),
    tag="EF TimeDomain",
    server_ip=data_host,
)

# %% [markdown]
# ## Set EF Pi Pulse

# %%
DefaultCfg.set_qub_pulse(
    qubit_name,
    efpi={
        "style": ef_style,
        "freq": ef_freq,
        "gain": pi_gain,
        "phase": 0,
        "length": ef_pulse_len,
        "desc": "ef pi pulse",
    },
    efpi2={
        "style": ef_style,
        "freq": ef_freq,
        "gain": pi2_gain,
        "phase": 0,
        "length": ef_pulse_len,
        "desc": "ef pi/2 pulse",
    },
)

# %% [markdown]
# # Dispersive shift


# %%
exp_cfg = {
    "res_pulse": ro_pulse,
    "ef_pulse": "efpi",
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(r_f - 5, r_f + 5, 5)
cfg = make_cfg(exp_cfg, reps=1000, rounds=1)

fpts, g_signals, e_signals = zs.measure_ef_dispersive(soc, soccfg, cfg)

# %%
readout_f1, readout_f2 = zf.dispersive_analyze(fpts, g_signals, e_signals, asym=False)
readout_f1, readout_f2

# %%
# readout_f1 = 5700

# %%
filename = "ef dispersive shift"
save_cfg(os.path.join(database_path, filename), cfg)
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={
        "name": "Signal",
        "unit": "a.u.",
        "values": np.array((g_signals, e_signals)),
    },
    y_info={"name": "ge", "unit": "", "values": np.array([0, 1])},
    comment=make_comment(cfg, f"SNR1 = {readout_f1}MHz\nSNR2 = {readout_f2}MHz"),
    tag="EF Dispersive",
    server_ip=data_host,
)


# %% [markdown]
# ## Set Dispersive readout

# %%
DefaultCfg.set_res_pulse(
    res_name,
    readout_ef_dp1={
        **DefaultCfg.get_res_pulse(res_name, ro_pulse),
        "freq": readout_f1,
        "desc": "Readout with largest ef dispersive shift",
    },
    readout_ef_dp2={
        **DefaultCfg.get_res_pulse(res_name, ro_pulse),
        "freq": readout_f2,
        "desc": "Readout with second largest ef dispersive shift",
    },
)
ro_pulse = "readout_ef_dp1"

# %% [markdown]
# # T2Ramsey

# %%
activate_detune = 3.0
orig_ef_freq = ef_freq
exp_cfg = {
    "res_pulse": ro_pulse,
    "ef_pulse": {
        **DefaultCfg.get_qub_pulse(qubit_name, "efpi2"),
        "freq": ef_freq + activate_detune,
    },
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 0.5, 5)  # us
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

Ts1, signals1 = zs.measure_ef_t2ramsey(soc, soccfg, cfg)

t2f, detune = zf.T2fringe_analyze(soc.cycles2us(Ts1), signals1)
detune


# %%
exp_cfg["sweep"] = make_sweep(0, 0.5, 5)
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

Ts2, signals2 = zs.measure_ef_t2ramsey(soc, soccfg, cfg)


# %%
t2d = zf.T2decay_analyze(soc.cycles2us(Ts2), signals2)
t2d

# %%
ef_freq = orig_ef_freq + activate_detune - detune

filename = "ef_t2ramsey"
save_cfg(os.path.join(database_path, filename), cfg)
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": soc.cycles2us(Ts2)},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2},
    comment=make_comment(
        cfg,
        f"activate detune = {activate_detune}MHz\n \
        detune = {detune}MHz\n \
        ef t2f = {t2f}us",
    ),
    tag="EF TimeDomain",
    server_ip=data_host,
)


# %% [markdown]
# # T1

# %%
exp_cfg = {
    "res_pulse": ro_pulse,
    "ef_pulse": "efpi",
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 5, 5)
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

Ts, signals = zs.measure_ef_t1(soc, soccfg, cfg)

# %%
skip_points = 0

t1 = zf.T1_analyze(soc.cycles2us(Ts[skip_points:]), signals[skip_points:])
t1

# %%
filename = "ef_t1"
save_cfg(os.path.join(database_path, filename), cfg)
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": soc.cycles2us(Ts)},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"ef t1 = {t1}us"),
    tag="EF TimeDomain",
    server_ip=data_host,
)

# %% [markdown]
# # T2Echo

# %%
exp_cfg = {
    "res_pulse": ro_pulse,
    "ef_pulse": [("pi", "efpi"), ("pi2", "efpi2")],
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 5, 5)
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

Ts, signals = zs.measure_ef_t2echo(soc, soccfg, cfg)

t2e = zf.T2decay_analyze(soc.cycles2us(Ts * 2), signals)
t2e

# %%
filename = "ef_t2echo"
save_cfg(os.path.join(database_path, filename), cfg)
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": soc.cycles2us(Ts * 2)},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"ef t2echo = {t2e}us"),
    tag="EF TimeDomain",
    server_ip=data_host,
)

# %% [markdown]
# # Dump Configurations

# %%
DefaultCfg.dump(os.path.join(database_path, "default_cfg"))
