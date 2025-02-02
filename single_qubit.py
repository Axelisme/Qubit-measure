# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# # %cd /home/xilinx/jupyter_notebooks/nthu/sinica-5q/Axel/Qubit-measure
print(os.getcwd())
sys.path.append(os.getcwd())

# %autoreload 2
import zcu_tools.analysis as zf  # noqa: E402
import zcu_tools.schedule as zs  # noqa: E402

# ruff: noqa: I001
from zcu_tools import (  # noqa: E402
    DefaultCfg,
    create_datafolder,
    make_cfg,
    make_sweep,
    save_data,
    make_comment,
)

# %% [markdown]
# # Connect to zcu216
# %%
# from qick.pyro import make_proxy  # noqa: E402

# # ns_host = "140.114.82.71"
# ns_host = "100.101.250.4"  # tailscale
# ns_port = 8887
# proxy_name = "myqick"

# soc, soccfg = make_proxy(ns_host=ns_host, ns_port=ns_port, proxy_name=proxy_name)
# print(soccfg)


# %%
from qick import QickSoc  # noqa: E402

soc = QickSoc()
soccfg = soc
print(soc)

# %% [markdown]
# # Create data folder

# %%
database_path = create_datafolder(os.getcwd(), "Axel")

data_host = "192.168.10.232"  # cmd-> ipconfig -> ipv4 #controling computer
# data_host = "100.76.229.37"  # tailscale
# data_host = None

# %% [markdown]
# # Predefine parameters

# %%
DefaultCfg.set_dac(res_ch=0, qub_ch=6)
DefaultCfg.set_adc(ro_chs=[0])

# %%
# DefaultCfg.load("default_cfg.yaml")
DefaultCfg.dac_cfgs

# %% [markdown]
# # Initialize the flux

# %%
from zcu_tools.device import YokoDevControl  # noqa: E402

YokoDevControl.connect_server(
    {
        "host_ip": data_host,
        # "host_ip": "127.0.0.1",
        "dComCfg": {"address": "0x0B21::0x0039::90ZB35281", "interface": "USB"},
        "outputCfg": {"Current - Sweep rate": 10e-6},
    },
    reinit=True
)
DefaultCfg.set_default(flux_dev="yoko")

# %%
cur_flux = -4.56e-3
# YokoDevControl.set_current(cur_flux)
DefaultCfg.set_default(flux_dev="none")
DefaultCfg.set_default(flux=cur_flux)


# %% [markdown]
# # Probe pulse

# %%
DefaultCfg.set_pulse(
    probe_rf={
        "style": "flat_top",
        "raise_pulse": {
            "style": "gauss",
            "length": 0.6,
            "sigma": 0.1
        },
        "length": 5.0,  # us
        "trig_offset": 2.5,
        "ro_length": 2.5
    },
)
DefaultCfg.dac_cfgs['pulses']['probe_rf']

# %% [markdown]
# # Lookback

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            **DefaultCfg.get_pulse("probe_rf"),
#             **DefaultCfg.get_pulse("readout_dpm"),
            "gain": 5000,
            "freq": 6027.5,
        },
    },
    "adc": {
        "ro_length": 6.0,  # us
#         "trig_offset": 0.0,  # us
        "trig_offset": 0.3,  # us
    },
    "relax_delay": 0.0,  # us
}


# %%
cfg = make_cfg(exp_cfg, rounds=10000)

Ts, Is, Qs = zs.measure_lookback(soc, soccfg, cfg)

# %%
predict_offset = zf.lookback_analyze(Ts, Is, Qs, ratio=0.1, pulse_cfg=cfg['dac']['res_pulse'])
predict_offset


# %%
timeFly = float(predict_offset)
# timeFly = 0.5158072916666666
timeFly

# %%
filename = f"lookback_{cur_flux*1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": Is + 1j * Qs},
    comment=make_comment(cfg, f"adc_trig_offset = {timeFly}us"),
    tag="Lookback",
    server_ip=data_host,
)

# %%
DefaultCfg.set_adc(trig_offset=timeFly)


# %% [markdown]
# # Resonator Frequency

# %%
res_name = f"r5_{cur_flux*1e3:.3f}mA"
res_name

# %%
exp_cfg = {
    "dac": {
#         "res_pulse": "readout_rf",
        "res_pulse": {
            **DefaultCfg.get_pulse("probe_rf"),
            "nqz": 2,
            "gain": 500,
        },
#         "reset_pulse": "reset_red"
    },
#     "reset": "pulse",
    "relax_delay": 0.0,  # us
}

# %%
import zcu_tools.schedule.resonator.onetone as zreo
import zcu_tools.schedule.resonator as zre
importlib.reload(zreo)
importlib.reload(zre)
importlib.reload(zs)

# %%
exp_cfg["sweep"] = make_sweep(6010, 6040, 101)
cfg = make_cfg(exp_cfg, reps=200, rounds=200)

fpts, signals = zs.measure_res_freq(soc, soccfg, cfg, instant_show=False, soft_loop=False)

# %%
r_f = zf.freq_analyze(fpts, signals, asym=True)
r_f

# %%
filename = f"res_freq_{res_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
#     comment=make_comment(cfg, f"resonator frequency = {r_f}MHz"),
        comment=make_comment(cfg),
    tag="OneTone/freq",
    server_ip=data_host,
)

# %%
# r_f =  6026.1
r_f

# %% [markdown]
# # Onetone Dependences

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "probe_rf",
#         "reset_pulse": "reset_red",
    },
#     "reset": "pulse",
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = {
    "gain": make_sweep(100, 1000, step=100),
    "freq": make_sweep(6010, 6040, 51),
#     "freq": make_sweep(r_f-5, r_f+5, 51),
}
cfg = make_cfg(exp_cfg, reps=10000, rounds=1)

fpts, pdrs, signals2D = zs.measure_res_pdr_dep(soc, soccfg, cfg, instant_show=True, dynamic_reps=True, gain_ref=800)


# %%
filename = f"res_pdr_dep_{res_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Power", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg),
    tag="OneTone/pdr",
    server_ip=data_host,
)

# %% [markdown]
# ## Flux dependence

# %%
cur_flux = -3.2e-3
YokoDevControl.set_current(cur_flux)
DefaultCfg.set_default(flux=cur_flux)

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "probe_rf",
#         "res_pulse": {
#             "style": res_style,
#             "freq": r_f,
#             "gain": max_res_gain,
#             "length": res_length,  # us
#         },
    },
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = {
    "flux": make_sweep(-3.2e-3, -4.3e-3, 220),
    "freq": make_sweep(6020, 6030, 51)
}
cfg = make_cfg(exp_cfg, reps=100000, rounds=1)

fpts, flxs, signals2D = zs.measure_res_flux_dep(soc, soccfg, cfg, instant_show=True)

# %%
filename = "res_flux_dep"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Flux", "unit": "a.u.", "values": flxs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg),
    tag="OneTone/flux",
    server_ip=data_host,
)

# %%
cur_flux = -3.8e-3
YokoDevControl.set_current(cur_flux)
DefaultCfg.set_default(flux=cur_flux)

# %% [markdown]
# # Set readout pulse

# %%
DefaultCfg.set_pulse(
    readout_rf={
        **DefaultCfg.get_pulse("probe_rf"),
        "freq": r_f,  # MHz
        "gain": 500,
    },
)
DefaultCfg.get_pulse("readout_rf")

# %% [markdown]
# # TwoTone

# %%
# del DefaultCfg.dac_cfgs['pulses']['probe_qf']

# %%
DefaultCfg.set_pulse(
    probe_qf={
        "style": "flat_top",
        "raise_pulse": {
            "style": "gauss",
            "length": 0.02,
            "sigma": 0.003
        },
    },
)
DefaultCfg.get_pulse("probe_qf")

# %% [markdown]
# # Twotone Frequency

# %%
# cur_flux = -1.7e-3
YokoDevControl.set_current(cur_flux)
DefaultCfg.set_default(flux=cur_flux)
cur_flux

# %%
qub_name = f"q5_{cur_flux*1e3:.3f}mA"

# %%
exp_cfg = {
    "dac": {
#         "res_pulse": "readout_rf",
        "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse('probe_qf'),
            "ch": 6,
            "nqz": 1,
            "gain": 30000,
            "length": 30,  # us
        },
#         "qub_pulse": "reset_red",
#         "qub_pulse": "pi_len",
        "reset_pulse": "reset_red",
#         "reset_pulse": {
#             **DefaultCfg.get_pulse('probe_qf'),
#             "ch": 4,
#             "nqz": 2,
#             "gain": 10000,
#             "length": 4,  # us
#         },
    },
    "reset": "pulse",
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(48.5, 49.5, 201)
# exp_cfg["sweep"] = make_sweep(5950, 5990, 101)
# exp_cfg["sweep"] = make_sweep(q_f-10, q_f+10, 101)
# exp_cfg["sweep"] = make_sweep(reset_f-30, reset_f+30, 101)
# exp_cfg["sweep"] = make_sweep(r_f-reset_f-25, r_f-reset_f+25, 101)
cfg = make_cfg(exp_cfg, reps=100, rounds=200)

er_f = 6020.5
fpts, signals = zs.measure_qub_freq(
    soc, soccfg, cfg, instant_show=True, soft_loop=False, conjugate_reset=False, r_f=er_f, sub_ground=False
)

# %%
f = zf.freq_analyze(fpts, signals, max_contrast=True)
f

# %%
# q_f = f
reset_f = f
# reset_f = r_f - q_f

# %%
filename = f"qub_freq_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"frequency = {f}MHz"),
    tag="TwoTone/freq",
    server_ip=data_host,
)

# %%
q_f

# %% [markdown]
# ## reset pulse

# %%
exp_cfg = {
    "dac": {
#         "res_pulse": "readout_rf",
        "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse('probe_qf'),
            "ch": 4,
            "nqz": 2,
            "freq": reset_f,
            "gain": 30000,
        },
#         "reset_pulse": {
#             **DefaultCfg.get_pulse('probe_qf'),
#             "nqz": 1,
#             "ch": 6,
#             "freq": q_f,
#             "gain": 30000,
#             "length": 20,
#         }
    },
#     "reset": "pulse",
    "relax_delay": 20.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0.1, 5.0, 51)
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

Ts, signals = zs.measure_lenrabi(soc, soccfg, cfg, instant_show=True)

# %%
zf.contrast_plot(Ts, signals, max_contrast=True, xlabel='Length (us)')

# %%
filename = f"reset_time_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Length", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg),
    tag="TwoTone/reset",
    server_ip=data_host,
)

# %%
DefaultCfg.set_pulse(
    reset_red={
        **exp_cfg['dac']['qub_pulse'],
        "length": 2.0,  # us
    },
)
DefaultCfg.get_pulse("reset_red")

# %% [markdown]
# # Twotone Dependences

# %% [markdown]
# ## Power dependence

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        "qub_pulse": {
            **DefaultCfg.get_pulse('probe_qf'),
            "nqz": 1,
            "ch": 6,
            "gain": 30000,
            "length": 5,  # us
        },
    },
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = {
    "gain": make_sweep(10, 30000, 40, force_int=True),
    "freq": make_sweep(q_f - 25, q_f + 25, 51),
#     "freq": make_sweep(1700, 2000, 301)
}
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

fpts, pdrs, signals2D = zs.measure_qub_pdr_dep(
    soc, soccfg, cfg, instant_show=True, soft_freq=True, soft_pdr=True
)

# %%
filename = f"qub_pdr_dep_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Power", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg),
    tag="TwoTone/pdr",
    server_ip=data_host,
)

# %% [markdown]
# ## Flux Dependence

# %%
cur_flux = -4.65e-3
YokoDevControl.set_current(cur_flux)
DefaultCfg.set_default(flux=cur_flux)

# %%
exp_cfg = {
    "dac": {
#         "res_pulse": "readout_rf",
        "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse('probe_qf'),
            "ch": 6,
            "nqz": 1,
            "gain": 30000,
            "length": 5,  # us
        },
        "reset_pulse": "reset_red",
    },
    "reset": "pulse",
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = {
    "flux": make_sweep(-4.65e-3, -4.46e-3, 25),
    "freq": make_sweep(30, 200, 301)
}
cfg = make_cfg(exp_cfg, reps=20, rounds=100)

fpts, flxs, signals2D = zs.measure_qub_flux_dep(soc, soccfg, cfg, instant_show=True, conjugate_reset=True, r_f=r_f)

# %%
filename = "qub_flux_dep"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Flux", "unit": "a.u.", "values": flxs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg),
    tag="TwoTone/flux",
    server_ip=data_host,
)

# %%
cur_flux = 6.0292e-3
YokoDevControl.set_current(cur_flux)
DefaultCfg.set_default(flux=cur_flux)

# %% [markdown]
# # Rabi

# %% [markdown]
# ## Length Rabi

# %%
exp_cfg = {
    "dac": {
#         "res_pulse": "readout_rf",
        "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse('probe_qf'),
            "freq": q_f,
            "gain": 30000
        },
        "reset_pulse": "reset_red",
    },
    "reset": "pulse",
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0.026, 20.0, 50)
cfg = make_cfg(exp_cfg, reps=100, rounds=1000)

Ts, signals = zs.measure_lenrabi(soc, soccfg, cfg, instant_show=True)

# %%
pi_len, pi2_len = zf.rabi_analyze(Ts, signals, decay=True, max_contrast=True, xlabel='Time (us)')
pi_len, pi2_len

# %%
filename = f"len_rabi_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Pulse Length", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"pi len = {pi_len}us\npi/2 len = {pi2_len}us"),
    tag="TimeDomain/rabi",
    server_ip=data_host,
)

# %%
# pi_len = 1.0
# pi2_len = 0.5

# %%
DefaultCfg.set_pulse(
    pi_len={
        **cfg['dac']['qub_pulse'],
        "length": pi_len,
        "desc": "pi pulse",
    },
    pi2_len={
        **cfg['dac']['qub_pulse'],
        "length": pi2_len,
        "desc": "pi/2 pulse",
    },
)
DefaultCfg.get_pulse("pi_len")

# %% [markdown]
# ## Amplitude Rabi

# %%
# qub_len = 0.2

# %%
qub_len = pi_len * 5.0
exp_cfg = {
    "dac": {
        "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse('probe_qf'),
            "freq": q_f,
            "length": qub_len,
        },
        "reset_pulse": "reset_red",
    },
    "reset": "pulse",
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 30000, step=1000)
cfg = make_cfg(exp_cfg, reps=100, rounds=1000)

pdrs, signals = zs.measure_amprabi(soc, soccfg, cfg, soft_loop=True)

# %%
pi_gain, pi2_gain = zf.rabi_analyze(pdrs, signals, decay=True, max_contrast=True, xlabel='Power (a.u.)')
pi_gain = int(pi_gain + 0.5)
pi2_gain = int(pi2_gain + 0.5)
pi_gain, pi2_gain

# %%
filename = f"amp_rabi_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Amplitude", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"pi gain = {pi_gain}\npi/2 gain = {pi2_gain}"),
    tag="TimeDomain/rabi",
    server_ip=data_host,
)

# %%
# pi_gain = 30000
# pi2_gain = 15000

# %%
DefaultCfg.set_pulse(
    pi_amp={
        **cfg['dac']['qub_pulse'],
        "gain": pi_gain,
        "desc": "pi pulse",
    },
    pi2_amp={
        **cfg['dac']['qub_pulse'],
        "gain": pi2_gain,
        "desc": "pi/2 pulse",
    },
)
DefaultCfg.get_pulse("pi_amp")

# %% [markdown]
# # Dispersive shift

# %% [markdown]
# ## Power dependence

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
#         "qub_pulse": "pi_len",
#         "qub_pulse": "pi_amp",
        "qub_pulse": "reset_red"
#         "qub_pulse": {
#             **DefaultCfg.get_pulse('probe_qf'),
#             "freq": q_f,
#             "gain": 30000,
#             "length": 5,  # us
#         },
#         "reset_pulse": "reset_red",
    },
#     "reset": "pulse",
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = {
#     "gain": make_sweep(500, 15000, step=1000),
#     "freq": make_sweep(6023, 6032, 31),
    "gain": make_sweep(11000, 16000, step=200),
    "freq": make_sweep(6027, 6029, 41),
}
# cfg = make_cfg(exp_cfg, reps=5000, rounds=1)
cfg = make_cfg(exp_cfg, reps=10000, rounds=1)

fpts, pdrs, snr2D = zs.qubit.measure_ge_pdr_dep(soc, soccfg, cfg, instant_show=True)

# %%
fpt_max, pdr_max = zf.dispersive2D_analyze(fpts, pdrs, snr2D, xlabel='Frequency (MHz)', ylabel='SNR (a.u.)')
fpt_max, pdr_max

# %%
filename = f"qub_ge_contrast_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Power", "unit": "a.u.", "values": pdrs},
    z_info={"name": "SNR", "unit": "a.u.", "values": snr2D},
    comment=make_comment(cfg),
    tag="TwoTone/dispersive",
    server_ip=data_host,
)

# %% [markdown]
# ## Readout dependence

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            **DefaultCfg.get_pulse("readout_rf"),
            "freq": fpt_max,
            "gain": pdr_max,
        },
#         "qub_pulse": "pi_len",
#         "qub_pulse": "pi_amp",
        "qub_pulse": "reset_red",
#         "qub_pulse": {
#             **DefaultCfg.get_pulse('probe_qf'),
#             "freq": q_f,
#             "gain": 30000,
#             "length": 5,  # us
#         },
#         "reset_pulse": "reset_red"
    },
#     "reset": "pulse",
#     "adc": {"trig_offset": 0},
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0.5, 10.0, 51)
cfg = make_cfg(exp_cfg, reps=100000, rounds=1)

ro_lens, snrs = zs.qubit.measure_ge_ro_dep(soc, soccfg, cfg, instant_show=True)

# %%
ro_max = zf.dispersive1D_analyze(ro_lens, snrs, xlabel="Readout Length (us)")

# %%
filename = f"qub_ge_ro_dep_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Readout length", "unit": "s", "values": ro_lens*1e-6},
    z_info={"name": "SNR", "unit": "a.u.", "values": snrs},
    comment=make_comment(cfg),
    tag="TwoTone/dispersive",
    server_ip=data_host,
)

# %% [markdown]
# ## Trigger offset dependence

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            **DefaultCfg.get_pulse("readout_rf"),
            "freq": fpt_max,
            "gain": pdr_max,
        },
#         "qub_pulse": "pi_len",
#         "qub_pulse": "pi_amp",
        "qub_pulse": "reset_red",
#         "qub_pulse": {
#             **DefaultCfg.get_pulse('probe_qf'),
#             "freq": q_f,
#             "gain": 30000,
#             "length": 5,  # us
#         },
#         "reset_pulse": "reset_red"
    },
#     "reset": "pulse",
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(timeFly-0.5, 4.0, 51)
cfg = make_cfg(exp_cfg, reps=10000, rounds=1)

offsets, snrs = zs.qubit.measure_ge_trig_dep(soc, soccfg, cfg, instant_show=True)

# %%
offset_max = zf.dispersive1D_analyze(offsets, snrs, xlabel="Trigger offset (us)")

# %%
filename = f"qub_ge_trig_dep_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Trigger offset", "unit": "s", "values": offsets*1e-6},
    z_info={"name": "SNR", "unit": "a.u.", "values": snrs},
    comment=make_comment(cfg),
    tag="TwoTone/dispersive",
    server_ip=data_host,
)

# %%
DefaultCfg.set_pulse(
    readout_dpm={
        **DefaultCfg.get_pulse("readout_rf"),
        "freq": fpt_max,
        "gain": pdr_max,
#         "length": ro_max + cfg['adc']['trig_offset'] + 1.0,
#         "ro_length": ro_max,
        "length": ro_max + offset_max + 1.0,
        "ro_length": ro_max + offset_max - cfg['adc']['trig_offset'],
        "trig_offset": offset_max,
        "desc": "Readout with largest dispersive shift",
    },
)
DefaultCfg.get_pulse("readout_dpm")

# %% [markdown]
# ## Readout Lookback

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_dpm",
#         "qub_pulse": "pi_amp",
        "qub_pulse": "pi_len",
    },
    "adc": {
        "trig_offset": 0.4,
        "ro_length": ro_max + 5.0
    },
    "relax_delay": 0.0,  # us
}

# %%
cfg = make_cfg(exp_cfg, rounds=10000)

Ts, Ise, Qse = zs.measure_lookback(soc, soccfg, cfg, qub_pulse=True)
signals_e = Ise+1j*Qse

cfg["dac"]["qub_pulse"]["gain"] = 0
Ts, Isg, Qsg = zs.measure_lookback(soc, soccfg, cfg, qub_pulse=True)
signals_g = Isg+1j*Qsg

# %%
zf.ge_lookback_analyze(Ts, signals_g, signal_e, cfg['res_pulse'])

# %%
filename = f"qub_ge_length_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts*1e-6},
    z_info={
        "name": "Signal",
        "unit": "a.u.",
        "values": np.array((signals_g, signals_e)),
    },
    y_info={"name": "ge", "unit": "", "values": np.array([0, 1])},
    comment=make_comment(cfg),
    tag="TwoTone/dispersive",
    server_ip=data_host,
)

# %% [markdown]
# # T2Ramsey

# %%
# activate_detune = 0.2
activate_detune = 0.0
orig_q_f = q_f
exp_cfg = {
    "dac": {
        "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse("pi2_len"),
            "freq": orig_q_f + activate_detune,
        },
        "reset_pulse": "reset_red",
    },
    "reset": "pulse",
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 30.0, 100)  # us
cfg = make_cfg(exp_cfg, reps=100, rounds=1000)

Ts, signals = zs.measure_t2ramsey(soc, soccfg, cfg)

# %%
t2f, detune = zf.T2fringe_analyze(Ts, signals, max_contrast=True)
t2d = zf.T2decay_analyze(Ts, signals, max_contrast=True)
detune, t2f, t2d

# %%
filename = f"t2ramsey_{qub_name}"
# filename = f"t2decay_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"detune = {detune}MHz\nt2f = {t2f}us\nt2d = {t2d}us"),
    tag="TimeDomain/t2ramsey",
    server_ip=data_host,
)

# %%
q_f = orig_q_f + activate_detune - detune
q_f

# %% [markdown]
# # T1

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_dpm",
#         "qub_pulse": "pi_amp",
        "qub_pulse": "pi_len",
#         "qub_pulse": "reset_red",
#         "qub_pulse": {
#             **DefaultCfg.get_pulse('probe_qf'),
#             "freq": q_f,
#             "gain": 30000,
#             "length": 20,  # us
#         },
        "reset_pulse": "reset_red",
    },
    "reset": "pulse",
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 60, 101)
cfg = make_cfg(exp_cfg, reps=100, rounds=1000)

Ts, signals = zs.measure_t1(soc, soccfg, cfg)

# %%
start = 0
stop = -1

t1 = zf.T1_analyze(Ts[start:stop], signals[start:stop], max_contrast=True, dual_exp=False)
t1

# %%
filename = f"t1_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"t1 = {t1}us"),
    tag="TimeDomain/t1",
    server_ip=data_host,
)


# %% [markdown]
# ## T1 overnight

# %%
from tqdm.auto import trange
import pandas as pd 
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

exp_cfg = {
    "dac": {
        "res_pulse": "readout_dpm",
        "qub_pulse": "pi_len",
    },
    "relax_delay": 1000.0,  # us
}
exp_cfg["sweep"] = make_sweep(0, 1200, 51)
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

start = 3
stop = -1

T1s = []
errs = []
try:
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    dh = display(fig, display_id=True)
    for i in trange(40):
        Ts, signals = zs.measure_t1(soc, soccfg, cfg)
        t1, err = zf.T1_analyze(Ts[start:stop], signals[start:stop], max_contrast=True, dual_exp=False, plot=False, return_err=True)
        T1s.append(t1)
        errs.append(err)
        save_data(
            filepath=os.path.join(database_path, f"t1_{qub_name}"),
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=make_comment(cfg, f"t1 = {t1}us"),
            tag="TimeDomain/t1overnight",
            server_ip=data_host,
        )
        
        xs = np.arange(len(T1s))
        ax[0].clear()
        ax[1].clear()
        ax[0].errorbar(xs, T1s, errs, capsize=4, marker='o', markersize=4, ls='none')
        ax[1].plot(Ts, np.abs(signals))
        dh.update(fig)
    else:
        clear_output()
except Exception as e:
    print(e)

df = pd.DataFrame([T1s, errs])
df.to_csv("T1s.csv")
fig.savefig("t1overnight")

# %% [markdown]
# # T2Echo

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_dpm",
        "pi_pulse": "pi_len",
        "pi2_pulse": "pi2_len",
        "reset_pulse": "reset_red",
    },
    "reset": "pulse",
    "relax_delay": 0.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 20, 100)
cfg = make_cfg(exp_cfg, reps=100, rounds=1000)

Ts, signals = zs.measure_t2echo(soc, soccfg, cfg)

# %%
t2e = zf.T2decay_analyze(Ts, signals)
t2e

# %%
filename = f"t2echo_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"t2echo = {t2e}us"),
    tag="TimeDomain/t2echo",
    server_ip=data_host,
)

# %% [markdown]
# # Single shot

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_dpm",
        "qub_pulse": "pi_len",
        "reset_pulse": "reset_red",
    },
    "reset": "pulse",
    "relax_delay": 0.0,  # us
}

# %%
cfg = make_cfg(exp_cfg, shots=5000)

fid, threshold, angle, signals = zs.measure_fid_auto(soc, soccfg, cfg, plot=True)
print(f"Optimal fidelity after rotation = {fid:.1%}")

# %%
filename = f"single_shot_rf_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "shot", "unit": "point", "values": np.arange(cfg["shots"])},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    y_info={"name": "ge", "unit": "", "values": np.array([0, 1])},
    comment=make_comment(cfg, f"fide {fid:.1%}"),
    tag="SingleShot",
    server_ip=data_host,
)

# %% [markdown]
# ## Tuning single shot readout

# %%
# initial parameters
best_freq = r_f
best_pdr = max_gain
best_res_len = res_length
best_offset = trig_offset
best_ro_len = res_length
DefaultCfg.set_pulse(
    readout_fid={
        "style": res_style,
        "freq": best_freq,
        "gain": best_pdr,
        "phase": 0,
        "length": best_res_len,
        "trig_offset": best_offset,
        "ro_length": best_ro_len,
        "desc": "Readout with best fidelity",
    },
)

# %% [markdown]
# ### Scan readout power

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": res_style,
            "freq": best_freq,
            "length": best_res_len,
            "trig_offset": best_offset,
            "ro_length": best_ro_len,
        },
        "qub_pulse": "pi_amp",
    },
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(best_pdr - 1000, best_pdr + 3000, step=500)
cfg = make_cfg(exp_cfg, shots=5000)

pdrs, fids = zs.qubit.scan_pdr(soc, soccfg, cfg, instant_show=True, reps=40)

# %%
max_id = np.argmax(fids)
best_pdr, best_fid = pdrs[max_id], fids[max_id]
plt.plot(pdrs, fids, marker="s")
plt.xlabel("Pulse gain")
plt.ylabel("Fidelity")
plt.title(f"Max fide = {max(fids):.2f}")
plt.axvline(best_pdr, label=f"max fide gain = {best_pdr}", ls="--")
plt.legend()

# %%
# best_pdr = max_res_gain

# %%
filename = f"scan_gain_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "gain", "unit": "a. u.", "values": pdrs},
    z_info={"name": "Score", "unit": "a.u.", "values": fids},
    comment=make_comment(cfg, f"best gain = {best_pdr}, best fide = {best_fid:.2}"),
    tag="SingleShot/pdr",
    server_ip=data_host,
)

# %% [markdown]
# ### Scan readout frequency

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": res_style,
            "gain": best_pdr,
            "length": best_res_len,
            "trig_offset": best_offset,
            "ro_length": best_ro_len,
        },
        "qub_pulse": "pi_len",
    },
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(best_freq - 3, best_freq + 5, 21)
cfg = make_cfg(exp_cfg, shots=5000)

fpts, fids = zs.qubit.scan_freq(soc, soccfg, cfg, instant_show=True, reps=20)

# %%
max_id = np.argmax(fids)
best_freq, best_fid = fpts[max_id], fids[max_id]
plt.plot(fpts, fids, marker="s")
plt.xlabel("Frequency")
plt.ylabel("Fidelity")
plt.title(f"Max fide = {best_fid:.2}")
plt.axvline(best_freq, label=f"max fide freq = {best_freq}", ls="--")
plt.legend()
plt.show()

# %%
# best_freq = readout_fm

# %%
filename = f"scan_freq_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Pulse freq", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Score", "unit": "a.u.", "values": fids},
    comment=make_comment(cfg, f"best freq = {best_freq}, best fide = {best_fid:.2}"),
    tag="SingleShot/freq",
    server_ip=data_host,
)

# %% [markdown]
# ### Scan readout trig_offset

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": res_style,
            "freq": best_freq,
            "length": best_res_len,
            "gain": best_pdr,
            "trig_offset": best_offset,
            "ro_length": best_ro_len,
        },
        "qub_pulse": "pi_amp",
    },
    "relax_delay": 15.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(best_offset-0.2, best_offset+0.2, 11)
cfg = make_cfg(exp_cfg, shots=5000)

offsets, fids = zs.qubit.scan_offset(soc, soccfg, cfg, instant_show=True, reps=40)

# %%
max_id = np.argmax(fids)
best_offset, best_fid = offsets[max_id], fids[max_id]
best_ro_len = cfg["adc"]["ro_length"] + cfg["adc"]["trig_offset"] - best_offset
plt.close()
plt.plot(offsets, fids, marker="s")
plt.xlabel("Pulse length")
plt.ylabel("Fidelity")
plt.title(f"Max fide = {max(fids):.2f}")
plt.axvline(best_offset, label=f"max fide offset = {best_offset}", ls="--")
plt.legend()
plt.show()
best_offset, best_ro_len

# %%
filename = f"scan_offset_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "trig offset", "unit": "us", "values": lens},
    z_info={"name": "Score", "unit": "a.u.", "values": fids},
    comment=make_comment(cfg, f"best trig offset = {best_offset}, best fide = {best_fid:.2f}"),
    tag="SingleShot/trig_offset",
    server_ip=data_host,
)

# %% [markdown]
# ### Scan readout length

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": res_style,
            "freq": best_freq,
            "gain": best_pdr,
            "length": best_res_len,
            "trig_offset": best_offset,
        },
        "qub_pulse": "pi_amp",
    },
    "relax_delay": 15.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(best_ro_len - 0.1, best_ro_len + 0.2, 11)
cfg = make_cfg(exp_cfg, shots=5000)

lens, fids = zs.qubit.scan_ro_len(soc, soccfg, cfg, instant_show=True, reps=40)

# %%
max_id = np.argmax(fids)
# max_id = 10
best_ro_len, best_fid = lens[max_id], fids[max_id]
plt.plot(lens, fids, marker="s")
plt.xlabel("Pulse length")
plt.ylabel("Fidelity")
plt.title(f"Max fide = {max(fids):.2f}")
plt.axvline(best_ro_len, label=f"max fide ro_length = {best_ro_len}", ls="--")
plt.legend()
best_ro_len

# %%
# best_len = res_length

# %%
filename = f"scan_ro_len"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "RO length", "unit": "us", "values": lens},
    z_info={"name": "Score", "unit": "a.u.", "values": fids},
    comment=make_comment(cfg, f"best ro_length = {best_ro_len}, best fide = {best_fid:.2}"),
    tag="SingleShot/ro_length",
    server_ip=data_host,
)

# %% [markdown]
# ### Scan readout pulse length

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": res_style,
            "gain": best_pdr,
            "freq": best_freq,
            "trig_offset": best_offset,
            "length": best_res_len,
            "ro_length": best_ro_len,
        },
        "qub_pulse": "pi_amp",
    },
    "relax_delay": 15.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(best_res_len - 0.2, best_res_len + 0.2, 11)
cfg = make_cfg(exp_cfg, shots=5000)

lens, fids = zs.qubit.scan_res_len(soc, soccfg, cfg, instant_show=True, reps=40)

# %%
max_id = np.argmax(fids)
best_res_len, best_fid = lens[max_id], fids[max_id]
best_ro_len = cfg["adc"]["ro_length"] + best_res_len - cfg["dac"]["res_pulse"]["length"]
plt.plot(lens, fids, marker="s")
plt.xlabel("Res length")
plt.ylabel("Fidelity")
plt.title(f"Max fide = {best_fid:.2}")
plt.axvline(best_res_len, label=f"max fide res length = {best_res_len}", ls="--")
plt.legend()
plt.show()
best_res_len, best_ro_len

# %%
# best_res_len = 0.5

# %%
filename = f"scan_res_len_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Pulse length", "unit": "us", "values": lens},
    z_info={"name": "Score", "unit": "a.u.", "values": fids},
    comment=make_comment(cfg, f"best res length = {best_res_len}, best fide = {best_fid:.2f}"),
    tag="SingleShot/res_length",
    server_ip=data_host,
)

# %% [markdown]
# ### Set best readout pulse

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": res_style,
            "freq": best_freq,
            "gain": best_pdr,
            "length": best_res_len,
            "trig_offset": best_offset,
            "ro_length": best_ro_len,
        },
        "qub_pulse": "pi_amp",
    },
    "relax_delay": 15.0,  # us
}
cfg = make_cfg(exp_cfg, shots=5000)

fid, threshold, angle, signals = zs.measure_fid_auto(soc, soccfg, cfg, plot=True)
print(f"Optimal fidelity after rotation = {fid:.1%}")

# %%
filename = f"single_shot_fid_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "shot", "unit": "point", "values": np.arange(cfg["shots"])},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    y_info={"name": "ge", "unit": "", "values": np.array([0, 1])},
    comment=make_comment(cfg, f"fide {fid:.2f}"),
    tag="SingleShot",
    server_ip=data_host,
)

# %%
DefaultCfg.set_pulse(
    readout_fid2={
        "style": res_style,
        "freq": best_freq,
        "gain": best_pdr,
        "phase": -angle,
        "length": best_res_len,
        "trig_offset": best_offset,
        "ro_length": best_ro_len,
        "threshold": threshold,
    },
)

# %% [markdown]
# # Dump Configurations

# %%
DefaultCfg.dump("default_cfg.yaml")

# %%
