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
from pprint import pprint

# %autoreload 2
import zcu_tools.analysis as zf  # noqa: E402
import zcu_tools.schedule.v1 as zs  # noqa: E402

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

zc.config.DATA_DRY_RUN = True  # don't save data
zc.config.YOKO_DRY_RUN = True  # don't run yoko
# zc.config.ZCU_DRY_RUN = True  # don't run zcu

# %% [markdown]
# # Connect to zcu216
# %%
from zcu_tools.remote import make_proxy
from zcu_tools.program.base import MyProgram  # noqa: E402
from zcu_tools.tools import get_ip_address

ns_host = "zcu216-2"
ns_port = 8887
zc.config.LOCAL_IP = get_ip_address("tailscale0")
zc.config.LOCAL_PORT = 8887

soc, soccfg, rm_prog = make_proxy(ns_host, ns_port)
MyProgram.init_proxy(rm_prog, test=True)
print(soccfg)


# %%
# from qick import QickSoc  # noqa: E402

# soc = QickSoc()
# soccfg = soc
# print(soc)

# %% [markdown]
# # Create data folder

# %%
chip_name = "S001"

# data_host = "192.168.10.232"  # cmd-> ipconfig -> ipv4 #controling computer
# data_host = "100.76.229.37"  # tailscale
data_host = None


database_path = create_datafolder(os.getcwd(), prefix=chip_name)

# %% [markdown]
# # Predefine parameters

# %%
DefaultCfg.set_dac(res_ch=4, qub_ch=0)
DefaultCfg.set_adc(ro_chs=[0])
DefaultCfg.set_dev(flux_dev="none", flux=0.0)

# %%
# DefaultCfg.load("default_cfg.yaml")

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
    reinit=True,
)
DefaultCfg.set_dev(flux_dev="yoko")

# %%
cur_flux = 0.0e-3
YokoDevControl.set_current(cur_flux)
DefaultCfg.set_dev(flux=cur_flux)


# %% [markdown]
# # Probe pulse

# %%
DefaultCfg.set_pulse(
    probe_rf={
        # "style": "const",
        # "style": "cosine",
        # "style": "gauss",
        # "sigma": 9.5/4,  # us
        "style": "flat_top",
        # "raise_pulse": {"style": "gauss", "length": 5.0, "sigma": 0.2},
        "raise_pulse": {"style": "cosine", "length": 0.1},
        "length": 5.0,  # us
        "trig_offset": 1.0,  # us
        "ro_length": 4.0,  # us
    },
)
pprint(DefaultCfg.dac_cfgs["pulses"]["probe_rf"])

# %% [markdown]
# # Lookback

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            **DefaultCfg.get_pulse("probe_rf"),
            # **DefaultCfg.get_pulse("readout_dpm"),
            # "gain": 1.0,
            "gain": 30000,
            "freq": 2231.5,
        },
    },
    "adc": {
        "ro_length": 6.0,  # us
        # "trig_offset": 0.0,  # us
        "trig_offset": 0.0,  # us
        "relax_delay": 0.0,  # us
    },
}


# %%
cfg = make_cfg(exp_cfg, rounds=100)

Ts, signals = zs.measure_lookback(soc, soccfg, cfg, progress=True)

# %%
predict_offset = zf.lookback_show(
    Ts, signals, ratio=0.1, pulse_cfg=cfg["dac"]["res_pulse"]
)
predict_offset


# %%
timeFly = float(predict_offset)
# timeFly = 0.5158072916666666
timeFly

# %%
filename = f"lookback_{cur_flux * 1e3:.3f}mA"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"adc_trig_offset = {timeFly}us"),
    tag="Lookback",
    server_ip=data_host,
)

# %%
DefaultCfg.set_adc(timeFly=timeFly)


# %% [markdown]
# ## FFT

# %%
# pulse_name = "const"
pulse_name = cfg["dac"]["res_pulse"]["style"]

signal_records = signal_records if "signal_records" in locals() else {}  # type: ignore  # noqa
signal_records[pulse_name] = (Ts, signals)
pprint(signal_records.keys())
# del signal_records

# %%
freqs, ffts = zf.lookback_fft(signal_records, xrange=(-1, 1), pad_ratio=20)

# %% [markdown]
# # Resonator Frequency

# %%
res_name = f"r5_{cur_flux * 1e3:.3f}mA"
res_name

# %%
exp_cfg = {
    "dac": {
        # "res_pulse": "readout_rf",
        "res_pulse": {
            **DefaultCfg.get_pulse("probe_rf"),
            "nqz": 2,
            # "gain": 1.0,
            "gain": 30000,
        },
        # "reset": "pulse",
        # "reset_pulse": "reset_red"
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = make_sweep(100, 2000, 101)
cfg = make_cfg(exp_cfg, reps=1, rounds=1)

fpts, signals = zs.measure_res_freq(soc, soccfg, cfg)

# %%
r_f = zf.freq_analyze(fpts, signals, asym=True)
r_f

# %%
filename = f"res_freq_{res_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    # comment=make_comment(cfg, f"resonator frequency = {r_f}MHz"),
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
        "res_pulse": {
            **DefaultCfg.get_pulse("probe_rf"),
            "nqz": 2,
        },
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = {
    # "gain": make_sweep(0.05, 1.0, 50),
    "gain": make_sweep(10, 30000, 10, force_int=True),
    "freq": make_sweep(5010, 6040, 10),
    # "freq": make_sweep(r_f-5, r_f+5, 51),
}
cfg = make_cfg(exp_cfg, reps=1, rounds=1)

pdrs, fpts, signals2D = zs.measure_res_pdr_dep(
    soc, soccfg, cfg, dynamic_reps=True, gain_ref=1000
)


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
DefaultCfg.set_dev(flux=cur_flux)

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            **DefaultCfg.get_pulse("probe_rf"),
            "nqz": 2,
            # "gain": 1.0,
            "gain": 30000,
        },
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = {
    "flux": make_sweep(-3.2e-3, -4.3e-3, 11),
    "freq": make_sweep(5020, 6030, 11),
}
cfg = make_cfg(exp_cfg, reps=1, rounds=1)

flxs, fpts, signals2D = zs.measure_res_flux_dep(soc, soccfg, cfg)

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
DefaultCfg.set_dev(flux=cur_flux)

# %% [markdown]
# # Set readout pulse

# %%
r_f = 4000
DefaultCfg.set_pulse(
    readout_rf={
        **DefaultCfg.get_pulse("probe_rf"),
        "freq": r_f,  # MHz
        "gain": 500,
    },
)
pprint(DefaultCfg.get_pulse("readout_rf"))

# %% [markdown]
# # TwoTone

# %%
DefaultCfg.set_pulse(
    probe_qf={
        "style": "flat_top",
        # "raise_pulse": {"style": "gauss", "length": 0.02, "sigma": 0.003},
        "raise_pulse": {"style": "cosine", "length": 0.02},
    },
)
pprint(DefaultCfg.get_pulse("probe_qf"))

# %% [markdown]
# # Twotone Frequency

# %%
# cur_flux = -1.7e-3
YokoDevControl.set_current(cur_flux)
DefaultCfg.set_dev(flux=cur_flux)
cur_flux

# %%
qub_name = f"q5_{cur_flux * 1e3:.3f}mA"

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        # "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse("probe_qf"),
            # "gain": 1.0,
            "nqz": 2,
            "gain": 30000,
            "length": 3,  # us
        },
        # "qub_pulse": "reset_red",
        # "qub_pulse": "pi_len",
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
        # "reset_pulse": {
        #     **DefaultCfg.get_pulse('probe_qf'),
        #     "ch": 2,
        #     "nqz": 2,
        #     "gain": 1.0,
        #     "length": 4,  # us
        # },
    },
    "adc": {
        "relax_delay": 0.4,  # us
    },
}

# %%
# exp_cfg["sweep"] = make_sweep(48.5, 49.5, 51)
exp_cfg["sweep"] = make_sweep(5950, 5990, 101)
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

er_f = 6020.5
fpts, signals = zs.measure_qub_freq(soc, soccfg, cfg)

# %%
f = zf.freq_analyze(fpts, signals, max_contrast=True)
f

# %%
q_f = f
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

# %% [markdown]
# ## reset pulse

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        # "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse("probe_qf"),
            "freq": reset_f,
            # "gain": 1.0,
            "gain": 30000,
        },
        # "reset": "pulse",
        # "reset_pulse": {
        #     **DefaultCfg.get_pulse('probe_qf'),
        #     "nqz": 1,
        #     "ch": 6,
        #     "freq": q_f,
        #     "gain": 30000,
        #     "length": 20,
        # }
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = make_sweep(0.1, 5.0, 101)
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

Ts, signals = zs.measure_lenrabi(soc, soccfg, cfg)

# %%
zf.contrast_plot(Ts, signals, max_contrast=True, xlabel="Length (us)")

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
        **exp_cfg["dac"]["qub_pulse"],
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
            **DefaultCfg.get_pulse("probe_qf"),
            "nqz": 1,
            "length": 5,  # us
        },
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = {
    # "gain": make_sweep(0.05, 1.0, 30),
    "gain": make_sweep(10, 30000, 30, force_int=True),
    "freq": make_sweep(1700, 2000, 30),
}
cfg = make_cfg(exp_cfg, reps=50, rounds=50)

fpts, pdrs, signals2D = zs.measure_qub_pdr_dep(soc, soccfg, cfg)

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
DefaultCfg.set_dev(flux=cur_flux)

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        # "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse("probe_qf"),
            "nqz": 2,
            "gain": 30000,
            "length": 5,  # us
        },
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = {
    "flux": make_sweep(-4.65e-3, -4.46e-3, 11),
    "freq": make_sweep(5000, 8000, 11),
}
cfg = make_cfg(exp_cfg, reps=10, rounds=10)

fpts, flxs, signals2D = zs.measure_qub_flux_dep(soc, soccfg, cfg, reset_rf=None)

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
# YokoDevControl.set_current(cur_flux)
DefaultCfg.set_dev(flux=cur_flux)

# %% [markdown]
# # Rabi

# %% [markdown]
# ## Length Rabi

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        # "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse("probe_qf"),
            "freq": q_f,
            # "gain": 1.0,
            "gain": 30000,
        },
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = make_sweep(0.026, 2.0, 100)
cfg = make_cfg(exp_cfg, reps=10, rounds=10)

Ts, signals = zs.measure_lenrabi(soc, soccfg, cfg)

# %%
pi_len, pi2_len = zf.rabi_analyze(
    Ts, signals, decay=True, max_contrast=True, xlabel="Time (us)"
)
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
        **cfg["dac"]["qub_pulse"],
        "length": pi_len,
        "desc": "pi pulse",
    },
    pi2_len={
        **cfg["dac"]["qub_pulse"],
        "length": pi2_len,
        "desc": "pi/2 pulse",
    },
)
pprint(DefaultCfg.get_pulse("pi_len"))

# %% [markdown]
# ## Amplitude Rabi

# %%
qub_len = 0.2
# qub_len = pi_len * 5.0
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        # "res_pulse": "readout_dpm",
        "qub_pulse": {
            **DefaultCfg.get_pulse("probe_qf"),
            "freq": q_f,
            "length": qub_len,
        },
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = make_sweep(0, 30000, step=1000)
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

pdrs, signals = zs.measure_amprabi(soc, soccfg, cfg)

# %%
pi_gain, pi2_gain = zf.rabi_analyze(
    pdrs, signals, decay=True, max_contrast=True, xlabel="Power (a.u.)"
)
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
        **cfg["dac"]["qub_pulse"],
        "gain": pi_gain,
        "desc": "pi pulse",
    },
    pi2_amp={
        **cfg["dac"]["qub_pulse"],
        "gain": pi2_gain,
        "desc": "pi/2 pulse",
    },
)
pprint(DefaultCfg.get_pulse("pi_amp"))

# %% [markdown]
# # Dispersive shift

# %% [markdown]
# ## Power dependence

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        # "qub_pulse": "pi_len",
        "qub_pulse": "pi_amp",
        # "qub_pulse": "reset_red",
        # "qub_pulse": {
        #     **DefaultCfg.get_pulse('probe_qf'),
        #     "freq": q_f,
        #     "gain": 30000,
        #     "length": 5,  # us
        # },
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = {
    "gain": make_sweep(500, 30000, 5, force_int=True),
    # "gain": make_sweep(0.0, 1.0, 31),
    "freq": make_sweep(6027, 6029, 5),
}
# cfg = make_cfg(exp_cfg, reps=5000, rounds=1)
cfg = make_cfg(exp_cfg, reps=50, rounds=50)

pdrs, fpts, snr2D = zs.qubit.measure_ge_pdr_dep(soc, soccfg, cfg)
ret, records = zs.qubit.ge_different.measure_ge_pdr_dep_auto(
    soc, soccfg, cfg, method="Nelder-Mead"
)

# %%
fpt_max, pdr_max = zf.dispersive2D_analyze(
    fpts, pdrs, snr2D, xlabel="Frequency (MHz)", ylabel="SNR (a.u.)"
)
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
        # "qub_pulse": "pi_len",
        "qub_pulse": "pi_amp",
        # "qub_pulse": "reset_red",
        # "qub_pulse": {
        #     **DefaultCfg.get_pulse('probe_qf'),
        #     "freq": q_f,
        #     "gain": 30000,
        #     "length": 5,  # us
        # },
        # "reset": "pulse",
        # "reset_pulse": "reset_red"
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = make_sweep(1.5, 8.0, 51)
cfg = make_cfg(exp_cfg, reps=1000, rounds=1)

ro_lens, snrs = zs.qubit.measure_ge_ro_dep(soc, soccfg, cfg)

# %%
ro_max = zf.dispersive1D_analyze(ro_lens, snrs, xlabel="Readout Length (us)")

# %%
filename = f"qub_ge_ro_dep_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Readout length", "unit": "s", "values": ro_lens * 1e-6},
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
        # "qub_pulse": "pi_len",
        "qub_pulse": "pi_amp",
        # "qub_pulse": "reset_red",
        # "qub_pulse": {
        #     **DefaultCfg.get_pulse('probe_qf'),
        #     "freq": q_f,
        #     "gain": 30000,
        #     "length": 5,  # us
        # },
        # "reset": "pulse",
        # "reset_pulse": "reset_red"
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
timeFly = 0.5
exp_cfg["sweep"] = make_sweep(timeFly - 0.5, 4.0, 51)
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

offsets, snrs = zs.qubit.measure_ge_trig_dep(soc, soccfg, cfg)

# %%
offset_max = zf.dispersive1D_analyze(offsets, snrs, xlabel="Trigger offset (us)")

# %%
filename = f"qub_ge_trig_dep_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Trigger offset", "unit": "s", "values": offsets * 1e-6},
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
        # "length": ro_max + cfg['adc']['trig_offset'] + 1.0,
        # "ro_length": ro_max,
        "length": ro_max + offset_max + 1.0,
        "ro_length": ro_max + offset_max - cfg["adc"]["trig_offset"],
        "trig_offset": offset_max,
        "desc": "Readout with largest dispersive shift",
    },
)
pprint(DefaultCfg.get_pulse("readout_dpm"))

# %% [markdown]
# ## Readout Lookback

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        "qub_pulse": "pi_amp",
        # "qub_pulse": "pi_len",
    },
    "adc": {"trig_offset": 0.4, "ro_length": ro_max + 5.0, "relax_delay": 0.0},
}

# %%
cfg = make_cfg(exp_cfg, rounds=1000)

Ts, signals_e = zs.measure_lookback(soc, soccfg, cfg, qub_pulse=True)

cfg["dac"]["qub_pulse"]["gain"] = 0
Ts, signals_g = zs.measure_lookback(soc, soccfg, cfg, qub_pulse=True)

# %%
zf.ge_lookback_analyze(Ts, signals_g, signals_e, cfg["dac"]["res_pulse"])

# %%
filename = f"qub_ge_length_{qub_name}"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
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
        "res_pulse": "readout_rf",
        "qub_pulse": {
            **DefaultCfg.get_pulse("pi2_amp"),
            "freq": orig_q_f + activate_detune,
        },
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = make_sweep(0, 30.0, 100)  # us
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

Ts, signals = zs.measure_t2ramsey(soc, soccfg, cfg)

# %%
t2f, detune, _, _ = zf.T2fringe_analyze(Ts, signals, max_contrast=True)  # type: ignore
t2d, _ = zf.T2decay_analyze(Ts, signals, max_contrast=True)
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
        "res_pulse": "readout_rf",
        "qub_pulse": "pi_amp",
        # "qub_pulse": "pi_len",
        # "qub_pulse": "reset_red",
        # "qub_pulse": {
        #     **DefaultCfg.get_pulse('probe_qf'),
        #     "freq": q_f,
        #     "gain": 30000,
        #     "length": 20,  # us
        # },
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = make_sweep(0, 60, 101)
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

Ts, signals = zs.measure_t1(soc, soccfg, cfg)

# %%
start = 0
stop = -1

t1, _ = zf.T1_analyze(
    Ts[start:stop], signals[start:stop], max_contrast=True, dual_exp=False
)
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
import time
from tqdm.auto import trange  # noqa
import pandas as pd  # noqa
import matplotlib.pyplot as plt  # noqa
from IPython.display import clear_output, display  # noqa

exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        "qub_pulse": "pi_len",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}
exp_cfg["sweep"] = make_sweep(0, 1200, 51)
cfg = make_cfg(exp_cfg, reps=10, rounds=10)

start = 3
stop = -1

T1s = []
errs = []
try:
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    dh = display(fig, display_id=True)
    for i in trange(40):
        Ts, signals = zs.measure_t1(soc, soccfg, cfg)
        t1, err = zf.T1_analyze(
            Ts[start:stop],
            signals[start:stop],
            max_contrast=True,
            dual_exp=False,
            plot=False,
        )
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
        ax[0].errorbar(xs, T1s, errs, capsize=4, marker="o", markersize=4, ls="none")
        ax[1].plot(Ts, np.abs(signals))
        clear_output(wait=True)
        dh.update(fig)  # type: ignore
        time.sleep(0.1)
    else:
        clear_output()
finally:
    df = pd.DataFrame([T1s, errs])
    # df.to_csv("T1s.csv")
    # fig.savefig("t1overnight")  # type: ignore

# %% [markdown]
# # T2Echo

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        "pi_pulse": "pi_amp",
        "pi2_pulse": "pi2_amp",
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
}

# %%
exp_cfg["sweep"] = make_sweep(0, 20, 100)
cfg = make_cfg(exp_cfg, reps=100, rounds=100)

Ts, signals = zs.measure_t2echo(soc, soccfg, cfg)

# %%
t2e, _ = zf.T2decay_analyze(Ts, signals)
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
        "res_pulse": "readout_rf",
        "qub_pulse": "pi_amp",
        # "reset": "pulse",
        # "reset_pulse": "reset_red",
    },
    "adc": {
        "relax_delay": 0.0,  # us
    },
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
# # Dump Configurations

# %%
DefaultCfg.dump("default_cfg.yaml")

# %%
