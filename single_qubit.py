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
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
DefaultCfg.set_dac(res_ch=0, qub_ch=3, res_nqz=2, qub_nqz=1)
DefaultCfg.set_adc(ro_chs=[0])

# %%
# DefaultCfg.load("default_cfg.yaml")

# %% [markdown]
# Initialize the flux

# %%
from zcu_tools.device import YokoDevControl  # noqa: E402

YokoDevControl.connect_server(
    {
        "host_ip": data_host,
        # "host_ip": "127.0.0.1",
        "dComCfg": {"address": "0x0B21::0x0039::90ZB35281", "interface": "USB"},
        "outputCfg": {"Current - Sweep rate": 10e-6},
    }
)
DefaultCfg.set_default(flux_dev="yoko")

# %%
cur_flux = 0
YokoDevControl.set_current(cur_flux)
DefaultCfg.set_default(flux=cur_flux)


# %% [markdown]
# # Lookback

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": "const",
            "freq": 5892,  # MHz
            "gain": 8000,
            "length": 1,  # us
        },
    },
    "adc": {
        "ro_length": 3,  # us
        # "trig_offset": 0,  # us
        # "trig_offset": 0.470,  # us
    },
    "relax_delay": 10.0,  # us
}


# %%
cfg = make_cfg(exp_cfg, rounds=10000)

Ts, Is, Qs = zs.measure_lookback(soc, soccfg, cfg)

# %%
predict_offset = zf.lookback_analyze(Ts, Is, Qs, ratio=0.5)


# %%
# trig_offset = cfg["adc"]["trig_offset"]
trig_offset = float(predict_offset) + cfg["adc"]["trig_offset"]
trig_offset

# %%
filename = "lookback"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": Is + 1j * Qs},
    comment=make_comment(cfg, f"adc_trig_offset = {trig_offset}us"),
    tag="Lookback",
    server_ip=data_host,
)

# %%
DefaultCfg.set_adc(trig_offset=trig_offset)


# %% [markdown]
# # Resonator Frequency

# %%
res_style = "const"
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": res_style,
            "gain": 1000,
            "length": 5,
        },
    },
    "relax_delay": 0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(5400, 6300, 901)
cfg = make_cfg(exp_cfg, reps=5000, rounds=1)

fpts, signals = zs.measure_res_freq(soc, soccfg, cfg)

plt.plot(fpts, np.abs(signals))

# %%
sorted_fpts = fpts[np.argsort(np.abs(signals))]
print("Max Amp: ", np.sort(sorted_fpts[-6:]))
print("Min Amp: ", np.sort(sorted_fpts[:6]))


# %%
guess_r = 5882

exp_cfg["sweep"] = make_sweep(guess_r - 35, guess_r + 35, 121)
cfg = make_cfg(exp_cfg, res_pulse={"gain": 1000}, reps=100000, rounds=1)

fpts, signals = zs.measure_res_freq(soc, soccfg, cfg)

# %%
r_f, _ = zf.freq_analyze(fpts, signals, asym=True)
r_f

# %%
filename = "res_freq"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"resonator frequency = {r_f}MHz"),
    #     comment=make_comment(cfg),
    tag="OneTone/freq",
    server_ip=data_host,
)

# %% [markdown]
# # Onetone Dependences

# %%
res_length = 5

# %% [markdown]
# ## Power dependence

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": res_style,
            "freq": r_f,
            "length": res_length,  # us
        },
    },
    "relax_delay": 3.0,  # us
}

# %%
exp_cfg["sweep"] = {
    "pdr": make_sweep(100, 6000, 50, force_int=True),
    "freq": make_sweep(r_f - 15, r_f + 15, 60),
}
cfg = make_cfg(exp_cfg, reps=20000, rounds=1)

fpts, pdrs, signals2D = zs.measure_res_pdr_dep(
    soc, soccfg, cfg, instant_show=True, soft_loop=False
)


# %%
peak_freqs = zf.spectrum_analyze(fpts, pdrs, signals2D)

# %%
filename = "res_pdr_dep"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Power", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg),
    tag="OneTone/pdr",
    server_ip=data_host,
)

# %%
max_res_gain = 2000
DefaultCfg.set_dac(max_res_gain=max_res_gain)

# %% [markdown]
# ## Flux dependence

# %%
cur_flux = 0
YokoDevControl.set_current(cur_flux)
DefaultCfg.set_default(flux=cur_flux)

# %%
exp_cfg = {
    "dac": {
        "res_pulse": {
            "style": res_style,
            "freq": r_f,
            "gain": max_res_gain,
            "length": res_length,  # us
        },
    },
    "relax_delay": 3.0,  # us
}

# %%
exp_cfg["sweep"] = {
    "flux": make_sweep(-2.40e-3, -2.30e-3, 50),
    "freq": make_sweep(r_f - 10, r_f + 10, 50),
}
cfg = make_cfg(exp_cfg, reps=5000, rounds=1)

fpts, flxs, signals2D = zs.measure_res_flux_dep(soc, soccfg, cfg, instant_show=True)

# %%
peak_freqs = zf.spectrum_analyze(fpts, flxs, signals2D, f_axis="y-axis")


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

# %% [markdown]
# ## Set readout pulse

# %%
ro_length = 5
DefaultCfg.set_pulse(
    readout_rf={
        "style": res_style,
        "freq": r_f,
        "gain": max_res_gain,
        "length": ro_length,
        "desc": "Readout with resonator freq",
    }
)

# %% [markdown]
# # Qubit Frequency

# %%
qub_style = "cosine"
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        "qub_pulse": {
            "style": qub_style,
            "gain": 5000,
            "length": 4,
        },
    },
    "relax_delay": 0.0,  # us
}

# %%
quess_q = 4658
# exp_cfg["sweep"] = make_sweep(quess_q - 25, quess_q + 25, 5)
exp_cfg["sweep"] = make_sweep(6000, 7000, 501)
cfg = make_cfg(exp_cfg, reps=10000, rounds=1)

fpts, signals = zs.measure_qubit_freq(soc, soccfg, cfg)

# %%
f_amp, f_pha = zf.freq_analyze(fpts, signals)
f_amp, f_pha

# %%
# q_f = 1500
q_f = f_amp
# q_f = f_pha
q_f

# %%
filename = "qub_freq"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"qubit frequency = {q_f}MHz"),
    tag="TwoTone/freq",
    server_ip=data_host,
)

# %% [markdown]
# # Twotone Dependences

# %% [markdown]
# ## Power dependence

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        "qub_pulse": {
            "style": qub_style,
            "freq": q_f,
            "length": 4,  # us
        },
    },
    "relax_delay": 3.0,  # us
}

# %%
exp_cfg["sweep"] = {
    "pdr": make_sweep(100, 6000, 50, force_int=True),
    "freq": make_sweep(r_f - 15, r_f + 15, 60),
}
cfg = make_cfg(exp_cfg, reps=20000, rounds=1)

fpts, pdrs, signals2D = zs.measure_qub_pdr_dep(
    soc, soccfg, cfg, instant_show=True, soft_loop=False
)

# %%
peak_freqs = zf.spectrum_analyze(fpts, pdrs, signals2D)

# %%
filename = "qub_pdr_dep"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Power", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg),
    tag="TwoTone/pdr",
    server_ip=data_host,
)

# %%
max_qub_gain = 2000
DefaultCfg.set_dac(max_qub_gain=max_qub_gain)

# %% [markdown]
# ## Flux Dependence

# %%
cur_flux = 0
YokoDevControl.set_current(cur_flux)
DefaultCfg.set_default(flux=cur_flux)

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        "qub_pulse": {
            "style": qub_style,
            "freq": q_f,
            "gain": max_qub_gain,
            "length": 4,  # us
        },
    },
    "relax_delay": 3.0,  # us
}

# %%
exp_cfg["sweep"] = {
    #     "flux": make_sweep(-30000, 30000, step=10000),
    "flux": make_sweep(-2.40e-3, -2.30e-3, 50),
    "freq": make_sweep(r_f - 10, r_f + 10, 50),
}
cfg = make_cfg(exp_cfg, reps=5000, rounds=1)

fpts, flxs, signals2D = zs.measure_qub_flux_dep(soc, soccfg, cfg, instant_show=True)

# %%
peak_freqs = zf.spectrum_analyze(fpts, flxs, signals2D, f_axis="y-axis")

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
cur_flux = 0
YokoDevControl.set_current(cur_flux)
DefaultCfg.set_default(flux=cur_flux)

# %% [markdown]
# # Rabi

# %% [markdown]
# ## Length Rabi

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        "qub_pulse": {
            "style": qub_style,
            "freq": q_f,
            "gain": max_qub_gain,
        },
    },
    "relax_delay": 20.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 10, 50)
cfg = make_cfg(exp_cfg, reps=100, rounds=500)

Ts, signals = zs.measure_lenrabi(soc, soccfg, cfg)

# %%
pi_len, pi2_len, _ = zf.rabi_analyze(Ts, signals)
pi_len, pi2_len

# %%
filename = "len_rabi"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Pulse Length", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"pi len = {pi_len}us\npi/2 len = {pi2_len}us"),
    tag="TimeDomain/rabi",
    server_ip=data_host,
)

# %%
# pi_len = 10000
# pi2_len = 5000

# %% [markdown]
# ## Amplitude Rabi

# %%
qub_len = pi_len
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        "qub_pulse": {
            "style": qub_style,
            "freq": q_f,
            "length": qub_len,
        },
    },
    "relax_delay": 20.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 30000, step=500)
cfg = make_cfg(exp_cfg, reps=100, rounds=500)

pdrs, signals = zs.measure_amprabi(soc, soccfg, cfg)

# %%
pi_gain, pi2_gain, _ = zf.rabi_analyze(pdrs, signals)
pi_gain = int(pi_gain + 0.5)
pi2_gain = int(pi2_gain + 0.5)
pi_gain, pi2_gain

# %%
filename = "amp_rabi"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Amplitude", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"pi gain = {pi_gain}\npi/2 gain = {pi2_gain}"),
    tag="TimeDomain/rabi",
    server_ip=data_host,
)

# %%
# pi_gain = 10000
# pi2_gain = 5000

# %% [markdown]
# ## Set Pi / Pi2 Pulse

# %%
DefaultCfg.set_pulse(
    pi={
        "style": qub_style,
        "freq": q_f,
        "gain": pi_gain,
        "phase": 0,
        "length": qub_len,
        "desc": "pi pulse",
    },
    pi2={
        "style": qub_style,
        "freq": q_f,
        "gain": pi2_gain,
        "phase": 0,
        "length": qub_len,
        "desc": "pi/2 pulse",
    },
)

# %% [markdown]
# # Dispersive shift


# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_rf",
        "qub_pulse": "pi",
    },
    "relax_delay": 20.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(r_f - 10, r_f + 10, 100)
cfg = make_cfg(exp_cfg, reps=10000, rounds=1)

fpts, g_signals, e_signals = zs.measure_dispersive(soc, soccfg, cfg)

# %%
readout_f1, readout_f2 = zf.dispersive_analyze(fpts, g_signals, e_signals, asym=True)
readout_f1, readout_f2

# %%
# readout_f1 = 5881

# %%
filename = "disper_shift"
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
    tag="TwoTone/dispersive",
    server_ip=data_host,
)


# %% [markdown]
# ## Set Dispersive readout

# %%
DefaultCfg.set_pulse(
    readout_dp1={
        **DefaultCfg.get_pulse("readout_rf"),
        "freq": readout_f1,
        "desc": "Readout with largest dispersive shift",
    },
    readout_dp2={
        **DefaultCfg.get_pulse("readout_rf"),
        "freq": readout_f2,
        "desc": "Readout with second largest dispersive shift",
    },
)

# %% [markdown]
# # T2Ramsey

# %%
# activate_detune = 3.0
activate_detune = 0.0
orig_q_f = q_f
exp_cfg = {
    "dac": {
        "res_pulse": "readout_dp1",
        "qub_pulse": {
            **DefaultCfg.get_pulse("pi2"),
            "freq": orig_q_f + activate_detune,
        },
    },
    "relax_delay": 20.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 5, 200)  # us
cfg = make_cfg(exp_cfg, reps=100, rounds=500)

Ts, signals = zs.measure_t2ramsey(soc, soccfg, cfg)

# %%
t2f, detune = zf.T2fringe_analyze(Ts, signals)
t2d = zf.T2decay_analyze(Ts, signals)
detune, t2f, t2d

# %%
filename = "t2ramsey"
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

# %% [markdown]
# # T1

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_dp1",
        "qub_pulse": "pi",
    },
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 10, 200)
cfg = make_cfg(exp_cfg, reps=10000, rounds=1)

Ts, signals = zs.measure_t1(soc, soccfg, cfg)

# %%
skip_points = 0

t1 = zf.T1_analyze(Ts[skip_points:], signals[skip_points:])
t1

# %%
filename = "t1"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"t1 = {t1}us"),
    tag="TimeDomain/t1",
    server_ip=data_host,
)


# %% [markdown]
# # T2Echo

# %%
exp_cfg = {
    "dac": {
        "res_pulse": "readout_dp1",
        "pi_pulse": "pi",
        "pi2_pulse": "pi2",
    },
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 5, 5)
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

Ts, signals = zs.measure_t2echo(soc, soccfg, cfg)

# %%
t2e = zf.T2decay_analyze(Ts, signals)
t2e

# %%
filename = "t2echo"
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
        "res_pulse": "readout_dp1",
        "qub_pulse": "pi",
    },
    "relax_delay": 50.0,  # us
}

# %%
cfg = make_cfg(exp_cfg, shots=5000)


# %%
fid, threshold, angle, signals = zs.measure_fid_auto(soc, soccfg, cfg, plot=True)
print(f"Optimal fidelity after rotation = {fid:.1%}")

# %%
filename = "single_shot_dp1"
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
best_freq = readout_f1
best_pdr = max_res_gain
best_len = res_length
DefaultCfg.set_pulse(
    readout_fid={
        "style": res_style,
        "freq": best_freq,
        "gain": best_pdr,
        "phase": 0,
        "length": best_len,
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
            "length": best_len,
        },
        "qub_pulse": "pi",
    },
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(best_pdr - 500, best_pdr + 2000, step=500)
cfg = make_cfg(exp_cfg, shots=5000)

pdrs, fids = zs.qubit.scan_pdr_fid(soc, soccfg, cfg, instant_show=True)

max_id = np.argmax(fids)
best_pdr, best_fid = pdrs[max_id], fids[max_id]
plt.plot(pdrs, fids, marker="s")
plt.xlabel("Pulse gain")
plt.ylabel("Fidelity")
plt.title(f"Max fide = {max(fids)*100:.2f}%")
plt.axvline(best_pdr, label=f"max fide gain = {best_pdr}", ls="--")
plt.legend()

# %%
filename = "scan_gain"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "gain", "unit": "a. u.", "values": pdrs},
    z_info={"name": "Fidelity", "unit": "%", "values": fids},
    comment=make_comment(cfg, f"best gain = {best_pdr}, best fide = {best_fid:.1%}"),
    tag="SingleShot/pdr",
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
        },
        "qub_pulse": "pi",
    },
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(best_len / 2, 3 * best_len, 5)
cfg = make_cfg(exp_cfg, shots=5000)

# lens, fids = zs.scan_len_fid(soc, soccfg, cfg)
lens, fids = zs.qubit.scan_len_fid(soc, soccfg, cfg, instant_show=True)

max_id = np.argmax(fids)
best_len, best_fid = lens[max_id], fids[max_id]
plt.plot(lens, fids, marker="s")
plt.xlabel("Pulse length")
plt.ylabel("Fidelity")
plt.title(f"Max fide = {max(fids)*100:.2f}%")
plt.axvline(best_len, label=f"max fide length = {best_len}", ls="--")
plt.legend()

# %%
filename = "scan_len"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Pulse length", "unit": "us", "values": lens},
    z_info={"name": "Fidelity", "unit": "%", "values": fids},
    comment=make_comment(cfg, f"best length = {best_len}, best fide = {best_fid:.1%}"),
    tag="SingleShot/length",
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
            "length": best_len,
        },
        "qub_pulse": "pi",
    },
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(best_freq - 5, best_freq + 5, 5)
cfg = make_cfg(exp_cfg, shots=5000)

fpts, fids = zs.qubit.scan_freq_fid(soc, soccfg, cfg, instant_show=True)

max_id = np.argmax(fids)
best_freq, best_fid = fpts[max_id], fids[max_id]
plt.plot(fpts, fids, marker="s")
plt.xlabel("Frequency")
plt.ylabel("Fidelity")
plt.title(f"Max fide = {best_fid:.1%}")
plt.axvline(best_freq, label=f"max fide freq = {best_freq}", ls="--")
plt.legend()

# %%
filename = "scan_freq"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Pulse freq", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Fidelity", "unit": "%", "values": fids},
    comment=make_comment(cfg, f"best freq = {best_freq}, best fide = {best_fid:.1%}"),
    tag="SingleShot/freq",
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
            "length": best_len,
        },
        "qub_pulse": "pi",
    },
    "relax_delay": 50.0,  # us
}
cfg = make_cfg(exp_cfg, shots=5000)

fid, threshold, angle, signals = zs.measure_fid_auto(soc, soccfg, cfg, plot=True)
print(f"Optimal fidelity after rotation = {fid:.1%}")

# %%
filename = "single_shot_fid"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "shot", "unit": "point", "values": np.arange(cfg["shots"])},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    y_info={"name": "ge", "unit": "", "values": np.array([0, 1])},
    comment=make_comment(cfg, f"fide {fid:.1%}"),
    tag="SingleShot",
    server_ip=data_host,
)

# %%
DefaultCfg.set_pulse(
    readout_fid={
        "style": res_style,
        "freq": best_freq,
        "gain": best_pdr,
        "phase": -angle,
        "length": best_len,
        "threshold": threshold,
    },
)

# %% [markdown]
# # Dump Configurations

# %%
DefaultCfg.dump("default_cfg.yaml")
