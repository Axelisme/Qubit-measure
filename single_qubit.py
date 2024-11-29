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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# # %cd /home/xilinx/jupyter_notebooks/nthu/sinica-5q/Axel/Qubit-measure
print(os.getcwd())
sys.path.append(os.getcwd())

import zcu_tools.analysis as zf  # noqa: E402
import zcu_tools.schedule as zs  # noqa: E402

# ruff: noqa: I001
from zcu_tools import (  # noqa: E402
    # reload_zcutools,
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
# # Create data folder

# %%
database_path = create_datafolder(os.getcwd(), "Axel")

# data_host = "192.168.10.252"  # cmd-> ipconfig -> ipv4 #controling computer
data_host = "100.76.229.37"  # tailscale
# data_host = None

# %% [markdown]
# # Predefine parameters

# %%
res_name = "res"
qubit_name = "qub"
# flux_dev = "zcu216"
flux_dev = "labber_yoko"
# flux_dev = "qcodes_yoko"

# flux_host = data_host
flux_host = "127.0.0.1"

DefaultCfg.init_global(
    res_cfgs={res_name: {"res_ch": 0, "ro_chs": [0], "nqz": 2}},
    qub_cfgs={qubit_name: {"qub_ch": 2, "nqz": 2}},
    flux_cfgs={
        "zcu216": {
            "ch": 4,
            "saturate": 0.1,  # us
        },
        "labber_yoko": {
            "server_ip": flux_host,
            "sHardware": "Yokogawa GS200 DC Source",
            "dev_cfg": {"address": "0x0B21::0x0039::91WB18861", "interface": "USB"},
            "flux_cfg": {"Current - Sweep rate": 10e-6},
        },
    },
    # overwrite=True,
)
DefaultCfg.set_default(resonator=res_name, flux_dev=flux_dev)

# %% [markdown]
# # Lookback

# %%
res_length = 1
ro_length = 1
exp_cfg = {
    "res_pulse": {
        # "style": "const",
        # "style": "cosine",
        # "style": "gauss",
        "style": "flat_top",
        "raise_pulse": {
            "style": "cosine",
            # "style": "gauss",
            "length": 0.1 * res_length,  # us
            # "sigma": 0.25,  # us
        },
        "freq": 5000,  # MHz
        "phase": 0,
        "gain": 5000,
        "length": res_length,  # us
        # "sigma": 0.25,  # us
    },
    "readout_length": ro_length,  # us
    "adc_trig_offset": 0,  # us
    "relax_delay": 10.0,  # us
}


# %%
cfg = make_cfg(exp_cfg, rounds=10)

Is, Qs = zs.measure_lookback(soc, soccfg, cfg)

Ts = soc.cycles2us(1) * np.arange(len(Is))
predict_offset = zf.lookback_analyze(Ts, Is, Qs)

# Plot results.
plt.figure()
plt.plot(Ts, Is, label="I value")
plt.plot(Ts, Qs, label="Q value")
plt.plot(Ts, np.abs(Is + 1j * Qs), label="mag")
plt.axvline(predict_offset, color="r", linestyle="--", label="predict_offset")
plt.ylabel("a.u.")
plt.xlabel("us")
plt.legend()
plt.title("Averages = " + str(cfg["rounds"]))

# %%
# adc_trig_offset = cfg["adc_trig_offset"]
adc_trig_offset = float(predict_offset) + cfg["adc_trig_offset"]
adc_trig_offset

# %%
DefaultCfg.set_res(res_name, adc_trig_offset=adc_trig_offset)
DefaultCfg.set_default(adc_trig_offset=adc_trig_offset, readout_length=ro_length)

filename = "lookback"
ts = soc.cycles2us(1) * np.arange(len(Is))
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": Is + 1j * Qs},
    comment=make_comment(cfg, f"adc_trig_offset = {adc_trig_offset}us"),
    tag="Lookback",
    server_ip=data_host,
)


# %% [markdown]
# # Resonator Frequency

# %%
# res_style = "const"
res_style = "flat_top"
exp_cfg = {
    "res_pulse": {
        "style": res_style,
        "phase": 0,
        "gain": 5000,
        "length": res_length,  # us
    },
    "relax_delay": 3.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(5700, 6150, 11)
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

fpts, signals = zs.measure_res_freq(soc, soccfg, cfg)

plt.plot(fpts, np.abs(signals))

# print five lowest frequencies
print(fpts[np.argsort(np.diff(np.abs(signals)))[:5]])


# %%
guess_r = 5990

exp_cfg["sweep"] = make_sweep(guess_r - 20, guess_r + 20, 5)
cfg = make_cfg(exp_cfg, res_pulse={"gain": 1000}, reps=2000, rounds=1)

fpts, signals = zs.measure_res_freq(soc, soccfg, cfg)

# %%
r_f, _ = zf.spectrum_analyze(fpts, signals)
r_f

# %%
# r_f = 5000
DefaultCfg.set_res(res_name, freq=r_f)

# %%
filename = "res_freq"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"resonator frequency = {r_f}MHz"),
    tag="OneTone",
    server_ip=data_host,
)

# %% [markdown]
# # Onetone Dependences

# %%
DefaultCfg.set_res_pulse(
    res_name,
    readout_rf={
        "style": res_style,
        "freq": r_f,
        "phase": 0,
        "gain": 5000,
        "length": res_length,  # us
        "desc": "Readout with resonance frequency",
    },
)
DefaultCfg.set_default(res_pulse="readout_rf")

# %% [markdown]
# ## Power dependence

# %%
exp_cfg = {
    "relax_delay": 3.0,  # us
}

# %%
exp_cfg["sweep"] = {
    "pdr": make_sweep(500, 30000, 5, force_int=True),
    "freq": make_sweep(r_f - 20, r_f + 20, 5),
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=1)

fpts, pdrs, signals2D = zs.measure_power_dependent(soc, soccfg, cfg)


# %%
def NormalizeData(signals2D: np.ndarray) -> np.ndarray:
    # normalize on frequency axis
    mins = np.min(signals2D, axis=1, keepdims=True)
    maxs = np.max(signals2D, axis=1, keepdims=True)
    return (signals2D - mins) / (maxs - mins)


plt.figure()
plt.pcolormesh(fpts, pdrs, NormalizeData(np.abs(signals2D)))


# %%
filename = "res_power_dep"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Power", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg),
    tag="OneTone",
    server_ip=data_host,
)

# %% [markdown]
# ## Update Readout pulse

# %%
max_gain = 2200
DefaultCfg.set_res_pulse(res_name, readout_rf={"gain": max_gain})
DefaultCfg.set_res(res_name, max_gain=max_gain)

# %% [markdown]
# ## Flux dependence

# %%
exp_cfg = {
    "relax_delay": 3.0,  # us
}

# %%
exp_cfg["sweep"] = {
    "flux": make_sweep(-30000, 30000, step=10000),
    "freq": make_sweep(r_f - 3, r_f + 3, 5),
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=1)

fpts, flxs, signals2D = zs.measure_flux_dependent(soc, soccfg, cfg)

# %%
plt.figure()
plt.pcolormesh(fpts, flxs, np.abs(signals2D))


# %%
filename = "res_flux_dep"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Flux", "unit": "a.u.", "values": flxs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment=make_comment(cfg),
    tag="OneTone",
    server_ip=data_host,
)

# %%
sw_spot = 10000

DefaultCfg.set_qub(qubit_name, sw_spot={flux_dev: sw_spot})
DefaultCfg.set_default(flux="sw_spot")

# %% [markdown]
# # Qubit Frequency

# %%
DefaultCfg.set_default(qubit=qubit_name)

# %%
qub_style = "cosine"
exp_cfg = {
    "res_pulse": "readout_rf",
    "qub_pulse": {
        "style": qub_style,
        "gain": 5000,
        "phase": 0,
        "length": 4,
    },
    "relax_delay": 5.0,  # us
}

# %%
quess_q = 4658
exp_cfg["sweep"] = make_sweep(quess_q - 25, quess_q + 25, 5)
cfg = make_cfg(exp_cfg, reps=8, rounds=1)

fpts, signals = zs.measure_qubit_freq(soc, soccfg, cfg)

# %%
f_amp, f_pha = zf.spectrum_analyze(fpts, signals)
f_amp, f_pha

# %%
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
    tag="TwoTone",
    server_ip=data_host,
)

# %%
# qub_freq = 4900
DefaultCfg.set_qub(qubit_name, freq=q_f)

# %% [markdown]
# # Amplitude Rabi

# %%
qub_pulse_len = 0.07
exp_cfg = {
    "res_pulse": "readout_rf",
    "qub_pulse": {
        "style": qub_style,
        "freq": q_f,
        "phase": 0,
        "length": qub_pulse_len,
    },
    "relax_delay": 70.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 30000, step=5000)
cfg = make_cfg(exp_cfg, reps=500, rounds=1)

pdrs, signals = zs.measure_amprabi(soc, soccfg, cfg)

# %%
pi_gain, pi2_gain, _ = zf.amprabi_analyze(pdrs, signals)
pi_gain = int(pi_gain + 0.5)
pi2_gain = int(pi2_gain + 0.5)
pi_gain, pi2_gain

# %%
# pi_gain = 10000
# pi2_gain = 5000

# %%
filename = "amp_rabi"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Amplitude", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"pi gain = {pi_gain}\npi/2 gain = {pi2_gain}"),
    tag="TimeDomain",
    server_ip=data_host,
)

# %% [markdown]
# ## Set Pi / Pi2 Pulse

# %%
DefaultCfg.set_qub_pulse(
    qubit_name,
    pi={
        "style": qub_style,
        "freq": q_f,
        "gain": pi_gain,
        "phase": 0,
        "length": qub_pulse_len,
        "desc": "pi pulse",
    },
    pi2={
        "style": qub_style,
        "freq": q_f,
        "gain": pi2_gain,
        "phase": 0,
        "length": qub_pulse_len,
        "desc": "pi/2 pulse",
    },
)

# %% [markdown]
# # Dispersive shift


# %%
exp_cfg = {
    "res_pulse": "readout_rf",
    "qub_pulse": "pi",
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(r_f - 5, r_f + 5, 5)
cfg = make_cfg(exp_cfg, reps=1000, rounds=1)

fpts, g_signals, e_signals = zs.measure_dispersive(soc, soccfg, cfg)

# %%
readout_f1, readout_f2 = zf.dispersive_analyze(fpts, g_signals, e_signals, asym=False)
readout_f1, readout_f2

# %%
# readout_f1 = 5700

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
    tag="Dispersive",
    server_ip=data_host,
)


# %% [markdown]
# ## Set Dispersive readout

# %%
DefaultCfg.set_res_pulse(
    res_name,
    readout_dp1={
        **DefaultCfg.get_res_pulse(res_name, "readout_rf"),
        "freq": readout_f1,
        "desc": "Readout with largest dispersive shift",
    },
    readout_dp2={
        **DefaultCfg.get_res_pulse(res_name, "readout_rf"),
        "freq": readout_f2,
        "desc": "Readout with second largest dispersive shift",
    },
)

# %% [markdown]
# # T2Ramsey

# %%
activate_detune = 3.0
orig_q_f = q_f
exp_cfg = {
    "res_pulse": "readout_dp1",
    "qub_pulse": {
        **DefaultCfg.get_qub_pulse(qubit_name, "pi2"),
        "freq": orig_q_f + activate_detune,
    },
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 0.5, 5)  # us
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

Ts, signals = zs.measure_t2ramsey(soc, soccfg, cfg)

# %%
t2f, detune = zf.T2fringe_analyze(soc.cycles2us(Ts), signals)
t2d = zf.T2decay_analyze(soc.cycles2us(Ts), signals)
detune, t2f, t2d

# %%
filename = "t2ramsey"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": soc.cycles2us(Ts)},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"detune = {detune}MHz\nt2f = {t2f}us\nt2d = {t2d}us"),
    tag="TimeDomain",
    server_ip=data_host,
)

# %%
q_f = orig_q_f + activate_detune - detune
DefaultCfg.set_qub(qubit_name, freq=q_f)


# %% [markdown]
# # T1

# %%
exp_cfg = {
    "res_pulse": "readout_dp1",
    "qub_pulse": "pi",
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 5, 5)
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

Ts, signals = zs.measure_t1(soc, soccfg, cfg)

# %%
skip_points = 0

t1 = zf.T1_analyze(soc.cycles2us(Ts[skip_points:]), signals[skip_points:])
t1

# %%
filename = "t1"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": soc.cycles2us(Ts)},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"t1 = {t1}us"),
    tag="TimeDomain",
    server_ip=data_host,
)

# %% [markdown]
# # T2Echo

# %%
exp_cfg = {
    "res_pulse": "readout_dp1",
    "qub_pulse": [("pi", "pi"), ("pi2", "pi2")],
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 5, 5)
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

Ts, signals = zs.measure_t2echo(soc, soccfg, cfg)

t2e = zf.T2decay_analyze(soc.cycles2us(Ts * 2), signals)
t2e

# %%
filename = "t2echo"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": soc.cycles2us(Ts * 2)},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=make_comment(cfg, f"t2echo = {t2e}us"),
    tag="TimeDomain",
    server_ip=data_host,
)

# %% [markdown]
# # Single shot

# %%
exp_cfg = {
    "res_pulse": "readout_dp1",
    "qub_pulse": "pi",
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
best_style = res_style
best_freq = readout_f1
best_pdr = max_gain
best_ro_len = res_length
DefaultCfg.set_res_pulse(
    res_name,
    readout_fid={
        **DefaultCfg.get_res_pulse(res_name, "readout_dp1"),
        "style": best_style,
        "freq": best_freq,
        "gain": best_pdr,
        "length": best_ro_len,
    },
)

# %% [markdown]
# ### Scan readout style

# %%
exp_cfg = {
    "res_pulse": "readout_fid",
    "qub_pulse": "pi",
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = ["const", "gauss", "cosine", "flat_top"]
cfg = make_cfg(exp_cfg, shots=5000)

fids = zs.scan_style_fid(soc, soccfg, cfg)

# sort by fid, where fids is a dict
fid_dict = dict(sorted(fids.items(), key=lambda x: x[1], reverse=True))
for style, fid in fid_dict.items():
    print(f"Style: {style}, FID: {fid}")

best_style, best_fid = list(fid_dict.items())[0]


# %%
filename = "scan_style"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "shot", "unit": "style", "values": np.arange(len(fids))},
    z_info={"name": "Fidelity", "unit": "%", "values": np.array(list(fids.values()))},
    comment=make_comment(cfg, f"best style = {best_style}, best fide = {best_fid:.1%}"),
    tag="SingleShot",
    server_ip=data_host,
)

# %% [markdown]
# ### Scan readout power

# %%
exp_cfg = {
    "res_pulse": "readout_fid",
    "qub_pulse": "pi",
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(max_gain - 500, max_gain + 2000, step=500)
cfg = make_cfg(exp_cfg, shots=5000)

pdrs, fids = zs.scan_pdr_fid(soc, soccfg, cfg)

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
    tag="SingleShot",
    server_ip=data_host,
)

# %% [markdown]
# ### Scan readout length

# %%
exp_cfg = {
    "res_pulse": "readout_fid",
    "qub_pulse": "pi",
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(res_length / 2, 3 * res_length, 5)
cfg = make_cfg(exp_cfg, shots=5000)

lens, fids = zs.scan_len_fid(soc, soccfg, cfg)

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
    tag="SingleShot",
    server_ip=data_host,
)

# %% [markdown]
# ### Scan readout frequency

# %%
exp_cfg = {
    "res_pulse": "readout_fid",
    "qub_pulse": "pi",
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(r_f - 5, r_f + 5, 5)
cfg = make_cfg(exp_cfg, shots=5000)

fpts, fids = zs.scan_freq_fid(soc, soccfg, cfg)

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
    tag="SingleShot",
    server_ip=data_host,
)

# %% [markdown]
# ### Set best readout pulse

# %%
exp_cfg = {
    "res_pulse": {
        "style": best_style,
        "freq": best_freq,
        "gain": best_pdr,
        "phase": 0,
        "length": best_len,
    },
    "qub_pulse": "pi",
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
DefaultCfg.set_res_pulse(
    res_name,
    readout_fid={
        "style": best_style,
        "freq": best_freq,
        "gain": best_pdr,
        "phase": -angle,
        "length": best_len,
        "threshold": threshold,
        "desc": "Readout with optimal fidelity",
    },
)

# %% [markdown]
# # Dump Configurations

# %%
DefaultCfg.dump(os.path.join(database_path, "default_cfg"))
