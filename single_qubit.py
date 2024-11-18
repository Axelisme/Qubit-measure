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
import zcu_tools.program as zp  # noqa: E402
import zcu_tools.schedule as zs  # noqa: E402

# ruff: noqa: I001
from zcu_tools import DefaultCfg, create_datafolder, make_cfg, make_sweep, save_data  # noqa: E402

# %% [markdown]
# # Connect to zcu216
# %%
from qick.pyro import make_proxy

# ns_host = "140.114.82.71"
ns_host = "100.101.250.4"  # tailscale
ns_port = 8887
proxy_name = "myqick"

soc, soccfg = make_proxy(ns_host=ns_host, ns_port=ns_port, proxy_name=proxy_name)
print(soccfg)


# %%
from qick import QickSoc  # noqa: E402

soc = QickSoc()
soccfg = soc
print(soc)

# %% [markdown]
# # Utility functions


# %%
def reload_zcutools():
    import importlib
    from types import ModuleType

    excluded = ["qick", "numpy", "matplotlib.pyplot", "importlib"]
    visited = set()

    def reload(module, depth=1, level=1):
        if level > depth:
            return

        nonlocal visited
        if module in visited:
            return
        visited.add(module)

        print(" " * level + module.__name__)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, ModuleType) and attr.__name__ not in excluded:
                reload(attr, depth, level + 1)

        importlib.reload(module)

    print("reloaded:")
    reload(zp, 2)
    reload(zf, 2)
    reload(zs, 3)


# %% [markdown]
# # Create data folder

# %%
data_root = os.path.join(os.getcwd(), "Database")
database_path = create_datafolder(data_root)

# %% [markdown]
# # Predefine parameters

# %%
res_name = "r1"
qubit_name = "q1"
flux_method = "zcu216"

DefaultCfg.init_global(
    res_cfgs={res_name: {"res_ch": 0, "ro_chs": [0], "nqz": 2}},
    qub_cfgs={qubit_name: {"qub_ch": 2, "nqz": 2}},
    flux_cfgs={
        "yokogawa": {
            "name": "gs200",
            "address": "USB::0x0B21::0x0039::91S522309::INSTR",
        },
        "zcu216": {"ch": 4},
    },
)


# %% [markdown]
# # Lookback

# %%
exp_cfg = {
    "resonator": res_name,
    "res_pulse": {
        "style": "const",
        "freq": 5700,  # MHz
        "phase": 0,
        "gain": 5000,
        "length": 0.2,  # us
    },
    "flux": {"method": flux_method, "value": 0},
    "adc_trig_offset": 0.518,  # us
    "relax_delay": 0.0,  # us
}


# %%
cfg = make_cfg(exp_cfg, reps=10, rounds=2)

Is, Qs = zs.measure_lookback(soc, soccfg, cfg)

# Plot results.
plt.figure(1)
plt.plot(Is, label="I value")
plt.plot(Qs, label="Q value")
plt.plot(np.abs(Is + 1j * Qs), label="mag")
plt.ylabel("a.u.")
plt.xlabel("Clock ticks")
plt.title("Averages = " + str(cfg["rounds"]))

# %%
adc_trig_offset = exp_cfg["adc_trig_offset"]
DefaultCfg.set_res(res_name, adc_trig_offset=adc_trig_offset)

filename = "lookback"
ts = soc.cycles2us(1) * np.arange(len(Is))
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": ts * 1e-6},
    z_info={"name": "Signal", "unit": "a.u.", "values": Is + 1j * Qs},
    comment=f"adc_trig_offset = {adc_trig_offset}us",
    tag="Lookback",
)


# %% [markdown]
# # Resonator Frequency

# %%
res_style = "const"
ro_length = 1
exp_cfg = {
    "resonator": res_name,
    "res_pulse": {
        "style": res_style,
        "phase": 0,
        "gain": 5000,
        "length": ro_length,  # us
    },
    "flux": {"method": flux_method, "value": 0},
    "relax_delay": 3.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(5700, 6150, 5)
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

fpts, signals = zs.measure_res_freq(soc, soccfg, cfg)

plt.plot(fpts, np.abs(signals))


# %%
guess_r = 5990

exp_cfg["sweep"] = make_sweep(guess_r - 20, guess_r + 20, 5)
cfg = make_cfg(exp_cfg, res_pulse={"gain": 1000}, reps=2000, rounds=1)

fpts, signals1 = zs.measure_res_freq(soc, soccfg, cfg)

# %%
r_f, _ = zf.spectrum_analyze(fpts, signals1)
r_f

# %%
cfg = make_cfg(exp_cfg, res_pulse={"gain": 30000}, reps=2000, rounds=1)

fpts, signals2 = zs.measure_res_freq(soc, soccfg, cfg)

# %%
r_lf, _ = zf.spectrum_analyze(fpts, signals2)
r_lf

# %%
r_f - r_lf

# %%
# r_f = 5000
# r_lf = 5001

# %%
filename = "res_freq 1k gain"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals1},
    comment=f"resonator frequency = {r_f}MHz",
    tag="OneTone",
)

filename = "res_freq 30k gain"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2},
    comment=f"resonator frequency = {r_lf}MHz",
    tag="OneTone",
)

# %% [markdown]
# # Onetone Dependences

# %% [markdown]
# ## Power dependence

# %%
exp_cfg = {
    "resonator": res_name,
    "res_pulse": {
        "style": res_style,
        "phase": 0,
        "freq": r_f,
        "gain": 1000,
        "length": ro_length,
    },
    "flux": {"method": flux_method, "value": 0},
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
exp_cfg["sweep"] = {
    "pdr": make_sweep(500, 7500, step=1000),
    "freq": make_sweep(r_f - 10, r_f + 10, 5),
}
cfg = make_cfg(exp_cfg, reps=1000, rounds=1)

fpts, pdrs, signals2D = zs.measure_power_dependent(soc, soccfg, cfg)

plt.figure()
plt.pcolormesh(fpts, pdrs, NormalizeData(np.abs(signals2D)))

# %%
res_gain = 2200

filename = "res_power_dependence"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Power", "unit": "a.u.", "values": pdrs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment="power dependence",
    tag="OneTone",
)

# %% [markdown]
# ## Flux dependence

# %%
exp_cfg = {
    "resonator": res_name,
    "res_pulse": {
        "style": res_style,
        "phase": 0,
        "freq": r_f,
        "gain": res_gain,
        "length": ro_length,
    },
    "flux": {"method": flux_method},
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
sw_spot = 10000

DefaultCfg.set_qub(qubit_name, sw_spot={flux_method: sw_spot})

filename = "res_flux_dependence"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    y_info={"name": "Flux", "unit": "a.u.", "values": flxs},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
    comment="",
    tag="OneTone",
)

# %% [markdown]
# ## Set Readout pulse

# %%
DefaultCfg.set_res_pulse(
    res_name,
    readout_rf={
        "style": res_style,
        "freq": r_f,
        "phase": 0,
        "gain": res_gain,
        "length": ro_length,
        "desc": "Readout with resonance frequency",
    },
)

# %% [markdown]
# # Qubit Frequency

# %%
qub_style = "cosine"
exp_cfg = {
    "resonator": res_name,
    "res_pulse": "readout_rf",
    "qubit": qubit_name,
    "qub_pulse": {
        "style": qub_style,
        "gain": 5000,
        "phase": 0,
        "length": 4,
    },
    "flux": {"method": flux_method, "value": sw_spot},
    "relax_delay": 5.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(3200, 5000, 5)
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

fpts, signals = zs.measure_qubit_freq(soc, soccfg, cfg)
amps = np.abs(signals)
angle = np.angle(signals)

plt.figure()
plt.plot(fpts, amps / np.mean(amps), label="amp")
plt.plot(fpts, angle / np.mean(angle), label="angle")
plt.legend()

print("amps: ", fpts[np.argmax(np.abs(amps - np.mean(amps)))])
print("angle: ", fpts[np.argmax(np.abs(angle - np.mean(angle)))])


# %%
quess_q = 4658
exp_cfg["sweep"] = make_sweep(quess_q - 25, quess_q + 25, 5)
cfg = make_cfg(exp_cfg, reps=8, rounds=1)

fpts, signals = zs.measure_qubit_freq(soc, soccfg, cfg)

# %%
f_amp, f_pha = zf.spectrum_analyze(fpts, signals)
f_amp, f_pha

# %%
qub_freq = f_amp
# qub_freq = f_pha
qub_freq

# %%
# qub_freq = 4900

# %%
filename = "qub_freq"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    comment=f"qubit frequency = {qub_freq}MHz",
    tag="TwoTone",
)

# %% [markdown]
# # Amplitude Rabi

# %%
qub_pulse_len = 0.07
exp_cfg = {
    "resonator": res_name,
    "res_pulse": "readout_rf",
    "qubit": qubit_name,
    "qub_pulse": {
        "style": qub_style,
        "freq": qub_freq,
        "phase": 0,
        "length": qub_pulse_len,
    },
    "flux": {"method": flux_method, "value": sw_spot},
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
    comment=f"pi gain = {pi_gain}, pi/2 gain = {pi2_gain}",
    tag="TimeDomain",
)

# %% [markdown]
# ## Set Pi / Pi2 Pulse

# %%
DefaultCfg.set_qub_pulse(
    qubit_name,
    pi={
        "style": qub_style,
        "freq": qub_freq,
        "gain": pi_gain,
        "phase": 0,
        "length": qub_pulse_len,
        "desc": "pi pulse",
    },
    pi2={
        "style": qub_style,
        "freq": qub_freq,
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
    "resonator": res_name,
    "res_pulse": "readout_rf",
    "qubit": qubit_name,
    "qub_pulse": "pi",
    "flux": {"method": flux_method, "value": sw_spot},
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
filename = "dispersive shift"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
    z_info={
        "name": "Signal",
        "unit": "a.u.",
        "values": np.array((g_signals, e_signals)),
    },
    y_info={"name": "ge", "unit": "", "values": np.array([0, 1])},
    comment=f"SNR1 = {readout_f1}MHz, SNR2 = {readout_f2}MHz",
    tag="Dispersive",
)


# %% [markdown]
# ## Set Dispersive readout

# %%
DefaultCfg.set_res_pulse(
    res_name,
    readout_dp1={
        "style": res_style,
        "freq": readout_f1,
        "gain": res_gain,
        "phase": 0,
        "length": ro_length,
        "desc": "Readout with largest dispersive shift",
    },
    readout_dp2={
        "style": res_style,
        "freq": readout_f2,
        "gain": res_gain,
        "phase": 0,
        "length": ro_length,
        "desc": "Readout with second largest dispersive shift",
    },
)

# %% [markdown]
# # T2Ramsey

# %%
activate_detune = 3.0
exp_cfg = {
    "resonator": res_name,
    "res_pulse": "readout_dp1",
    "qubit": qubit_name,
    "qub_pulse": {
        "style": qub_style,
        "freq": qub_freq + activate_detune,
        "phase": 0,
        "gain": pi2_gain,
        "length": qub_pulse_len,
    },
    "flux": {"method": flux_method, "value": sw_spot},
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(0, 0.5, 5)  # us
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

Ts1, signals1 = zs.measure_t2ramsey(soc, soccfg, cfg)

t2f, detune = zf.T2fringe_analyze(soc.cycles2us(Ts1), signals1)
detune


# %%
exp_cfg["sweep"] = make_sweep(0, 0.5, 5)
cfg = make_cfg(exp_cfg, reps=2000, rounds=1)

Ts2, signals2 = zs.measure_t2ramsey(soc, soccfg, cfg)


# %%
t2d = zf.T2decay_analyze(soc.cycles2us(Ts2), signals2)
t2d

# %%
qub_freq = qub_freq + activate_detune - detune

filename = "t2ramsey"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "Time", "unit": "s", "values": soc.cycles2us(Ts2)},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals2},
    comment=f"activate detune = {activate_detune}MHz, detune = {detune}MHz",
    tag="TimeDomain",
)


# %% [markdown]
# # T1

# %%
exp_cfg = {
    "resonator": res_name,
    "res_pulse": "readout_dp1",
    "qubit": qubit_name,
    "qub_pulse": "pi",
    "flux": {"method": flux_method, "value": sw_spot},
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
    comment=f"t1 = {t1}us",
    tag="TimeDomain",
)

# %% [markdown]
# # T2Echo

# %%
exp_cfg = {
    "resonator": res_name,
    "res_pulse": "readout_dp1",
    "qubit": qubit_name,
    "qub_pulse": [("pi", "pi"), ("pi2", "pi2")],
    "flux": {"method": flux_method, "value": sw_spot},
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
    comment=f"t2echo = {t2e}us",
    tag="TimeDomain",
)

# %% [markdown]
# # Single shot

# %%
exp_cfg = {
    "resonator": res_name,
    "res_pulse": "readout_dp1",
    "qubit": qubit_name,
    "qub_pulse": "pi",
    "flux": {"method": flux_method, "value": sw_spot},
    "relax_delay": 50.0,  # us
}

# %%
cfg = make_cfg(exp_cfg, shots=5000)


# %%
fid, threshold, angle, signals = zs.measure_fid(
    soc, soccfg, cfg, plot=True, verbose=True
)
print("Optimal fidelity after rotation = %.3f" % fid)

# %% [markdown]
# ## Tuning single shot readout

# %%
# initial parameters
best_style = res_style
best_freq = readout_f1
best_pdr = res_gain
best_ro_len = ro_length
DefaultCfg.set_res_pulse(
    res_name,
    readout_fid={
        "style": best_style,
        "freq": best_freq,
        "gain": best_pdr,
        "phase": 0,
        "length": best_ro_len,
        "desc": "Readout for best fidelity",
    },
)

# %% [markdown]
# ### Scan readout style

# %%
exp_cfg = {
    "resonator": res_name,
    "res_pulse": "readout_fid",
    "qubit": qubit_name,
    "qub_pulse": "pi",
    "flux": {"method": flux_method, "value": sw_spot},
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = ["const", "gauss", "cosine"]
cfg = make_cfg(exp_cfg, shots=5000)

fids = zs.scan_style_fid(soc, soccfg, cfg)

# sort by fid, where fids is a dict
fids = dict(sorted(fids.items(), key=lambda x: x[1]))
for style, fid in fids.items():
    print(f"Style: {style}, FID: {fid}")

best_style = list(fids.keys())[0]

# %% [markdown]
# ### Scan readout power

# %%
exp_cfg = {
    "resonator": res_name,
    "res_pulse": "readout_fid",
    "qubit": qubit_name,
    "qub_pulse": "pi",
    "flux": {"method": flux_method, "value": sw_spot},
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(res_gain - 500, res_gain + 2000, step=500)
cfg = make_cfg(exp_cfg, shots=5000)

fpts, fids = zs.scan_pdr_fid(soc, soccfg, cfg)

best_pdr = fpts[np.argmax(fids)]
plt.plot(fpts, fids, marker="s")
plt.xlabel("Pulse gain")
plt.ylabel("Fidelity")
plt.title(f"Max fide = {max(fids)*100:.2f}%")
plt.axvline(best_pdr, label=f"max fide gain = {best_pdr}", ls="--")
plt.legend()

# %% [markdown]
# ### Scan readout length

# %%
exp_cfg = {
    "resonator": res_name,
    "res_pulse": "readout_fid",
    "qubit": qubit_name,
    "qub_pulse": "pi",
    "flux": {"method": flux_method, "value": sw_spot},
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(ro_length / 2, 3 * ro_length, 5)
cfg = make_cfg(exp_cfg, shots=5000)

fpts, fids = zs.scan_len_fid(soc, soccfg, cfg)

best_len = fpts[np.argmax(fids)]
plt.plot(fpts, fids, marker="s")
plt.xlabel("Pulse length")
plt.ylabel("Fidelity")
plt.title(f"Max fide = {max(fids)*100:.2f}%")
plt.axvline(best_len, label=f"max fide length = {best_len}", ls="--")
plt.legend()

# %% [markdown]
# ### Scan readout frequency


# %%
exp_cfg = {
    "resonator": res_name,
    "res_pulse": "readout_fid",
    "qubit": qubit_name,
    "qub_pulse": "pi",
    "flux": {"method": flux_method, "value": sw_spot},
    "relax_delay": 50.0,  # us
}

# %%
exp_cfg["sweep"] = make_sweep(r_f - 5, r_f + 5, 5)
cfg = make_cfg(exp_cfg, shots=5000)

fpts, fids = zs.scan_freq_fid(soc, soccfg, cfg)

best_freq = fpts[np.argmax(fids)]
plt.plot(fpts, fids, marker="s")
plt.xlabel("Frequency")
plt.ylabel("Fidelity")
plt.title(f"Max fide = {max(fids)*100:.2f}%")
plt.axvline(best_freq, label=f"max fide freq = {best_freq}", ls="--")
plt.legend()

# %% [markdown]
# ### Set best readout pulse

# %%
exp_cfg = {
    "resonator": res_name,
    "res_pulse": {
        "style": res_style,
        "freq": best_freq,
        "gain": best_pdr,
        "phase": 0,
        "length": best_len,
    },
    "qubit": qubit_name,
    "qub_pulse": "pi",
    "flux": {"method": flux_method, "value": sw_spot},
    "relax_delay": 50.0,  # us
}
cfg = make_cfg(exp_cfg, shots=5000)

fid, threshold, angle, signals = zs.measure_fid(
    soc, soccfg, cfg, plot=True, verbose=True
)
print("Optimal fidelity after rotation = %.3f" % fid)

# %%
filename = "single_shot"
save_data(
    filepath=os.path.join(database_path, filename),
    x_info={"name": "shot", "unit": "point", "values": np.arange(cfg["shots"])},
    z_info={"name": "Signal", "unit": "a.u.", "values": signals},
    y_info={"name": "ge", "unit": "", "values": np.array([0, 1])},
    comment=f"fide {fid:.3f}",
    tag="SingleShot",
)

# %%
DefaultCfg.set_res_pulse(
    res_name,
    readout_fid={
        "style": res_style,
        "freq": best_freq,
        "gain": best_pdr,
        "phase": -angle,
        "length": best_len,
        "threshold": threshold,
        "desc": "Readout with optimal fidelity",
    },
    overwrite=True,
)

# %% [markdown]
# # Dump Configurations

# %%
DefaultCfg.dump()

# %%
