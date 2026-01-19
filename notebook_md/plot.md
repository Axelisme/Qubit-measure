---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

```python
%load_ext autoreload
import os
from typing import Dict, List, cast, Tuple

import qutip as qt
import numpy as np
from numpy.typing import NDArray
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from scqubits.core.fluxonium import Fluxonium
from scqubits.core.oscillator import Oscillator
from scqubits.core.hilbert_space import HilbertSpace

%autoreload 2
from zcu_tools.notebook.persistance import load_result
from zcu_tools.notebook.analysis.mist.branch import round_to_nearest
from zcu_tools.table import MetaDict
from zcu_tools.simulate import mA2flx

from zcu_tools.simulate.fluxonium.branch.floquet import FloquetBranchAnalysis
```

```python
qub_name = "Q12_2D[5]/Q1"

result_dir = f"../result/{qub_name}"

os.makedirs(f"{result_dir}/image/branch_floquet", exist_ok=True)
os.makedirs(f"{result_dir}/web/branch_floquet", exist_ok=True)
os.makedirs(f"{result_dir}/data/branch_floquet", exist_ok=True)
```

```python
loadpath = f"{result_dir}/params.json"
_, params, mA_c, period, allows, data_dict = load_result(loadpath)

print(f"EJ: {params[0]:.3f} GHz, EC: {params[1]:.3f} GHz, EL: {params[2]:.3f} GHz")

if dispersive_cfg := data_dict.get("dispersive"):
    g = dispersive_cfg["g"]
    r_f = dispersive_cfg["r_f"]
    print(f"g: {g} GHz, r_f: {r_f} GHz")
elif "r_f" in allows:
    r_f = allows["r_f"]
    print(f"r_f: {r_f} GHz")


# g = 70e-3  # GHz
# rf_w = 6.1e-3  # GHz
```

```python
md = MetaDict(json_path=f"{result_dir}/meta_info.json", read_only=True)
```

```python
# flx = 0.8
flx = mA2flx(md.cur_A, md.mA_c, 2 * abs(md.mA_e - md.mA_c))
flx
```

# TwoTone Flux Dep

```python
from zcu_tools.notebook.persistance import load_spects
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flx

processed_spect_path = f"{result_dir}/data/fluxdep/spectrums.hdf5"
s_spects = load_spects(processed_spect_path)
mA_bound = (
    np.nanmin([np.nanmin(s["spectrum"]["mAs"]) for s in s_spects.values()]),
    np.nanmax([np.nanmax(s["spectrum"]["mAs"]) for s in s_spects.values()]),
)
fpt_bound = (
    np.nanmin([np.nanmin(s["points"]["fpts"]) for s in s_spects.values()]),
    np.nanmax([np.nanmax(s["points"]["fpts"]) for s in s_spects.values()]),
)
t_mAs = np.linspace(mA_bound[0], mA_bound[1], 200)
t_fpts = np.linspace(fpt_bound[0], fpt_bound[1], 200)
t_flxs = mA2flx(t_mAs, mA_c, period)

_, energies = calculate_energy_vs_flx(params, t_flxs, cutoff=40, evals_count=5)
```

```python
%matplotlib inline
from zcu_tools.notebook.analysis.fluxdep.processing import cast2real_and_norm
from zcu_tools.notebook.analysis.fluxdep.models import energy2transition

fig, ax = plt.subplots(figsize=(10, 5))


for name, spect in s_spects.items():
    signals = spect["spectrum"]["data"]
    flx_mask = np.any(~np.isnan(signals), axis=1)
    fpt_mask = np.any(~np.isnan(signals), axis=0)
    signals = signals[flx_mask, :][:, fpt_mask]
    values = spect["spectrum"]["mAs"][flx_mask]
    fpts = spect["spectrum"]["fpts"][fpt_mask]

    # Normalize data
    norm_signals = cast2real_and_norm(signals)

    # convert values to flxs
    flxs = mA2flx(values, spect["mA_c"], spect["period"])

    ax.imshow(
        norm_signals.T,
        aspect="auto",
        extent=(flxs[0], flxs[-1], fpts[0], fpts[-1]),
        cmap="RdBu_r",
        origin="lower",
    )

allows = {
    "transitions": [(0, 1), (0, 2), (1, 2), (1, 3)],
    "red side": [(0, 1)],
    "mirror": [(0, 2), (0, 3), (1, 3)],
    "r_f": 5.352,
    "sample_f": 9.584640 / 2,
    # "sample_f": 6.881280 / 2,
}

fs, labels = energy2transition(energies, allows)

for i, label in enumerate(labels):
    label = label.replace(" -> ", "-")
    label = label.replace("mirror", "image")
    ax.plot(t_flxs, fs[:, i], label=label, zorder=1, linewidth=1)
ax.legend(loc=(0.46, 0.09))

ax.axvline(flx, color="r", linestyle="--", linewidth=3, zorder=10)
ax.text(flx, 1.8, f"{flx:.2f}", fontsize=14, ha="center", va="center", color="r")

ax.set_xlabel(r"$\Phi_{ext}$", fontsize=14)
ax.set_ylabel(r"Frequency [MHz]", fontsize=14)
ax.set_xlim(t_flxs[0], t_flxs[-1])
ax.set_ylim(fpts[0], fpts[-1])


plt.show(fig)
fig.savefig("../post_images/twotone_fluxdep.png", dpi=300)
plt.close(fig)
```

# T1 & T2

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.time_domain import T1Exp

filepath = r"../Database/Q12_2D[5]/Q1/Q1_t1@-2.500mA_1.hdf5"

exp = T1Exp()
_ = exp.load(filepath)

*_, fig = exp.analyze()

ax = fig.get_axes()[0]
ax.axhline(-2, color="k", linestyle="--", linewidth=5)
ax.axhline(-12.1, color="k", linestyle="--", linewidth=5)

ax.set_ylabel("")
ax.set_yticks([])
ax.grid(False)


fig.set_size_inches(8, 6)


plt.show(fig)
fig.savefig("../post_images/t1.png", dpi=300)
plt.close(fig)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.time_domain import T2EchoExp

filepath = r"../Database/Q12_2D[5]/Q1/Q1_t2echo@-2.500mA_1.hdf5"

exp = T2EchoExp()
_ = exp.load(filepath)

*_, fig = exp.analyze(fit_method="fringe")

ax = fig.get_axes()[0]
ax.axhline(0.45, color="k", linestyle="--", linewidth=5)
ax.axhline(-0.85, color="k", linestyle="--", linewidth=5)

ax.set_ylabel("")
ax.set_yticks([])
ax.grid(False)

fig.set_size_inches(8, 6)


plt.show(fig)
fig.savefig("../post_images/t2echo.png", dpi=300)
plt.close(fig)
```

# Dispersive Shift

```python
%matplotlib inline
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from zcu_tools.experiment.v2.twotone.dispersive import DispersiveExp
from zcu_tools.utils.fitting.resonance import (
    fit_edelay,
    get_proper_model,
    normalize_signal,
    remove_edelay,
)

filepath = r"../Database/Q12_2D[5]/Q1/Q1_dispersive_shift_gain0.050@-2.600mA_3.hdf5"


fpts, signals = DispersiveExp().load(filepath)
g_signals, e_signals = signals[0, :], signals[1, :]
g_amps, e_amps = np.abs(g_signals), np.abs(e_signals)

g_edelay = fit_edelay(fpts, g_signals)
e_edelay = fit_edelay(fpts, e_signals)
edelay = 0.5 * (g_edelay + e_edelay)

model = get_proper_model(fpts, g_signals)
g_params = model.fit(fpts, g_signals, edelay=edelay)
e_params = model.fit(fpts, e_signals, edelay=edelay)

g_freq, g_kappa = g_params["freq"], g_params["kappa"]
e_freq, e_kappa = e_params["freq"], e_params["kappa"]

g_fit = np.abs(model.calc_signals(fpts, **g_params))  # type: ignore
e_fit = np.abs(model.calc_signals(fpts, **e_params))  # type: ignore

# Calculate dispersive shift and average linewidth
chi = abs(g_freq - e_freq) / 2  # dispersive shift χ/2π
avg_kappa = (g_kappa + e_kappa) / 2  # average linewidth κ/2π

fig = plt.figure(figsize=(8, 4))
spec = fig.add_gridspec(2, 3, wspace=0.2)
ax_main = fig.add_subplot(spec[:, :2])
ax_g = fig.add_subplot(spec[0, 2])
ax_e = fig.add_subplot(spec[1, 2])

fig.suptitle(f"Dispersive shift χ/2π = {chi:.3f} MHz, κ/2π = {avg_kappa:.1f} MHz")

g_label = r"$|0\rangle$"
e_label = r"$|1\rangle$"

# Plot data and fits
label_g = f"{g_label}: {g_freq:.1f} MHz"
label_e = f"{e_label}: {e_freq:.1f} MHz"
ax_main.scatter(fpts, g_amps, marker=".", c="b")
ax_main.scatter(fpts, e_amps, marker=".", c="r")
ax_main.plot(fpts, g_fit, "b-", alpha=0.7, label=label_g)
ax_main.plot(fpts, e_fit, "r-", alpha=0.7, label=label_e)

# Mark resonance frequencies
ax_main.set_xlabel("Frequency (MHz)", fontsize=14)
ax_main.set_ylabel("Signal Amplitude (a.u.)", fontsize=14)

ax_main.legend(loc="upper right")
ax_main.grid(True)


ax_main.axvline(md.r_f, color="k", ls="--", alpha=0.7)
fpt_idx = np.argmin(np.abs(fpts - md.r_f))


def _plot_circle_fit(
    ax: Axes, signals: NDArray, params_dict: dict, color: str, label: str
) -> None:
    rot_signals = remove_edelay(fpts, signals, edelay)
    norm_signals, norm_circle_params = normalize_signal(
        rot_signals, params_dict["circle_params"], params_dict["a0"]
    )
    norm_xc, norm_yc, norm_r0 = norm_circle_params

    ax.plot(
        norm_signals.real,
        norm_signals.imag,
        color=color,
        marker=".",
        markersize=1,
        label=label,
    )
    ax.add_patch(Circle((norm_xc, norm_yc), norm_r0, fill=False, color=color))
    ax.axhline(0, color="k")
    ax.scatter(
        [norm_signals[fpt_idx].real],
        [norm_signals[fpt_idx].imag],
        color=color,
        marker="*",
        s=50,
    )
    ax.set_aspect("equal")
    ax.grid(True)
    # ax.set_xlabel(r"$Re[S_{21}]$", fontsize=14)
    ax.set_ylabel(r"$Im[S_{21}]$", fontsize=14)
    ax.yaxis.set_label_position("right")
    ax.legend()


# Plot individual circle fit
_plot_circle_fit(ax_g, g_signals, dict(g_params), "b", g_label)
_plot_circle_fit(ax_e, e_signals, dict(e_params), "r", e_label)
ax_e.set_xlabel(r"$Re[S_{21}]$", fontsize=14)

ax_main.scatter([fpts[fpt_idx]], [g_amps[fpt_idx]], color="b", marker="*", s=150)
ax_main.scatter([fpts[fpt_idx]], [e_amps[fpt_idx]], color="r", marker="*", s=150)

fig.set_size_inches(8, 5)

plt.show(fig)
fig.savefig("../post_images/dispersive_shift.png", pad_inches=0.5, dpi=300)
plt.close(fig)
```

# AcStark

```python
%matplotlib inline
from zcu_tools.experiment.v2.twotone.ac_stark import AcStarkExp

filepath = r"../Database/Q12_2D[5]/Q1/Q1_ac_stark@-2.500mA_1.hdf5"

exp = AcStarkExp()
_ = exp.load(filepath)

*_, fig = exp.analyze(chi=chi, kappa=avg_kappa)
plt.show(fig)
fig.savefig("../post_images/ac_stark.png", dpi=300)
plt.close(fig)
```

# Confusion Matrix

```python
%matplotlib inline

from zcu_tools.experiment.v2.singleshot.ge import GE_Exp

filepath = r"../Database/Q12_2D[5]/Q1/SingleShot/Q1_singleshot_w_jpa_wo_reset_log_20251228_032946@-2.500mA_1.hdf5"

exp = GE_Exp()
_ = exp.load(filepath)

%matplotlib inline
_, pops, _, fig = exp.analyze(
    backend="center",
    length_ratio=exp.last_cfg["readout"]["ro_cfg"]["ro_length"] / md.t1_with_tone,  # type: ignore
    logscale=True,
    # align_t1=False,
)
plt.close(fig)

_, fig = exp.calc_confusion_matrix(
    md.g_center, md.e_center, md.singleshot_radius, init_pops=pops
)
plt.show(fig)
fig.savefig("../post_images/confusion_matrix.png", dpi=300)
plt.close(fig)
```

# Rabi


## Before correction

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.len_rabi import LenRabiExp

filepath = r"../Database/Q12_2D[5]/Q1/Q1_rabi_length_singleshot@-2.500mA_3.hdf5"


exp = LenRabiExp()
_ = exp.load(filepath)

fig = exp.analyze()
plt.show(fig)
fig.savefig("../post_images/len_rabi_before.png", dpi=300)
plt.close(fig)
```

## After correction

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.len_rabi import LenRabiExp

filepath = r"../Database/Q12_2D[5]/Q1/Q1_rabi_length_singleshot@-2.500mA_3.hdf5"


exp = LenRabiExp()
_ = exp.load(filepath)

fig = exp.analyze(confusion_matrix=md.confusion_matrix)
plt.show(fig)
fig.savefig("../post_images/len_rabi_after.png", dpi=300)
plt.close(fig)
```

# MIST

```python
mist_gains = [0.025, 0.05, 0.11, 0.133]
mist_ns = [md.ac_stark_coeff * gain**2 for gain in mist_gains]
print("Mist Photon Number:")
for n in mist_ns:
    print(f"{n:.0f}")
```

## Overnight

```python
%matplotlib inline
from zcu_tools.experiment.v2.overnight.singleshot.mist import MistOvernightAnalyzer

fig = plt.figure(figsize=(13, 4))
fig1, fig2, fig3 = fig.subfigures(1, 3)

analyzer = MistOvernightAnalyzer()

filepaths = [
    r"../Database/Q12_2D[5]/Q1/MIST_overnight/Q1_mist_overnight@-2.500mA_mist_g_short_g_populations_1.hdf5",
    r"../Database/Q12_2D[5]/Q1/MIST_overnight/Q1_mist_overnight@-2.500mA_mist_g_short_e_populations_1.hdf5",
]

_ = analyzer.load(filepaths)
analyzer.analyze(fig1, ac_coeff=md.ac_stark_coeff, confusion_matrix=md.confusion_matrix)

filepaths = [
    r"../Database/Q12_2D[5]/Q1/MIST_overnight/Q1_mist_overnight@-2.500mA_mist_e_short_g_populations_1.hdf5",
    r"../Database/Q12_2D[5]/Q1/MIST_overnight/Q1_mist_overnight@-2.500mA_mist_e_short_e_populations_1.hdf5",
]
_ = analyzer.load(filepaths)
analyzer.analyze(fig2, ac_coeff=md.ac_stark_coeff, confusion_matrix=md.confusion_matrix)

filepaths = [
    r"../Database/Q12_2D[5]/Q1/MIST_overnight/Q1_mist_overnight@-2.500mA_mist_steady_g_populations_1.hdf5",
    r"../Database/Q12_2D[5]/Q1/MIST_overnight/Q1_mist_overnight@-2.500mA_mist_steady_e_populations_1.hdf5",
]
_ = analyzer.load(filepaths)
analyzer.analyze(fig3, ac_coeff=md.ac_stark_coeff, confusion_matrix=md.confusion_matrix)

for fig_i in (fig1, fig2, fig3):
    ax = fig_i.get_axes()[0]
    for n in mist_ns:
        ax.axvline(n, color="k", linestyle="--", alpha=0.5, zorder=10)

plt.show(fig)
fig.savefig("../post_images/mist_overnight.png", dpi=300)
plt.close(fig)
```

# T1


## With Tone 1

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.t1 import T1WithToneExp
from zcu_tools.experiment.v2.singleshot.util import calc_populations
from zcu_tools.utils.fitting.multi_decay import (
    calc_lambdas,
    fit_with_vadality,
    fit_dual_with_vadality,
    fit_dual_transition_rates,
    fit_transition_rates,
)

filepaths = [
    r"../Database/Q12_2D[5]/Q1/T1/Q1_t1_with_tone_0.03_pop_corr@-2_initg_1.hdf5",
    r"../Database/Q12_2D[5]/Q1/T1/Q1_t1_with_tone_0.03_pop_corr@-2_inite_1.hdf5",
]

# exp = T1WithToneExp()
# _ = exp.load(filepaths)

# fig = exp.analyze()


lens, populations = T1WithToneExp().load(filepaths)

lens = lens[1:]
populations = populations[1:]
populations = calc_populations(populations)  # (N, 2, 3)

populations = populations @ np.linalg.inv(md.confusion_matrix)
populations = np.clip(populations, 0.0, 1.0)

populations1 = populations[:, 0]  # init in g
populations2 = populations[:, 1]  # init in e

# fit_dual_with_vadality(lens, populations1, populations2)

# rate, _, fit_pops1, fit_pops2, *_ = fit_dual_transition_rates(
#     lens, populations1, populations2
# )
# fit_with_vadality(lens, populations2)
rate, _, fit_pops2, *_ = fit_transition_rates(lens, populations2)

lambdas, _ = calc_lambdas(rate)

t1 = 1.0 / lambdas[2]
t1_b = 1.0 / lambdas[1]

fig, ax2 = plt.subplots(figsize=(8, 5))

ax2.set_title(
    r"$T_{1a}$" + f" = {t1:.1f} μs,  " + r"$T_{1b}$" + f" = {t1_b:.1f} μs", fontsize=14
)

ax2.plot(lens, fit_pops2[:, 0], color="blue", label=r"$|0\rangle$")
ax2.plot(lens, fit_pops2[:, 1], color="red", label=r"$|1\rangle$")
ax2.plot(lens, fit_pops2[:, 2], color="green", label=r"$|L\rangle$")
ax2.scatter(lens, populations2[:, 0], color="blue", s=5)
ax2.scatter(lens, populations2[:, 1], color="red", s=5)
ax2.scatter(lens, populations2[:, 2], color="green", s=5)
ax2.set_xlabel("Time (μs)", fontsize=14)
ax2.set_ylabel("Population", fontsize=14)
ax2.legend(loc="center right", fontsize=14)
ax2.grid(True)

fig.tight_layout()

plt.show(fig)
fig.savefig("../post_images/t1_with_tone_0.025.png", dpi=300)
plt.close(fig)
```

## With Tone

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.t1 import T1WithToneExp
from zcu_tools.experiment.v2.singleshot.util import calc_populations
from zcu_tools.utils.fitting.multi_decay import calc_lambdas, fit_dual_transition_rates

filepaths = [
    r"../Database/Q12_2D[5]/Q1/T1/Q1_t1_with_tone_0.13_pop_corr@-2_initg_1.hdf5",
    r"../Database/Q12_2D[5]/Q1/T1/Q1_t1_with_tone_0.13_pop_corr@-2_inite_1.hdf5",
]

lens, populations = T1WithToneExp().load(filepaths)

populations = calc_populations(populations)  # (N, 2, 3)

populations = populations @ np.linalg.inv(md.confusion_matrix)
populations = np.clip(populations, 0.0, 1.0)

populations1 = populations[:, 0]  # init in g
populations2 = populations[:, 1]  # init in e

# fit_dual_with_vadality(lens, populations1, populations2)

rate, _, fit_pops1, fit_pops2, *_ = fit_dual_transition_rates(
    lens, populations1, populations2
)

lambdas, _ = calc_lambdas(rate)

t1 = 1.0 / lambdas[2]
t1_b = 1.0 / lambdas[1]

fig, ax2 = plt.subplots(figsize=(8, 5))

ax2.set_title(
    r"$T_{1a}$" + f" = {t1:.1f} μs,  " + r"$T_{1b}$" + f" = {t1_b:.1f} μs", fontsize=14
)

ax2.plot(lens, fit_pops2[:, 0], color="blue", label=r"$|0\rangle$")
ax2.plot(lens, fit_pops2[:, 1], color="red", label=r"$|1\rangle$")
ax2.plot(lens, fit_pops2[:, 2], color="green", label=r"$|L\rangle$")
ax2.scatter(lens, populations2[:, 0], color="blue", s=5)
ax2.scatter(lens, populations2[:, 1], color="red", s=5)
ax2.scatter(lens, populations2[:, 2], color="green", s=5)
ax2.set_xlabel("Time (μs)", fontsize=14)
ax2.set_ylabel("Population", fontsize=14)
ax2.legend(loc="center right", fontsize=14)
ax2.grid(True)

fig.tight_layout()

plt.show(fig)
fig.savefig("../post_images/t1_with_tone_0.17.png", dpi=300)
plt.close(fig)
```

## Power dependence

```python
from zcu_tools.experiment.v2.singleshot.t1 import T1WithToneSweepExp
from zcu_tools.experiment.v2.singleshot.util import calc_populations
from zcu_tools.utils.fitting.multi_decay import fit_dual_transition_rates

filepaths = [
    r"../Database/Q12_2D[5]/Q1/T1/Q1_t1_with_tone_gain_pop_20251231_111407_corr@-2.500mA_gg_pop_1.hdf5",
    r"../Database/Q12_2D[5]/Q1/T1/Q1_t1_with_tone_gain_pop_20251231_111407_corr@-2.500mA_ge_pop_1.hdf5",
    r"../Database/Q12_2D[5]/Q1/T1/Q1_t1_with_tone_gain_pop_20251231_111407_corr@-2.500mA_eg_pop_1.hdf5",
    r"../Database/Q12_2D[5]/Q1/T1/Q1_t1_with_tone_gain_pop_20251231_111407_corr@-2.500mA_ee_pop_1.hdf5",
]


xs, Ts, populations = T1WithToneSweepExp().load(filepaths)


valid_mask = np.all(np.isfinite(populations), axis=(1, 2, 3))
xs = xs[valid_mask]
populations = populations[valid_mask]


populations = calc_populations(populations)  # (xs, 2, Ts, 3)

populations = populations @ np.linalg.inv(md.confusion_matrix)
populations = np.clip(populations, 0.0, 1.0)

# (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
N = populations.shape[0]
rates = np.zeros((N, 6), dtype=np.float64)
rate_Covs = np.zeros((N, 6, 6), dtype=np.float64)
for i, pop in enumerate(tqdm(populations, desc="Fitting transition rates")):
    rate, *_, (_, pCov1), _ = fit_dual_transition_rates(Ts, pop[0], pop[1])
    rates[i] = rate
    rate_Covs[i] = pCov1[:6, :6]

xs = md.ac_stark_coeff * xs**2
```

```python
%matplotlib inline
fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey="row")
ax_eg, ax_ee, ax_eo, ax_Tg, ax_Te, ax_To = axs.flatten()


def _plot_population(ax, pop, label) -> None:
    ax.scatter([], [], s=0)
    ax.imshow(
        pop.T,
        aspect="auto",
        cmap="RdBu_r",
        extent=(xs[0], xs[-1], Ts[-1], Ts[0]),
    )
    # Create annotation box at upper right with label content, like a legend
    ax.annotate(
        label,
        xy=(0.95, 0.5),
        xycoords="axes fraction",
        fontsize=12,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )


_plot_population(ax_eg, populations[:, 1, :, 0], r"$|0\rangle$")
_plot_population(ax_ee, populations[:, 1, :, 1], r"$|1\rangle$")
_plot_population(ax_eo, populations[:, 1, :, 2], r"$|L\rangle$")

# (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
R_go = rates[:, 4]
R_ge = rates[:, 0]
R_eo = rates[:, 2]
R_eg = rates[:, 1]
Rerr_go = np.sqrt(rate_Covs[:, 4, 4])
Rerr_ge = np.sqrt(rate_Covs[:, 0, 0])
Rerr_eo = np.sqrt(rate_Covs[:, 2, 2])
Rerr_eg = np.sqrt(rate_Covs[:, 1, 1])
for i in range(rates.shape[0]):
    if i % 5 == 0:
        continue
    Rerr_go[i] = np.nan
    Rerr_ge[i] = np.nan
    Rerr_eo[i] = np.nan
    Rerr_eg[i] = np.nan

ax_Tg.errorbar(xs, R_go, yerr=Rerr_go, label="$Γ_{0L}$", color="dodgerblue")
ax_Tg.errorbar(xs, R_ge, yerr=Rerr_ge, label="$Γ_{01}$", color="blue")

ax_Te.errorbar(xs, R_eo, yerr=Rerr_eo, label="$Γ_{1L}$", color="darkorange")
ax_Te.errorbar(xs, R_eg, yerr=Rerr_eg, label="$Γ_{10}$", color="red")

ax_To.errorbar(xs, R_eo, yerr=Rerr_eo, label="$Γ_{1L}$", color="darkorange")
ax_To.errorbar(xs, R_go, yerr=Rerr_go, label="$Γ_{0L}$", color="dodgerblue")

for ax in fig.get_axes():
    ax.set_xlabel("")
    ax.set_ylabel("")
    for gain in mist_gains:
        ax.axvline(
            md.ac_stark_coeff * gain**2, color="k", linestyle="--", alpha=0.5, zorder=10
        )

max_rate = np.nanmax([R_go, R_ge, R_eo, R_eg]).item()
for ax in (ax_Tg, ax_Te, ax_To):
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2 * max_rate)
    ax.set_xlabel(r"$\bar{n}$", fontsize=14)


ax_eg.set_ylabel("Time (μs)", fontsize=14)
ax_Tg.set_ylabel("Rate (μs⁻¹)", fontsize=14)


fig.tight_layout()

plt.show(fig)
fig.savefig("../post_images/t1_with_tone_gain.png", dpi=300)
plt.close(fig)
```

# AcStark SingleShot

```python
qub_dim = 30
qub_cutoff = 50
max_photon = 400

amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), 1e-3 * md.rf_w)
photons = (amps / (2 * g)) ** 2


def calc_energies(branchs: List[int]) -> Dict[int, np.ndarray]:
    avg_times = np.linspace(0.0, 2 * np.pi / r_f, 100)

    fb_analysis = FloquetBranchAnalysis(
        params, r_f, g, flx=flx, qub_dim=qub_dim, qub_cutoff=qub_cutoff
    )

    fbasis_n = Parallel(n_jobs=-1)(
        delayed(fb_analysis.make_floquet_basis)(photon, precompute=avg_times)
        for photon in tqdm(photons, desc="Computing Floquet basis")
    )
    fbasis_n = cast(List[qt.FloquetBasis], fbasis_n)

    branch_infos = fb_analysis.calc_branch_infos(fbasis_n, branchs)
    branch_energies = fb_analysis.calc_branch_energies(fbasis_n, branch_infos)

    return {k: np.asarray(v) for k, v in branch_energies.items()}


branchs = list(range(15))
branch_energies = calc_energies(branchs)

# fluxonium = Fluxonium(*params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim)
# E_n0 = fluxonium.eigenvals(evals_count=15)

resonator = Oscillator(r_f, truncated_dim=30)
fluxonium = Fluxonium(*params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim)
hilbertspace = HilbertSpace([fluxonium, resonator])
hilbertspace.add_interaction(
    g=g, op1=fluxonium.n_operator, op2=resonator.creation_operator, add_hc=True
)
E_n0 = hilbertspace.eigenvals(evals_count=qub_dim * 15)

hilbertspace.generate_lookup(ordering="LX")


def calc_E_n0_ij(i: int, j: int) -> np.ndarray:
    i_dressed = hilbertspace.dressed_index((i, 0))
    j_dressed = hilbertspace.dressed_index((j, 0))

    return E_n0[j_dressed] - E_n0[i_dressed]


transitions = {}
E_01 = branch_energies[1] - branch_energies[0]
# E_01 += (E_n0[1] - E_n0[0]) - E_01[0]
E_01 += calc_E_n0_ij(0, 1) - E_01[0]
for i in (0, 1):
    for j in range(i + 1, 15):
        E_ij = branch_energies[j] - branch_energies[i]
        # E_ij += (E_n0[j] - E_n0[i]) - E_ij[0]
        E_ij += calc_E_n0_ij(i, j) - E_ij[0]

        transitions[f"{i} → {j}"] = round_to_nearest(E_01, E_ij, r_f)
        transitions[f"{i} → {j} image"] = (
            2 * allows["sample_f"] - transitions[f"{i} → {j}"]
        )

        for k in [2, 3]:
            transitions[f"{i} → {j} ({k} photon)"] = E_ij / k
```

```python
transitions = {}
E_01 = branch_energies[1] - branch_energies[0]
# E_01 += (E_n0[1] - E_n0[0]) - E_01[0]
E_01 += calc_E_n0_ij(0, 1) - E_01[0]
for i in (0, 1):
    for j in range(i + 1, 15):
        E_ij = branch_energies[j] - branch_energies[i]
        # E_ij += (E_n0[j] - E_n0[i]) - E_ij[0]
        E_ij += calc_E_n0_ij(i, j) - E_ij[0]

        E_ij_mod = round_to_nearest(E_01, E_ij, r_f)
        E_ij_image = 2 * allows["sample_f"] - E_ij_mod
        transitions[f"{i} → {j}"] = E_ij_mod
        transitions[f"{i} → {j} image"] = E_ij_image

        if (i, j) == (0, 4) or (i, j) == (0, 7):
            mean_E_ij = np.mean(E_ij)
            mean_E_ij_mod = np.mean(E_ij_mod)
            mean_E_ij_image = np.mean(transitions[f"{i} → {j} image"])
            print((i, j), " : ")
            print(f"\tE_ij = {mean_E_ij:.2f} GHz")
            print(f"\tE_ij (mod) = {mean_E_ij_mod:.2f} GHz")
            print(f"\tE_ij (image) = {mean_E_ij_image:.2f} GHz")
            print(f"\tphoton: {(mean_E_ij - mean_E_ij_image) / r_f:.2f}")

        for k in [2, 3]:
            transitions[f"{i} → {j} ({k} photon)"] = E_ij / k
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.ac_stark import AcStarkExp

filepaths = [
    r"../Database/Q12_2D[5]/Q1/Q1_ac_stark_pop/Q1_ac_stark_pop@-2.500mA_g_pop_2.hdf5",
    r"../Database/Q12_2D[5]/Q1/Q1_ac_stark_pop/Q1_ac_stark_pop@-2.500mA_e_pop_2.hdf5",
]
exp = AcStarkExp()
_ = exp.load(filepaths)
fig = exp.analyze(ac_coeff=md.ac_stark_coeff)


ax1, ax2, ax3 = fig.get_axes()


line_positions = {
    "0 → 1": (100, 4350),
    "0 → 9": (80, 4150),
    "0 → 3 (2 photon)": (240, 3890),
    "0 → 4 image": (30, 4530),
    "0 → 7 image": (70, 4460),
}


def make_darker(color):
    return (
        "black"
        if color in ("w", "white")
        else (
            (
                lambda c: (
                    np.clip(np.array(to_rgb(c)) * 0.65, 0, 1)
                    if isinstance(c, str)
                    else c
                )
            )(color)
        )
    )


for ax in [ax1, ax2, ax3]:
    ylim = ax.get_ylim()
    for name, E_ij in transitions.items():
        E_ij = 1e3 * E_ij

        if name not in line_positions:
            continue

        mask = np.bitwise_and(E_ij > ylim[0], E_ij < ylim[1])
        if np.any(mask):
            (line,) = ax.plot(photons, E_ij)
            default_xy = (float(np.median(photons[mask])), float(np.median(E_ij[mask])))
            ax.annotate(
                name,
                xy=line_positions.get(name, default_xy),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=make_darker(line.get_color()),
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    alpha=0.9,
                    ec="none",
                ),
            )
    mirror_rf = 1e3 * (2 * allows["sample_f"] - r_f)
    ax.axhline(mirror_rf, color="red", linestyle="--", alpha=0.7)
    ax.annotate(
        "resonator image",
        xy=(230, mirror_rf - 20),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color="k",
    )

for ax in fig.get_axes():
    for n in mist_ns:
        ax.axvline(n, color="k", linestyle="--", alpha=0.5, zorder=10)

plt.show(fig)
fig.savefig("../post_images/ac_stark_populations.png", dpi=300)
plt.close(fig)
```

```python
%matplotlib inline
from zcu_tools.experiment.v2.singleshot.ac_stark import AcStarkExp
from matplotlib.image import NonUniformImage


filepaths = [
    r"../Database/Q12_2D[5]/Q1/Q1_ac_stark_pop/Q1_ac_stark_pop_gain0.300_20260104_150055@-2.500mA_g_pop_1.hdf5",
    r"../Database/Q12_2D[5]/Q1/Q1_ac_stark_pop/Q1_ac_stark_pop_gain0.300_20260104_150055@-2.500mA_e_pop_1.hdf5",
]

gains, freqs, populations = AcStarkExp().load(filepaths)


populations = calc_populations(populations)  # (xs, 2, Ts, 3)

populations = populations @ np.linalg.inv(md.confusion_matrix)
populations = np.clip(populations, 0.0, 1.0)


# plot the data and the fitted polynomial
meas_photons = md.ac_stark_coeff * gains**2

fig, ax1 = plt.subplots(figsize=(6, 6))

# Use NonUniformImage for better visualization with pdr^2 as x-axis
im = NonUniformImage(ax1, cmap="RdBu_r", interpolation="nearest")
im.set_data(meas_photons, freqs, populations[..., 0].T)
im.set_extent((meas_photons[0], meas_photons[-1], freqs[0], freqs[-1]))
ax1.add_image(im)
ax1.set_xlim(meas_photons[0], meas_photons[-1])
ax1.set_ylim(freqs[0], freqs[-1])
ax1.set_aspect("auto")


line_positions = {
    "0 → 1": (50, 4420),
    "0 → 4 image": (10, 4525),
    "0 → 7 image": (35, 4475),
}

for n in mist_ns:
    ax1.axvline(n, color="k", linestyle="--", linewidth=3, zorder=10)


ylim = ax1.get_ylim()
for name, E_ij in transitions.items():
    E_ij = 1e3 * E_ij

    if name not in line_positions:
        continue

    mask = np.bitwise_and(E_ij > ylim[0], E_ij < ylim[1])
    if np.any(mask):
        (line,) = ax1.plot(photons, E_ij)
        default_xy = (float(np.median(photons[mask])), float(np.median(E_ij[mask])))
        ax1.annotate(
            name,
            xy=line_positions.get(name, default_xy),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=20,
            color=make_darker(line.get_color()),
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                alpha=0.9,
                ec="none",
            ),
        )
mirror_rf = 1e3 * (2 * allows["sample_f"] - r_f)
ax1.axhline(mirror_rf, color="k", linestyle="--", alpha=0.5)
ax1.annotate(
    "resonator image",
    xy=(230, mirror_rf + 20),
    xytext=(5, 0),
    textcoords="offset points",
    va="center",
    fontsize=9,
    color="k",
)


ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.set_xticks([])
ax1.set_yticks([])

fig.tight_layout()

plt.show(fig)
fig.savefig("../post_images/ac_stark_populations_gain2.png", dpi=300)
plt.close(fig)
```

# Flux Dependence

```python
from zcu_tools.experiment.v2.twotone.fluxdep import FreqFluxDepExp
from zcu_tools.utils.process import minus_background

filepath = r"../Database/Q12_2D[5]/Q1/Q1_flux_tls_3.hdf5"

exp = FreqFluxDepExp()
values, freqs, signals2D = exp.load(filepath)
measure_flxs = mA2flx(values, md.mA_c, 2 * abs(md.mA_e - md.mA_c))


qub_dim = 30
qub_cutoff = 50

photon = md.ac_stark_coeff * exp.last_cfg["init_pulse"]["gain"] ** 2  # type: ignore


branchs = list(range(25))


EV_n0s = [
    Fluxonium(*params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim).eigensys(
        evals_count=qub_dim
    )
    for flx in tqdm(measure_flxs, desc="Calculating E_n0")
]
E_n0 = np.asarray([E for E, _ in EV_n0s])
V_n0 = np.asarray([V for _, V in EV_n0s])


def _calc_esys(
    flx: float, photon: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    fluxonium = Fluxonium(*params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim)
    esys = fluxonium.eigensys(evals_count=qub_dim)
    H = qt.Qobj(fluxonium.hamiltonian(energy_esys=esys))
    n_op = qt.Qobj(fluxonium.n_operator(energy_esys=esys))
    H_with_drive = [H, [n_op, lambda t, amp: amp * np.cos(r_f * t)]]

    basis = qt.FloquetBasis(
        H_with_drive, 2 * np.pi / r_f, args=dict(amp=2 * g * np.sqrt(photon))
    )

    return basis.e_quasi, basis.state(t=0)  # type: ignore


esys_over_flxs = Parallel(n_jobs=-1)(
    delayed(_calc_esys)(flx, photon)
    for flx in tqdm(measure_flxs, desc="Calculating ESys")
)
esys_over_flxs = list(esys_over_flxs)
esys_over_flxs = cast(
    List[Tuple[NDArray[np.float64], NDArray[np.float64]]], esys_over_flxs
)


branch_energies = {b: [] for b in branchs}
for b in branchs:
    V_n0_b = V_n0[b][..., None]

    for E, fstates in esys_over_flxs:
        last_state = qt.basis(qub_dim, b)
        dists = [np.abs(last_state.dag() @ fstate) for fstate in fstates]
        Eb = E[np.argmax(dists)]

        branch_energies[b].append(Eb)

branch_energies = {b: np.asarray(v) for b, v in branch_energies.items()}

esys_n0_over_flxs = Parallel(n_jobs=-1)(
    delayed(_calc_esys)(flx, 0) for flx in tqdm(measure_flxs, desc="Calculating ESys")
)
esys_n0_over_flxs = list(esys_n0_over_flxs)
esys_n0_over_flxs = cast(
    List[Tuple[NDArray[np.float64], NDArray[np.float64]]], esys_n0_over_flxs
)

branch_n0_energies = {b: [] for b in branchs}
for b in branchs:
    V_n0_b = V_n0[b][..., None]

    for E, fstates in esys_n0_over_flxs:
        last_state = qt.basis(qub_dim, b)
        dists = [np.abs(last_state.dag() @ fstate) for fstate in fstates]
        Eb = E[np.argmax(dists)]

        branch_n0_energies[b].append(Eb)

branch_n0_energies = {b: np.asarray(v) for b, v in branch_n0_energies.items()}

transitions = {}
E_01 = branch_energies[1] - branch_energies[0]
E_01 += (E_n0[:, 1] - E_n0[:, 0]) - (branch_n0_energies[1] - branch_n0_energies[0])
for i in (0, 1):
    for j in range(i + 1, max(*branchs)):
        E_ij = branch_energies[j] - branch_energies[i]
        E_ij += (E_n0[:, j] - E_n0[:, i]) - (
            branch_n0_energies[j] - branch_n0_energies[i]
        )

        transitions[f"{i} → {j}"] = round_to_nearest(E_01, E_ij, r_f)
        transitions[f"{i} → {j} image"] = (
            2 * allows["sample_f"] - transitions[f"{i} → {j}"]
        )

        for k in [2]:
            transitions[f"{i} → {j} ({k} photon)"] = E_ij / k
```

```python
%matplotlib inline
fig, ax = plt.subplots()

ax.imshow(
    np.abs(minus_background(signals2D, axis=1)).T,
    aspect="auto",
    extent=(measure_flxs[0], measure_flxs[-1], freqs[0], freqs[-1]),
    cmap="RdBu_r",
    origin="lower",
)
ax.set_xlabel(r"$\Phi_{ext}$", fontsize=14)
ax.set_ylabel(r"Frequency [MHz]", fontsize=14)
ax.set_xlim(measure_flxs[0], measure_flxs[-1])
ax.set_ylim(freqs[0], freqs[-1])

line_positions = {
    "0 → 1": (0.85, 4540),
    "0 → 4 image": (0.83, 4540),
    "0 → 7 image": (0.84, 4410),
}

ax.axvline(flx, color="k", linestyle="--", alpha=0.5, linewidth=3)

ylim = ax.get_ylim()
for name, E_ij in transitions.items():
    E_ij = 1e3 * E_ij

    if name not in line_positions:
        continue

    mask = np.bitwise_and(E_ij > ylim[0], E_ij < ylim[1])
    if np.any(mask):
        (line,) = ax.plot(measure_flxs, E_ij)
        default_xy = (
            float(np.median(measure_flxs[mask])),
            float(np.median(E_ij[mask])),
        )
        ax.annotate(
            name,
            xy=line_positions.get(name, default_xy),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=15,
            color=make_darker(line.get_color()),
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                alpha=0.9,
                ec="none",
            ),
        )


plt.show(fig)
fig.savefig("../post_images/tls_flux_dep.png", dpi=300)
plt.close(fig)
```

```python

```
