# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
import matplotlib.pyplot as plt
import numpy as np

# %autoreload 2
from zcu_tools import load_data
from zcu_tools.analysis.fluxdep import (
    InteractiveLines,
    InteractiveSelector,
    calculate_energy,
    energy2transition,
    fit_spectrum,
    preprocess_data,
    search_in_database,
    spectrum_analyze,
)

# %% [markdown]
# # load data

# %%
spectrum, fpts, flxs = load_data("data/qub/qub_flux_dep_2.hdf5")
flxs, fpts, spectrum = preprocess_data(flxs, fpts, spectrum)

s_spectrum = spectrum.copy()
s_spectrum = s_spectrum - np.median(s_spectrum, axis=0, keepdims=True)
s_spectrum = s_spectrum - np.median(s_spectrum, axis=1, keepdims=True)
s_spectrum = np.abs(s_spectrum)

s_spectrum /= np.std(s_spectrum, axis=0)

# %%
# %matplotlib widget
if "cflx" not in locals():
    cflx = None
if "eflx" not in locals():
    eflx = None

actLine = InteractiveLines(s_spectrum, flxs, fpts, cflx, eflx)

# %%
cflx, eflx = actLine.get_positions()
halfp = eflx - cflx
cflx, eflx, halfp

# %%
flxs = (flxs - cflx) / (2 * halfp) + 0.5
cflx, eflx = 0.5, 1.0

# %%
# %matplotlib widget
s_flxs, s_fpts = spectrum_analyze(flxs, fpts, spectrum, 4.0)

if "colors" not in locals():
    colors = None

actSel = InteractiveSelector(s_spectrum, flxs, fpts, s_flxs, s_fpts, colors)

# %%
s_flxs, s_fpts, colors = actSel.get_selected_points()

# %%
allows = {
    "transitions": [(0, 1), (0, 2)],
    "sideband": [],
    "mirror": [(0, 1), (0, 2), (0, 3)],
    "cavity_f": 6.02808,
    "sample_f": 6.88128,
}

# %%
best_params, _, _ = search_in_database(
    s_flxs, s_fpts, "database/fluxonium_1.h5", allows
)
print(best_params)


# %%
f_energies = calculate_energy(flxs, *best_params, cutoff=40)

# %%
# %matplotlib inline
plt.imshow(
    s_spectrum,
    aspect="auto",
    origin="lower",
    extent=(flxs[0], flxs[-1], fpts[0], fpts[-1]),
)


fs, labels = energy2transition(f_energies, allows)
for i, label in enumerate(labels):
    plt.plot(flxs, fs[:, i], label=label)

plt.scatter(s_flxs, s_fpts, c="r", s=3)
plt.ylim(fpts[0], fpts[-1])
plt.xlim(flxs[0], flxs[-1])
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()
plt.close()

# %% [markdown]
# # Scipy

# %%
EJb = (2.0, 10.0)
ECb = (0.5, 3.0)
ELb = (0.1, 2.0)
# EJb = (1.0, 3.0)
# ECb = (0.5, 2.5)
# ELb = (0.1, 0.5)

# %%
# fit the spectrumData
sp_params = fit_spectrum(s_flxs, s_fpts, best_params, allows, (EJb, ECb, ELb))
# sp_params = fit_spectrum(s_flxs, s_fpts, sp_params, allows, (EJb, ECb, ELb))

# print the results
print("Fitted params:", *sp_params)

# %%
# sp_params = (8.51795441, 0.9041685, 1.09072694)
# f_energies = calculate_energy(flxs, *sp_params, 60)
f_energies = calculate_energy(flxs, 8.51, 0.904, 1.09, 60)

# %%
# %matplotlib inline
plt.imshow(
    s_spectrum,
    aspect="auto",
    origin="lower",
    extent=(flxs[0], flxs[-1], fpts[0], fpts[-1]),
)

fs, labels = energy2transition(f_energies, allows)
for i, label in enumerate(labels):
    plt.plot(flxs, fs[:, i], label=label)

plt.scatter(s_flxs, s_fpts, color="red", s=3)
plt.axhline(6.881280, label="sample freq")
# plt.axhline(6.881280*2, label="sample freq * 2")
plt.ylim(fpts[0], fpts[-1])
# plt.title(f"EJ/EC/EL = {sp_params[0]:.2f}/{sp_params[1]:.2f}/{sp_params[2]:.2f} GHz")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel("Flux (Phi/Phi0)")
plt.ylabel("Frequency (GHz)")
plt.show()

savepath = "result/2DQ9_Q5"
# plt.savefig(savepath)

plt.close()

# %%
# dump the data
with open(savepath + ".txt", "w") as file:
    file.write(f"Transition: {allows['transitions']}\n")
    file.write(f"Sideband: {allows['sideband']}\n")
    file.write(f"Mirror: {allows['mirror']}\n")
    file.write(f"Half flux: {cflx}\n")
    file.write(f"Integrer flux: {halfp+cflx}\n")
    file.write(f"EJ/EC/EL: {sp_params}\n")

# %%
# get some transition

flx = 0.5
wants = [(0, 1), (1, 2)]

flx_idx = np.argmin(np.abs(flxs - flx))
for want in wants:
    print(
        f"{want}: {(f_energies[flx_idx, want[1]] - f_energies[flx_idx, want[0]])*1e3:.3f} MHz"
    )

# %%
