import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from resonator_tools import circuit


def load_data(file_path):
    with h5py.File(file_path, "r") as file:
        data = file["Data"]["Data"]
        if data.shape[2] == 1:  # 1D data,
            x_data = data[:, 0, 0][:]
            y_data = None
            z_data = data[:, 1, 0][:] + 1j * data[:, 2, 0][:]
        else:
            x_data = data[:, 0, 0][:]
            y_data = data[0, 1, :][:]
            z_data = data[:, 2, :][:] + 1j * data[:, 3, :][:]
    return z_data, x_data, y_data


filepath = "data/res/res_freq_r5_g300.hdf5"
signals, fpts, _ = load_data(filepath)
signals = signals.conj()
amps = abs(signals)

# plot the data
ratio1, ratio2 = 0.42, 0.62

fmin, fmax = fpts.min(), fpts.max()
flen = fmax - fmin
plot_range = (fmin + ratio1 * flen, fmin + ratio2 * flen)
plot_range = fpts.min(), fpts.max()


port1 = circuit.notch_port(fpts, signals)
# port1.autofit(f_range=(6.00e9, 6.05e9))
port1.GUIfit()
# save the fit results
savepath = filepath.replace("data", "result").replace(".hdf5", "")
os.makedirs(os.path.dirname(savepath), exist_ok=True)
# port1.plotall(savepath=savepath + ".png", plot_range=plot_range)
port1.plotall(plot_range=plot_range)
fitresults = port1.fitresults.copy()
for key in fitresults:
    if hasattr(fitresults[key], "item"):
        fitresults[key] = fitresults[key].item()
with open(savepath + ".json", "w") as file:
    json.dump(fitresults, file, indent=4)

fr = port1.fitresults["fr"] / 1e9
Ql = port1.fitresults["Ql"]
kappa = fr / Ql

# plot the fit results
fig, ax = plt.subplots()
ax.plot(fpts / 1e9, amps, "-o", markersize=2)

ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Amplitude (a.u.)")
ax.set_xlim(plot_range[0] / 1e9, plot_range[1] / 1e9)

ax.plot(fpts / 1e9, np.abs(port1.z_data_sim))
ax.axvline(fr, color="red", linestyle="--", label=f"fr = {fr:.3f} GHz")
ax.axvline(
    (fr - 0.5 * kappa),
    color="green",
    linestyle="--",
    label=f"kappa = {kappa*1e3:.1f} MHz",
)
ax.axvline((fr + 0.5 * kappa), color="green", linestyle="--")
ax.legend()
# fig.savefig(savepath + "_fit.png")


plt.show()
