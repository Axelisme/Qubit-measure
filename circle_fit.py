import json
import os

import h5py

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


filepath = "res_freq_test2_1.hdf5"
signals, fpts, _ = load_data(os.path.join("data", filepath))
signals = signals.conj()

port1 = circuit.notch_port(fpts, signals)
# port1.autofit(f_range=(6.00e9, 6.05e9))
port1.GUIfit()


# save the fit results
savepath = os.path.join("result", filepath.split(".")[0])
os.makedirs(os.path.dirname(savepath), exist_ok=True)
port1.plotall(savepath=savepath + ".png")
fitresults = port1.fitresults.copy()
for key in fitresults:
    if hasattr(fitresults[key], "item"):
        fitresults[key] = fitresults[key].item()
with open(savepath + ".json", "w") as file:
    json.dump(fitresults, file, indent=4)
