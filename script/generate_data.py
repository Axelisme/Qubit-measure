# make energies of fluxonium under different external fluxes
# and save them in a file

import os

import h5py as h5
import numpy as np
import scqubits as scq
from tqdm.auto import tqdm

# parameters
data_path = "simulation_data/fluxonium_int3.h5"
num_per = 20
EJb = (2.0, 6.0)
ECb = (0.8, 2.0)
ELb = (0.01, 0.2)
# EJb = (3.0, 6.5)
# ECb = (0.3, 2.0)
# ELb = (0.5, 3.5)

DRY_RUN = False
scq.settings.PROGRESSBAR_DISABLED = True

level_num = 20
max_level = 15
cutoff = 50
flxs = np.linspace(0.0, 0.5, 120)


fluxonium = scq.Fluxonium(1.0, 1.0, 1.0, flux=0.0, cutoff=cutoff)


def calculate_spectrum(flxs, EJ, EC, EL):
    global fluxonium
    fluxonium.EJ = EJ
    fluxonium.EC = EC
    fluxonium.EL = EL
    spectrumData = fluxonium.get_spectrum_vs_paramvals(
        "flux", flxs, evals_count=max_level
    )

    return spectrumData.energy_table


def dump_data(filepath, flxs, params, energies, Ebounds):
    with h5.File(filepath, "w") as f:
        f.create_dataset("Ebounds", data=Ebounds)
        f.create_dataset("flxs", data=flxs)
        f.create_dataset("params", data=params)
        f.create_dataset("energies", data=energies)


EJc = (EJb[0] + EJb[1]) / 2
ECc = (ECb[0] + ECb[1]) / 2
ELc = (ELb[0] + ELb[1]) / 2

params = []
energies = []
print("Calculating on EC-EL plane")
for EC in tqdm(np.linspace(ECb[0], ECb[1], num_per + 1)):
    for EL in tqdm(np.linspace(ELb[0], ELb[1], num_per + 1)):
        if DRY_RUN:
            energy = np.random.randn(len(flxs), level_num)
        else:
            energy = calculate_spectrum(flxs, EJc, EC, EL)

        # since energy is proportional to EJ, we can just use the energy
        for EJ in np.linspace(EJb[0], EJb[1], num_per + 1):
            ratio = EJ / EJc
            tEC = EC * ratio
            tEL = EL * ratio

            params.append((EJ, tEC, tEL))
            energies.append(energy * ratio)

print("Calculating on EJ-EL plane")
for EJ in tqdm(np.linspace(EJb[0], EJb[1], num_per + 1)):
    for EL in tqdm(np.linspace(ELb[0], ELb[1], num_per + 1)):
        if DRY_RUN:
            energy = np.random.randn(len(flxs), level_num)
        else:
            energy = calculate_spectrum(flxs, EJ, ECc, EL)

        for EC in np.linspace(ECb[0], ECb[1], num_per + 1):
            ratio = EC / ECc
            tEJ = EJ * ratio
            tEL = EL * ratio

            params.append((tEJ, EC, tEL))
            energies.append(energy * ratio)

print("Calculating on EJ-EC plane")
for EJ in tqdm(np.linspace(EJb[0], EJb[1], num_per + 1)):
    for EC in tqdm(np.linspace(ECb[0], ECb[1], num_per + 1)):
        if DRY_RUN:
            energy = np.random.randn(len(flxs), level_num)
        else:
            energy = calculate_spectrum(flxs, EJ, EC, ELc)

        for EL in np.linspace(ELb[0], ELb[1], num_per + 1):
            ratio = EL / ELc
            tEJ = EJ * ratio
            tEC = EC * ratio

            params.append((tEJ, tEC, EL))
            energies.append(energy * ratio)

print("Total data points:", len(params))

scq.settings.PROGRESSBAR_DISABLED = False

# we can flip the data around 0.5 to make the other half
# since the fluxonium is symmetric
flxs = np.concatenate([flxs, 1.0 - flxs[::-1]])
for i in range(len(params)):
    energies[i] = np.concatenate([energies[i], energies[i][::-1]])

params = np.array(params)
energies = np.array(energies)
Ebounds = np.array((EJb, ECb, ELb))

os.makedirs(os.path.dirname(data_path), exist_ok=True)
dump_data(data_path, flxs, params, energies, Ebounds)
