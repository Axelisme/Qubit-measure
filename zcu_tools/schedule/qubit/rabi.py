from copy import deepcopy

from zcu_tools.program import AmplitudeRabiProgram


def measure_amprabi(soc, soccfg, cfg):
    prog = AmplitudeRabiProgram(soccfg, deepcopy(cfg))
    pdrs, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return pdrs, signals
