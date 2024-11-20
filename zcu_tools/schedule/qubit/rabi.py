from copy import deepcopy

from zcu_tools.program import AmpRabiProgram


def measure_amprabi(soc, soccfg, cfg):
    prog = AmpRabiProgram(soccfg, deepcopy(cfg))
    pdrs, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return pdrs, signals
