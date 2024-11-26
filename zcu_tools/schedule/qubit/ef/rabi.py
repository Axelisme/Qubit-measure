from copy import deepcopy

from zcu_tools.program.ef import EFAmpRabiProgram


def measure_ef_amprabi(soc, soccfg, cfg):
    prog = EFAmpRabiProgram(soccfg, deepcopy(cfg))
    pdrs, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return pdrs, signals
