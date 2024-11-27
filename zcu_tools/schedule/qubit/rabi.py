from copy import deepcopy

from zcu_tools.program import AmpRabiProgram


def measure_amprabi(soc, soccfg, cfg):
    cfg = deepcopy(cfg)
    cfg["qub_pulse"]["gain"] = cfg["sweep"]["start"]
    prog = AmpRabiProgram(soccfg, cfg)
    pdrs, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return pdrs, signals
