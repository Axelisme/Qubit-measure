from copy import deepcopy

from zcu_tools.program import AmpRabiProgram, LenRabiProgram


def measure_lenrabi(soc, soccfg, cfg):
    cfg = deepcopy(cfg)
    prog = LenRabiProgram(soccfg, cfg)
    Ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]
    Ts = prog.cycles2us(Ts, gen_ch=cfg["qubit"]["qub_ch"])

    return Ts, signals


def measure_amprabi(soc, soccfg, cfg):
    cfg = deepcopy(cfg)
    prog = AmpRabiProgram(soccfg, cfg)
    pdrs, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return pdrs, signals
