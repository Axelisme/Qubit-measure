from numpy.typing import NDArray

from zcu_tools.program import AmplitudeRabiProgram


def measure_amprabi(soc, soccfg, cfg) -> tuple[NDArray, NDArray]:
    prog = AmplitudeRabiProgram(soccfg, cfg)
    pdrs, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return pdrs, signals
