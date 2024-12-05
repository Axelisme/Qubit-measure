import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm

from zcu_tools.program import AmpRabiProgram, TwoToneProgram


def measure_lenrabi(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)

    sweep_cfg = cfg["sweep"]
    lens = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])
    signals = np.zeros(len(lens), dtype=np.complex128)

    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        fig, ax = plt.subplots()
        ax.set_xlabel("Length (ns)")
        ax.set_ylabel("Signal (a.u.)")
        ax.set_title("Length-dependent measurement")
        curve = ax.plot(lens, np.abs(signals))[0]
        dh = display(fig, display_id=True)

    qub_pulse = cfg["qub_pulse"]

    for i, length in enumerate(lens):
        qub_pulse["length"] = length
        prog = TwoToneProgram(soccfg, cfg)
        avgi, avgq = prog.acquire(soc, progress=False)
        signals[i] = avgi[0][0] + 1j * avgq[0][0]

        if instant_show:
            amps = np.ma.masked_equal(np.abs(signals), 0)
            curve.set_ydata(amps)
            ax.relim()
            ax.autoscale_view()
            dh.update(fig)

    if instant_show:
        clear_output()

    return lens, signals


def measure_amprabi(soc, soccfg, cfg, instant_show=False, soft_loop=False):
    cfg = deepcopy(cfg)

    if soft_loop:
        print("Use TwoToneProgram for soft loop")

        sweep_cfg = cfg["sweep"]
        pdrs = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])

        qub_pulse = cfg["qub_pulse"]

        signals = np.zeros(len(pdrs), dtype=np.complex128)

        if instant_show:
            import matplotlib.pyplot as plt
            from IPython.display import clear_output, display

            fig, ax = plt.subplots()
            ax.set_xlabel("Power (a.u.)")
            ax.set_ylabel("Signal (a.u.)")
            ax.set_title("Power-dependent measurement")
            curve = ax.plot(pdrs, np.abs(signals))[0]
            dh = display(fig, display_id=True)

        for i, pdr in enumerate(tqdm(pdrs)):
            qub_pulse["gain"] = pdr
            prog = TwoToneProgram(soccfg, cfg)
            avgi, avgq = prog.acquire(soc, progress=False)
            signals[i] = avgi[0][0] + 1j * avgq[0][0]

            if instant_show:
                amps = np.ma.masked_equal(np.abs(signals), 0)
                curve.set_ydata(amps)
                ax.relim()
                ax.autoscale_view()
                dh.update(fig)

        if instant_show:
            clear_output()

    else:
        print("Use AmpRabiProgram for hard loop")

        if instant_show:
            print("Instant show is not supported for hard loop, ignored.")

        prog = AmpRabiProgram(soccfg, cfg)
        pdrs, avgi, avgq = prog.acquire(soc, progress=True)
        signals = avgi[0][0] + 1j * avgq[0][0]

    return pdrs, signals
