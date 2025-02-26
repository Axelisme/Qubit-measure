from typing import Any, Callable, Dict

from numpy import ndarray
from tqdm.auto import tqdm


from qick.averager_program import AveragerProgram, RAveragerProgram
from zcu_tools.schedule.instant_show import InstantShow
from zcu_tools.schedule.flux import set_flux


def sweep1D_hard_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: type,
    *,
    init_xs: ndarray,
    init_signals: ndarray,
    progress: bool,
    instant_show: bool,
    xlabel: str,
    ylabel: str,
    **kwargs,
):
    xs = init_xs.copy()
    signals = init_signals.copy()
    if instant_show:
        viewer = InstantShow(xs, x_label=xlabel, y_label=ylabel)

        def callback(ir, sum_d):
            nonlocal signals
            signals = sum_d[0][0].dot([1, 1j]) / (ir + 1)
            viewer.update_show(signals)
    else:
        callback = None

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog: RAveragerProgram = prog_cls(soccfg, cfg)
    try:
        xs, avgi, avgq = prog.acquire(
            soc, progress=progress, callback=callback, **kwargs
        )
        signals: ndarray = avgi[0][0] + 1j * avgq[0][0]
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            viewer.update_show(signals, xs)
            viewer.close_show()

    return xs, signals


def sweep1D_soft_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: type,
    *,
    xs: ndarray,
    init_signals: ndarray,
    progress: bool,
    instant_show: bool,
    updateCfg: Callable,
    xlabel: str,
    ylabel: str,
    **kwargs,
):
    signals = init_signals.copy()
    if instant_show:
        viewer = InstantShow(xs, x_label=xlabel, y_label=ylabel)
        show_period = int(len(xs[0]) / 20 + 0.99)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog: AveragerProgram = prog_cls(soccfg, cfg)
    try:
        for i, x in enumerate(tqdm(xs, desc=xlabel, smoothing=0, disable=not progress)):
            updateCfg(cfg, i, x)

            set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

            avgi, avgq = prog.acquire(soc, progress=False, **kwargs)
            signals[i] = avgi[0][0] + 1j * avgq[0][0]

            if instant_show and i % show_period == 0:
                viewer.update_show(signals)

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            viewer.update_show(signals)
            viewer.close_show()

    return xs, signals
