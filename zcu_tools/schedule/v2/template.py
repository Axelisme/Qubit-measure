from typing import Any, Callable, Dict, Tuple

from numpy import ndarray
from tqdm.auto import tqdm

from qick.asm_v2 import AveragerProgramV2
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow


def sweep_hard_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: type,
    *,
    init_signals: ndarray,
    ticks: Tuple[ndarray, ...],
    progress: bool,
    instant_show: bool,
    xlabel: str,
    ylabel: str,
    **kwargs,
):
    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    signals = init_signals.copy()
    if instant_show:
        viewer = InstantShow(*ticks, x_label=xlabel, y_label=ylabel)

        def callback(ir, sum_d):
            nonlocal signals
            signals = sum_d[0][0].dot([1, 1j]) / (ir + 1)  # type: ignore
            viewer.update_show(signals)
    else:
        callback = None  # type: ignore

    prog: AveragerProgramV2 = prog_cls(soccfg, cfg)
    try:
        IQlist = prog.acquire(soc, progress=progress, round_callback=callback, **kwargs)
        signals: ndarray = IQlist[0][0].dot([1, 1j])  # type: ignore
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            viewer.update_show(signals)
            viewer.close_show()

    return prog, signals


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
    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    signals = init_signals.copy()
    if instant_show:
        viewer = InstantShow(xs, x_label=xlabel, y_label=ylabel)
        show_period = int(len(xs[0]) / 20 + 0.99)

    prog: AveragerProgramV2 = prog_cls(soccfg, cfg)
    try:
        for i, x in enumerate(tqdm(xs, desc=xlabel, smoothing=0, disable=not progress)):
            updateCfg(cfg, i, x)

            # set again in case of change
            set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

            IQlist = prog.acquire(soc, progress=False, **kwargs)  # type: ignore
            signals[i] = IQlist[0][0].dot([1, 1j])  # type: ignore

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
