import sys
from typing import Any, Callable, Dict, Tuple

from numpy import ndarray
from tqdm.auto import tqdm

from qick.asm_v2 import AveragerProgramV2
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow1D, InstantShow2D


def default_raw2signals(ir, sum_d, *_) -> ndarray:
    return sum_d[0][0].dot([1, 1j]) / (ir + 1)


def default_result2signals(IQlist) -> ndarray:
    return IQlist[0][0].dot([1, 1j])


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
    raw2signals: Callable = default_raw2signals,
    result2signals: Callable = default_result2signals,
    **kwargs,
) -> Tuple[AveragerProgramV2, ndarray]:
    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    signals = init_signals.copy()
    if instant_show:
        if len(ticks) == 1:
            xs = ticks[0]
            viewer = InstantShow1D(xs, xlabel, ylabel)
        elif len(ticks) == 2:
            xs, ys = ticks
            viewer = InstantShow2D(xs, ys, xlabel, ylabel)
        else:
            raise ValueError("ticks should be 1D or 2D")

        def callback(*args):
            nonlocal signals
            signals = raw2signals(*args)
            viewer.update_show(signals)
    else:
        callback = None

    prog: AveragerProgramV2 = prog_cls(soccfg, cfg)

    try:
        result = prog.acquire(soc, progress=progress, callback=callback, **kwargs)
        signals: ndarray = result2signals(result)
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception:
        print("Error during measurement:")
        err_msg = sys.exc_info()[1]
        if hasattr(err_msg, "_pyroTraceback"):
            print("".join(err_msg._pyroTraceback))
        else:
            print(err_msg)
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
    result2signals: Callable = default_result2signals,
    **kwargs,
) -> Tuple[ndarray, ndarray]:
    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    signals = init_signals.copy()
    if instant_show:
        viewer = InstantShow1D(xs, x_label=xlabel, y_label=ylabel)
        show_period = int(len(xs) / 20 + 0.99)

    try:
        for i, x in enumerate(tqdm(xs, desc=xlabel, smoothing=0, disable=not progress)):
            updateCfg(cfg, i, x)

            # set again in case of change
            set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

            prog: AveragerProgramV2 = prog_cls(soccfg, cfg)
            result = prog.acquire(soc, progress=False, **kwargs)
            signals[i] = result2signals(result)

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
