from typing import Optional, Tuple

from zcu_tools import make_cfg
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram

from ...tools import format_sweep1D, map2adcfreq, sweep2array, sweep2param
from ..template import sweep_hard_template


def measure_res_freq(soc, soccfg, cfg):
    """
    Measures the resonator frequency by performing a one-tone spectroscopy.

    This function configures and runs a one-tone spectroscopy experiment to characterize
    a resonator's frequency response. It sweeps the frequency of a pulse sent to the
    resonator and measures the amplitude response.

    Parameters
    ----------
    soc : object
        The system-on-chip object that handles the hardware interaction.
    soccfg : object
        The system-on-chip configuration object containing hardware settings.
    cfg : dict
        Configuration dictionary containing experiment parameters including:
        - dac.res_pulse: Settings for the resonator pulse
        - sweep: Frequency sweep parameters
        - adc.chs: ADC channels for measurement
    progress : bool, optional
        Whether to display a progress bar during the measurement, default is True.

    Returns
    -------
    fpts : ndarray
        Array of frequency points used in the measurement.
    signals : ndarray
        Measured amplitude response at each frequency point.

    Notes
    -----
    The frequency points are initially derived from the sweep configuration,
    but may be adjusted to match the actual hardware capabilities. The final
    frequency values are returned along with the measurement results.
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    res_pulse["freq"] = sweep2param("freq", sweep_cfg)

    fpts = sweep2array(sweep_cfg)  # predicted frequency points
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    prog: Optional[OneToneProgram] = None

    def measure_fn(cfg, callback) -> Tuple[list, list]:
        nonlocal prog
        prog = OneToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=True, callback=callback)

    signals = sweep_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Frequency (MHz)", "Amplitude"),
        ticks=(fpts,),
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return fpts, signals
