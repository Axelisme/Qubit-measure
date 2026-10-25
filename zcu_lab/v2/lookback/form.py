"""Fresh lookback defaults from the captured resonator and persistent templates."""

from math import isfinite

from zcu_tools.cfg_model import FieldMeta, Hidden, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.lookback.core import LookbackCfg


def _initial_trig_offset(ctx: FormContext) -> float | None:
    """Choose an editable offset in us from the fixed resonator TOF capture.

    Missing capture or absent/null TOF uses a 0 us first-capture starting point,
    not a calibration. A finite non-bool real TOF >= 0 uses TOF - 0.1 us,
    including negative offsets. Invalid stored TOF leaves the required field
    blank. Lookup source, alias and path errors propagate unchanged.
    """
    if ctx.parameters is None:
        return 0.0
    result = ctx.parameters.lookup("$resonator.time_of_fly")
    if not result.present or result.value is None:
        return 0.0
    value = result.value
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        return None
    return value - 0.1


def _initial_frequency(ctx: FormContext) -> float | None:
    """Return f_r + kappa in MHz from the fixed capture, or a required blank.

    Missing/null kappa uses 1 MHz. Stored frequency and linewidth must be
    positive finite non-bool real scalars; invalid values leave the field blank.
    Missing/null frequency or no capture has no numeric seed, regardless of
    linewidth. Source and alias errors propagate.
    No calibration is changed.
    """
    if ctx.parameters is None:
        return None
    frequency = ctx.parameters.lookup("$resonator.freq")
    linewidth = ctx.parameters.lookup("$resonator.kappa")
    freq = frequency.value
    kappa = (
        linewidth.value if linewidth.present and linewidth.value is not None else 1.0
    )
    if (
        isinstance(freq, bool)
        or not isinstance(freq, (int, float))
        or not isfinite(freq)
        or freq <= 0
        or isinstance(kappa, bool)
        or not isinstance(kappa, (int, float))
        or not isfinite(kappa)
        or kappa <= 0
    ):
        return None
    return freq + kappa


def make_form(ctx: FormContext) -> Form[LookbackCfg]:
    """Build editable Lookback defaults from one fixed Open/Reset capture.

    ctx supplies fixed resonator frequency, linewidth and TOF. Frequency starts
    at f_r + kappa in MHz, using 1 MHz for missing/null kappa. Missing/null f_r,
    no capture or invalid frequency/linewidth leaves frequency required and blank.
    Missing TOF starts at 0 us, valid TOF at TOF - 0.1 us and invalid TOF leaves
    the trigger offset required and blank. Capture lookup faults propagate.
    The readout window starts at 1.5 us; the operator must extend it if needed.
    Initial values remain editable. No calibration data is written.
    """
    return Form(
        LookbackCfg,
        field_order=(
            "reset",
            "init_pulse",
            "readout",
            "rounds",
            "relax_delay",
            "initial_delay",
        ),
        tree={
            "reset": Hidden(),
            "init_pulse": Hidden(),
            "readout": Template(
                "readout",
                bindings={"resonator": "$resonator"},
                set={
                    "gain": 1.0,
                    "freq": _initial_frequency(ctx),
                    "ro_length": 1.5,
                    "trig_offset": _initial_trig_offset(ctx),
                },
            ),
            "relax_delay": 0.0,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    ).with_seed_metadata(
        {
            ("readout", "module", "gain"): FieldMeta(
                description="未校準的起始值，來源：固定常數。lookback Template gain 1.0 a.u. within [-1, 1]; operator confirmation required",
                calibration="uncalibrated",
            ),
            ("readout", "module", "freq"): FieldMeta(
                description="$resonator.freq + kappa (MHz); missing/null kappa uses a 1 MHz offset，未校準的起始值，來源：固定常數。operator confirmation required",
                calibration="uncalibrated",
            ),
            ("readout", "module", "trig_offset"): FieldMeta(
                description="$resonator.time_of_fly - 0.1 us; missing/null TOF uses a 0 us offset，未校準的起始值，來源：固定常數。operator confirmation required",
                calibration="uncalibrated",
            ),
            ("readout", "module", "ro_length"): FieldMeta(
                description="未校準的起始值，來源：固定常數。lookback Template ADC window 1.5 us; operator confirmation required",
                calibration="uncalibrated",
            ),
        }
    )
