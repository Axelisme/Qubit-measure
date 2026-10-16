"""Fresh lookback defaults from the captured resonator and persistent templates."""

from math import isfinite

from zcu_tools.cfg_model import Hidden, Locked, Template
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
    result = ctx.parameters.lookup("$resonator.ext.time_of_fly")
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


def make_form(ctx: FormContext) -> Form[LookbackCfg]:
    """Build editable Lookback defaults from one fixed Open/Reset capture.

    ctx supplies the resonator TOF; missing TOF starts at 0 us, valid TOF at
    TOF - 0.1 us and invalid TOF leaves the trigger offset required and blank.
    Capture lookup faults propagate. The readout window starts at 1.5 us;
    the operator must extend it if needed. No calibration data is written.
    """
    return Form(
        LookbackCfg,
        tree={
            "modules": {
                "reset": Hidden(),
                "init_pulse": Hidden(),
                "readout": Template(
                    "readout",
                    set={
                        "gain": 1.0,
                        "ro_length": 1.5,
                        "trig_offset": _initial_trig_offset(ctx),
                    },
                ),
            },
            "relax_delay": 0.0,
            "reps": Locked(1),
            "rounds": 500,
            "initial_delay": Hidden(),
        },
    )
