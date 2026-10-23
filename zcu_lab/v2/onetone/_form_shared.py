"""Starting-value policy shared by the single-tone spectroscopy Forms."""

from math import isfinite

from zcu_tools.cfg_model import Expr, FormNode, Hidden, Range, Template, TemplateSeed
from zcu_tools.gui.form_adapter import FormContext

from zcu_lab.templates import READOUT_PULSE_LENGTH_US


def initial_ro_length() -> float:
    """Return an editable RO window in us, 0.01 us shorter than the readout pulse.

    The readout Template supplies the pulse source. Open/Reset creates a
    detached direct RO window that remains independently editable.
    """
    return READOUT_PULSE_LENGTH_US - 0.01


def initial_readout_timing(ctx: FormContext) -> dict[str, TemplateSeed]:
    """Return editable single-resonator Template timing seeds for Open/Reset.

    ctx supplies the fixed capture. If resonator kappa is absent or null,
    seed trig_offset=time_of_fly. Otherwise require a finite positive kappa
    in MHz and seed trig_offset=time_of_fly+5/(2*pi*kappa) us;
    kappa=1 MHz adds approximately 0.795775 us. Missing capture or invalid
    kappa raises ValueError. Alias and source faults propagate.
    Seed pulse length=max(original template length, trig_offset+ro_length)
    in us, retaining the template-based RO window.
    Return Expr seeds for trig_offset and waveform.length, and a direct
    ro_length seed. Open/Reset resolves these into detached editable values.
    Subsequent cfg output retains the operator's edited timing.
    """
    if ctx.parameters is None:
        raise ValueError("One-tone timing requires a parameter capture")
    source = ctx.parameters.lookup_point("$resonator.kappa")
    kappa = source.value
    trigger = "$resonator.time_of_fly"
    if source.present and kappa is not None:
        if (
            isinstance(kappa, bool)
            or not isinstance(kappa, (int, float))
            or not isfinite(kappa)
            or kappa <= 0
        ):
            raise ValueError("$resonator.kappa must be finite and positive in MHz")
        trigger += " + 5 / (2 * pi * $resonator.kappa)"
    window = f"({trigger}) + {initial_ro_length()!r}"
    original = READOUT_PULSE_LENGTH_US
    # Encode max using the expression language's abs: max(a,b)=(a+b+abs(a-b))/2.
    length = f"({original!r} + ({window}) + abs({original!r} - ({window}))) / 2"
    return {
        "trig_offset": Expr(trigger),
        "ro_length": initial_ro_length(),
        "waveform.length": Expr(length),
    }


def initial_spectrum_tree(ctx: FormContext) -> dict[str, FormNode]:
    """Seed editable pulse-readout, linear sweep and averaging for Open/Reset.

    ctx supplies the fixed resonator capture used for timing and sweep expressions.
    Run supplies the hidden readout frequency. Return a fresh tree with 301 points
    over freq ± 1.5*kappa, gain 0.05, 100 reps/rounds and 1 us relax delay.
    Timing source failures propagate from initial_readout_timing.
    """
    return {
        "modules": {
            "reset": Hidden(),
            "readout": Template(
                "readout",
                bindings={"resonator": "$resonator"},
                set={
                    "freq": Hidden(0.0),
                    "gain": 0.05,
                    **initial_readout_timing(ctx),
                },
            ),
        },
        "sweep": {
            "freq": Range.around("$resonator.freq", "1.5 * $resonator.kappa", expts=301)
        },
        "relax_delay": 1.0,
        "reps": 100,
        "rounds": 100,
        "initial_delay": Hidden(),
    }
