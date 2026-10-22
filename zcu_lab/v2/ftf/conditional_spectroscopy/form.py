"""Fixed-capture thermal spectroscopy Form with explicit two-line probe inputs."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_forms import (
    mux_readout_initial,
    mux_readout_seed_metadata,
)

from .._measure import captured_roles
from .core import SpectroscopyCfg


def make_form(ctx: FormContext) -> Form[SpectroscopyCfg]:
    """Capture roles, preparation/readout and available ordinary CZ pulse slots.

    Existing coupler CZ slots are editable starting points, not assumed weak
    probes. Without a slot, offer an explicit cosine envelope on the role's
    captured wiring channel; gain, duration and Nyquist zone remain required
    operator inputs. Run supplies both drive frequencies from the common scan.
    No model truth or safe-drive defaults are supplied. The frequency scan must
    be entered. Optional DC starts disabled.
    """
    roles = captured_roles(ctx)
    drive = {}
    for role in ("f1", "f2"):
        slot = f"$coupler.module.cz_{role}"
        drive[role] = (
            # dual_line_drive replaces both frequencies with the common scan.
            Slot(slot, copy=True, set={"freq": Hidden(0.0)})
            if ctx.has(slot)
            else Template(
                "pulse:blank",
                set={
                    "ch": Expr(f"${role}.wiring.ch"),
                    "nqz": None,
                    "freq": Hidden(0.0),
                    "gain": None,
                    "waveform.style": "cosine",
                    "waveform.length": None,
                },
            )
        )
    readout = mux_readout_initial(ctx)
    return Form(
        SpectroscopyCfg,
        tree={
            "roles": roles,
            "modules": {
                "x180_f1": Slot("$f1.module.x180", copy=True),
                "x180_f2": Slot("$f2.module.x180", copy=True),
                "readout": readout,
            },
            "drive": drive,
            "frequency": Range.between(None, None, expts=101),
            "dc_device": None,
            "dc_values": Range.between(0, 0, expts=1),
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
