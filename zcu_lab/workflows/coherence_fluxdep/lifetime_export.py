"""Scientific lifetime declaration selection for workflow conversion tools."""

from pathlib import Path

from zcu_tools.datafile import LabberPayload
from zcu_tools.experiment.workflows.export import load_trace

from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoExp
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyExp
from zcu_lab.workflows.coherence_fluxdep.measurements import LifetimeKind


def load_lifetime_trace(
    directory: Path, filename: str, kind: LifetimeKind
) -> LabberPayload:
    """Read a named completed lifetime with its own native declaration.

    kind selects T1, Ramsey or Echo, not a union schema or common delay grid.
    Directory/reference/schema/I/O errors retain load_trace's source diagnosis.
    """
    if kind == "t1":
        return load_trace(directory, filename, T1Exp.AXES_SPEC)
    if kind == "t2ramsey":
        return load_trace(directory, filename, T2RamseyExp.AXES_SPEC)
    return load_trace(directory, filename, T2EchoExp.AXES_SPEC)
