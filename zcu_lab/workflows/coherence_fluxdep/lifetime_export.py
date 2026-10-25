"""Scientific lifetime declaration selection for workflow conversion tools."""

from pathlib import Path

from zcu_tools.datafile import LabberPayload
from zcu_tools.experiment.workflows.export import load_trace

from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoExp
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyExp
from zcu_lab.workflows.coherence_fluxdep.measurements import EchoResult, RamseyResult
from zcu_lab.workflows.coherence_fluxdep.stages.t1 import T1Result


def load_lifetime_trace(
    directory: Path, result: T1Result | RamseyResult | EchoResult
) -> LabberPayload:
    """Read a named completed lifetime with its own native declaration.

    result is the point's t1, t2ramsey or t2echo field with a saved native file.
    Its kind selects the experiment declaration, not a common delay grid.
    Missing file raises ValueError. Directory/reference/schema/I/O errors retain
    load_trace's source diagnosis.
    """
    if result.file is None:
        raise ValueError("Lifetime result has no saved native file")
    if result.kind == "t1":
        return load_trace(directory, result.file, T1Exp.AXES_SPEC)
    if result.kind == "t2ramsey":
        return load_trace(directory, result.file, T2RamseyExp.AXES_SPEC)
    return load_trace(directory, result.file, T2EchoExp.AXES_SPEC)
