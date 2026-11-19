"""Shared label ownership and processor count contracts for compiled readouts."""

from collections.abc import Sequence

from .macro.readout import ReadoutEmission
from .modules.readout import MuxReadout, PulseReadout
from .readout_observation import ReadoutObservation
from .timeline import TimelineProgramV2


def emission_label(emission: ReadoutEmission) -> str:
    """Return an emit tag or module name; reject owners outside pulse/mux readouts."""
    owner = emission.owner
    if not isinstance(owner, (PulseReadout, MuxReadout)):
        raise ValueError("read labels require a PulseReadout or MuxReadout owner")
    return owner.name if emission.label is None else emission.label


def resolve_readout_label(
    observations: Sequence[ReadoutObservation], read: str
) -> PulseReadout | MuxReadout:
    """Resolve a label to its one borrowed pulse or mux readout owner.

    observations comes from observe_readouts or one observe_readout_paths path
    of a compiled program. read is an emit tag or an untagged module name.
    Repeated emissions of the same owner object are not ambiguous. Return that
    owner without mutation; emission multiplicity, ADC selection and control
    structure remain the caller's responsibility. Raise ValueError for missing
    labels or multiple distinct owners, listing available labels and owner count.
    """
    owners: dict[str, dict[int, PulseReadout | MuxReadout]] = {}
    for observation in observations:
        source = observation.source
        if isinstance(source, ReadoutEmission) and isinstance(
            source.owner, (PulseReadout, MuxReadout)
        ):
            owners.setdefault(emission_label(source), {})[id(source.owner)] = (
                source.owner
            )
    matches = owners.get(read, {})
    if len(matches) != 1:
        raise ValueError(
            f"label {read!r}: expected one readout owner, actual {len(matches)}; "
            f"available labels: {list(owners)}"
        )
    return next(iter(matches.values()))


def validate_declared_reads(prog: TimelineProgramV2, ch: int, expected: int) -> None:
    """Require a processor's expected read count on one ADC of compiled prog.

    Raise ValueError with the required and compiled counts if they differ.
    This checks the processor's input contract without changing prog.
    """
    actual = prog.ro_chs[ch]["trigs"]
    if actual != expected:
        raise ValueError(
            f"ADC {ch}: expected {expected} reads per shot; actual {actual}"
        )
