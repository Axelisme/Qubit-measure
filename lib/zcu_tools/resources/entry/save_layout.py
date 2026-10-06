"""Output destinations for one run; serialization belongs to the chosen writers."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from zcu_tools.datafile import reserve_labber_filepath

from ._path_segments import validate_path_segment
from .ledger_models import OutputFormat


@dataclass(frozen=True)
class Output:
    """One writer destination: format selects serialization, path is absolute and exact."""

    format: OutputFormat
    path: Path


@dataclass(frozen=True)
class ArtifactKey:
    """One artifact member in a caller-defined section.

    section and name are opaque, nonempty path components: no dot traversal,
    slash/backslash, NUL or POSIX/Windows anchors. member selects data,
    figure or analysis. member_name identifies a figure/analysis member and is
    required for those kinds; data must use None. Invalid components, unknown
    member kinds or inconsistent member_name raise ValueError at construction.
    """

    section: str
    name: str
    member: Literal["data", "figure", "analysis"]
    member_name: str | None = None

    def __post_init__(self) -> None:
        """Validate member identity without interpreting section or artifact names."""
        validate_path_segment(self.section, field="section")
        validate_path_segment(self.name, field="name")
        if self.member not in ("data", "figure", "analysis"):
            raise ValueError("member must be data, figure or analysis")
        if self.member == "data":
            if self.member_name is not None:
                raise ValueError("member_name must be None for data")
        else:
            if self.member_name is None:
                raise ValueError("member_name is required for figure and analysis")
            validate_path_segment(self.member_name, field="member_name")


def _aware_time(value: datetime, *, field: str) -> None:
    """Reject datetime values that cannot identify an instant."""
    if value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _encode(value: str) -> str:
    """Keep opaque name tokens distinct from underscore filename separators."""
    return value.replace("%", "%25").replace("_", "%5F")


def _checked_path(root: Path, destination: Path) -> Path:
    """Resolve relative/absolute destination against root, rejecting symlink escapes."""
    path = root / destination
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"{path}: output escapes its root")
    return path


class SaveLayout:
    """Calculate run and member destinations without publishing files or directories.

    Paths carry no experiment semantics. Fixed run/figure/JSON destinations may
    already exist; callers own replacement. Labber names are selected by checking
    existing files, not atomically reserved. Writers and save orchestration own
    serialization, locking and failure isolation.
    """

    def __init__(
        self,
        *,
        result_path: Path,
        database_path: Path,
        run_id: str,
        point: str | None,
        saved_at: datetime,
    ) -> None:
        """Bind exact entry paths and opaque run/point identities.

        result_path and database_path resolve to absolute entry directories, not
        parent roots; they need not exist. run_id and non-None point must be
        nonempty single components without dot traversal, slash/backslash, NUL
        or POSIX/Windows anchors. saved_at must be timezone-aware; its local date
        selects the Labber date tree. Invalid identities/time raise ValueError.
        """
        validate_path_segment(run_id, field="run_id")
        if point is not None:
            validate_path_segment(point, field="point")
        _aware_time(saved_at, field="saved_at")
        self._result_path = result_path.resolve()
        self._database_path = database_path.resolve()
        self._run_id = run_id
        self._point = point
        self._saved_at = saved_at

    def outputs(self, key: ArtifactKey) -> tuple[Output, ...]:
        """Return ordered destinations for one ArtifactKey without writing them.

        Data returns canonical data_h5 then a new Labber filename. Multiple data
        artifacts share the run's data.h5. Figure returns run PNG then point/global
        PNG copy. Analysis returns one run JSON. Filename tokens escape percent
        then underscore, preserving member identity. A symlink escaping either
        entry root raises ValueError; filesystem lookup errors propagate.
        """
        run = Path("runs") / self._run_id
        if key.member == "data":
            data_path = _checked_path(self._database_path, run / "data.h5")
            local = self._saved_at
            labber = (
                Path("Labber")
                / f"{local.year:04d}"
                / f"{local.month:02d}"
                / f"Data_{local.month:02d}{local.day:02d}"
                / f"{self._run_id}_1.hdf5"
            )
            initial = _checked_path(self._database_path, labber)
            labber_path = Path(reserve_labber_filepath(str(initial)))
            # The suffix search may select a different existing symlink path.
            labber_path = _checked_path(self._database_path, labber_path)
            return Output("data_h5", data_path), Output("labber", labber_path)

        # ArtifactKey validates this non-data invariant at construction.
        assert key.member_name is not None
        filename = "_".join(
            _encode(token) for token in (key.section, key.name, key.member_name)
        )
        if key.member == "analysis":
            return (
                Output(
                    "json",
                    _checked_path(
                        self._database_path, run / "analysis" / f"{filename}.json"
                    ),
                ),
            )

        figure = _checked_path(self._database_path, run / "figures" / f"{filename}.png")
        copy_folder = (
            Path("figures")
            if self._point is None
            else Path("points") / self._point / "figures"
        )
        copy = _checked_path(
            self._result_path, copy_folder / f"{_encode(self._run_id)}_{filename}.png"
        )
        return Output("png", figure), Output("png", copy)


def new_run_id(*, at: datetime | None = None) -> str:
    """Return `%Y%m%dT%H%M%SZ-<six lowercase UUID4 hex characters>`.

    at defaults to the current UTC instant. A supplied aware datetime converts
    to UTC; naive input raises ValueError. The suffix is not a cross-process
    reservation or a guarantee against collisions.
    """
    instant = datetime.now(timezone.utc) if at is None else at
    _aware_time(instant, field="at")
    utc = instant.astimezone(timezone.utc)
    # strftime %Y does not pad years below 1000 on every platform.
    return f"{utc.year:04d}{utc:%m%dT%H%M%SZ}-{uuid4().hex[:6]}"
