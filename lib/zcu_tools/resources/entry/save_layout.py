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


def _checked_path(root: Path, destination: Path) -> Path:
    """Resolve relative/absolute destination against root, rejecting symlink escapes."""
    path = root / destination
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"{path}: output escapes its root")
    return path


@dataclass(frozen=True)
class WorkflowRoots:
    """Exact absolute workflow directories returned by workflow_roots.

    result stores small workflow metadata, such as manifest and journal.
    database stores heavy acquisition data. Neither directory need exist;
    this value does not create them or reserve a workflow run identity.
    """

    result: Path
    database: Path


def workflow_roots(
    *, result_path: Path, database_path: Path, workflow: str, run_id: str
) -> WorkflowRoots:
    """Calculate two exact workflow directories without publishing anything.

    result_path and database_path are exact entry directories, not parent roots;
    relative paths become absolute and directories need not exist. workflow is
    the selected workflow name; slash becomes underscore. That normalized name
    and opaque run_id must be nonempty POSIX/Windows-safe single segments, using
    the same rules as SaveLayout. Return each entry's
    workflows/<workflow>_<run_id> directory, keeping the complete run_id.
    Invalid names or a destination resolving outside its entry root raise ValueError.
    Filesystem resolution errors propagate; do not create directories or files,
    reserve names, or require point identity or a save timestamp.
    """
    workflow = workflow.replace("/", "_")
    validate_path_segment(workflow, field="workflow")
    validate_path_segment(run_id, field="run_id")
    result_root = result_path.resolve()
    database_root = database_path.resolve()
    destination = Path("workflows") / f"{workflow}_{run_id}"
    return WorkflowRoots(
        result=_checked_path(result_root, destination),
        database=_checked_path(database_root, destination),
    )


class SaveLayout:
    """Calculate run and member destinations without publishing files or directories.

    Experiment names prefix run directories, Result figure copies and Labber exports.
    Fixed run/figure/JSON destinations may already exist; callers own replacement.
    Labber names are selected by checking
    existing files, not atomically reserved. Each instance tracks figure paths;
    different keys sharing a path raise ValueError. Use a new instance for each
    save selection. Writers and save orchestration own serialization, locking
    and failure isolation.
    """

    def __init__(
        self,
        *,
        result_path: Path,
        database_path: Path,
        run_id: str,
        experiment: str,
        point: str | None,
        saved_at: datetime,
    ) -> None:
        """Bind exact entry paths, experiment name and opaque run/point identities.

        result_path and database_path resolve to absolute entry directories, not
        parent roots; they need not exist. experiment is the frontend's launch
        name, with slash replaced by underscore for all destinations. The result,
        run_id and non-None point must be nonempty single components without dot
        traversal, slash/backslash, NUL or POSIX/Windows anchors. saved_at must be
        timezone-aware; its local date selects the Labber date tree. Invalid
        names, identities or time raise ValueError before filesystem lookup.
        """
        experiment = experiment.replace("/", "_")
        validate_path_segment(experiment, field="experiment")
        validate_path_segment(run_id, field="run_id")
        if point is not None:
            validate_path_segment(point, field="point")
        _aware_time(saved_at, field="saved_at")
        self._result_path = result_path.resolve()
        self._database_path = database_path.resolve()
        self._run_id = run_id
        self._experiment = experiment
        self._point = point
        self._saved_at = saved_at
        self._figure_keys: dict[Path, ArtifactKey] = {}

    def outputs(self, key: ArtifactKey) -> tuple[Output, ...]:
        """Return ordered destinations for one ArtifactKey without writing them.

        Data returns canonical data_h5 then <experiment>_<run_id last six>_N.hdf5
        in the Labber date tree, starting at N=1 and skipping existing files.
        Run IDs shorter than six characters are kept whole. Multiple data
        artifacts share runs/<experiment>_<complete run_id>/data.h5. Figure returns
        run PNG then point/global <experiment>_<complete run_id>_<figure>.png copy.
        Analysis returns analysis/<member_name>.json in the same run directory;
        section/name do not affect that destination. Producers supply their pane
        as member_name.
        Figure filename tokens are joined with underscores without encoding.
        Different figure keys sharing a path within this layout raise ValueError
        naming both keys and the path; repeating the same key is allowed.
        Data and analysis destinations may be shared across keys.
        A symlink escaping either entry root raises ValueError; filesystem
        lookup errors propagate.
        """
        run_name = f"{self._experiment}_{self._run_id}"
        run = Path("runs") / run_name
        if key.member == "data":
            data_path = _checked_path(self._database_path, run / "data.h5")
            local = self._saved_at
            labber = (
                Path("Labber")
                / f"{local.year:04d}"
                / f"{local.month:02d}"
                / f"Data_{local.month:02d}{local.day:02d}"
                / f"{self._experiment}_{self._run_id[-6:]}_1.hdf5"
            )
            initial = _checked_path(self._database_path, labber)
            labber_path = Path(reserve_labber_filepath(str(initial)))
            # The suffix search may select a different existing symlink path.
            labber_path = _checked_path(self._database_path, labber_path)
            return Output("data_h5", data_path), Output("labber", labber_path)

        # ArtifactKey validates this non-data invariant at construction.
        assert key.member_name is not None
        if key.member == "analysis":
            return (
                Output(
                    "json",
                    _checked_path(
                        self._database_path,
                        run / "analysis" / f"{key.member_name}.json",
                    ),
                ),
            )

        filename = "_".join((key.section, key.name, key.member_name))
        figure = _checked_path(self._database_path, run / "figures" / f"{filename}.png")
        copy_folder = (
            Path("figures")
            if self._point is None
            else Path("points") / self._point / "figures"
        )
        copy = _checked_path(
            self._result_path, copy_folder / f"{run_name}_{filename}.png"
        )
        for path in (figure, copy):
            previous_key = self._figure_keys.get(path)
            if previous_key is not None and previous_key != key:
                raise ValueError(
                    f"Figure output collision at {path}: {previous_key!r} and {key!r}"
                )
        for path in (figure, copy):
            self._figure_keys[path] = key
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
