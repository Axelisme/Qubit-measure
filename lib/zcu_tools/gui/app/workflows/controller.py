"""Owner-thread admission and worker lifecycle for native workflow runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel

from zcu_tools.experiment.workflows import (
    Engine,
    EnginePorts,
    RunIdentity,
    RunPaths,
    RunStatus,
)
from zcu_tools.experiment.workflows.declaration import WorkflowStep

if TYPE_CHECKING:
    from zcu_tools.gui.session.operation_runner import OperationRunner


class WorkflowController[C]:
    """Submit one Start/Stop operation through the session's shared runner.

    All methods except Engine queries are called on the runner's owner thread.
    The controller owns admission and the pending operation token, not native
    lifecycle or a second device registry. ``engine`` is None until the worker
    accepts startup; its status and artifacts remain authoritative thereafter.
    An operation token is not proof that Engine.start has succeeded. Pause and
    Resume are not part of this initial Start/Stop attachment.
    """

    def __init__(
        self,
        runner: OperationRunner,
        ports: EnginePorts[C],
        owner_id: str,
        *,
        on_finished: Callable[[RunStatus | None, Exception | None], None],
    ) -> None:
        """Borrow runner/ports and report worker completion on the owner thread.

        owner_id labels the run lease and progress. The worker uses the runner's
        bound progress factory instead of ports.progress. on_finished receives
        the authoritative status (None if startup was never accepted) and the
        original startup/worker error when present. It runs before handle settle
        and lease release; reentrant Start remains blocked until cleanup.
        The caller owns connections and finishes/releases ports.plots only after
        this callback or a synchronous begin failure. No rollback is promised.
        """
        raise NotImplementedError

    @property
    def engine(self) -> Engine[C] | None:
        """Return the accepted native engine, or None before accepted startup."""
        raise NotImplementedError

    @property
    def operation_id(self) -> int | None:
        """Return the pending runner token, or None after terminal delivery."""
        raise NotImplementedError

    def start[P: BaseModel, T: BaseModel, S, R](
        self,
        step: WorkflowStep[P, T, S, R, C],
        *,
        plan: P,
        tunables: T,
        paths: RunPaths,
        identity: RunIdentity,
    ) -> int:
        """Admit and submit captured, validated startup, returning a runner token.

        One controller accepts exactly one Start; duplicate/pending Start raises
        InvalidRunState. Native declaration validation belongs to Engine and is
        performed on the worker while holding the run lease. Runner admission,
        registration/factory/submit failures propagate synchronously and leave
        no native run directories. Worker startup failures settle failed with
        their original error passed to on_finished; partial files are retained.
        Stop before worker startup creates no native run. Stop arriving during
        startup is delivered before execute calls init/step. Terminal is not
        reported until the worker has returned. No retry or hardware cleanup.
        """
        raise NotImplementedError

    def status(self) -> RunStatus | None:
        """Read accepted native status; None distinguishes pending startup."""
        raise NotImplementedError

    def stop(self) -> RunStatus | None:
        """Request cooperative Stop without waiting for worker terminal delivery.

        Before acceptance this cancels pending startup and returns None. After
        acceptance it delegates to Engine.stop and returns its native status.
        It cannot free the run lease; the worker's terminal callback does that.
        Native failures propagate and no hardware rollback/cleanup occurs.
        """
        raise NotImplementedError
