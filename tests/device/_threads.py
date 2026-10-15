"""Own device-test workers, release gates on failure, and propagate errors."""

import logging
from collections.abc import Callable
from threading import Event, Thread


class DeviceThreads:
    """Track real callers and their external blockers through test teardown."""

    def __init__(self) -> None:
        self.threads: list[Thread] = []
        self.releases: list[Event] = []
        self.errors: list[BaseException] = []

    def thread(self, *, target: Callable[[], None], release: Event) -> Thread:
        """Build a caller; teardown releases its collaborator before joining it."""
        self.releases.append(release)

        def run() -> None:
            try:
                target()
            except BaseException as exc:
                logging.exception("Device caller failed; teardown will rethrow")
                self.errors.append(exc)

        worker = Thread(target=run)
        self.threads.append(worker)
        return worker

    def arrived(self, event: Event, message: str = "caller did not arrive") -> None:
        """Observe the signal emitted by a scheduled collaborator at its seam."""
        # The started caller announces entry/completion at the controlled seam.
        assert event.wait(30.0), message

    def join(self, worker: Thread) -> None:
        """Join a caller after its external collaborator has been released."""
        # Released driver I/O lets the scheduled caller finish.
        worker.join(30.0)
        assert not worker.is_alive(), "device caller did not finish after release"

    def finish(self) -> None:
        """Release every registered blocker, join started callers, rethrow errors."""
        for release in self.releases:
            release.set()
        for worker in self.threads:
            if worker.ident is not None:
                # Released collaborators must let every scheduled caller finish.
                worker.join(30.0)
        assert all(not worker.is_alive() for worker in self.threads), (
            "device caller did not finish after release"
        )
        if self.errors:
            raise BaseExceptionGroup("device caller failed", self.errors)
