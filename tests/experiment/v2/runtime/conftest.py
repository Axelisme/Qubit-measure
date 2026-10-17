"""Function-local lifetime for shared runtime recording fakes."""

from collections.abc import Iterator

import pytest

from tests.experiment.v2.runtime._flow_fakes import (
    AlwaysFailingProgram,
    FakeCachedProgram,
    FakeDecimatedProgram,
    FakeProgram,
    FlakyBuildProgram,
    InterruptingProgram,
    StopBeforeDataProgram,
)


@pytest.fixture(autouse=True)
def restore_flow_fake_state(request: pytest.FixtureRequest) -> Iterator[None]:
    """Restore shared recording state after each runtime case, naming any leak."""
    # Keep list identities: callers may retain a reference to a recording list.
    programs = FakeProgram.instances
    cached = FakeCachedProgram.instances
    decimated = FakeDecimatedProgram.instances
    flaky = FlakyBuildProgram.instances
    failing = AlwaysFailingProgram.instances
    interrupting = InterruptingProgram.instances
    programs_before = programs.copy()
    cached_before = cached.copy()
    decimated_before = decimated.copy()
    flaky_before = flaky.copy()
    failing_before = failing.copy()
    interrupting_before = interrupting.copy()
    external_stop_before = StopBeforeDataProgram.external_stop

    try:
        yield
    finally:
        programs[:] = programs_before
        cached[:] = cached_before
        decimated[:] = decimated_before
        flaky[:] = flaky_before
        failing[:] = failing_before
        interrupting[:] = interrupting_before
        StopBeforeDataProgram.external_stop = external_stop_before

        owner = request.node.nodeid
        assert FakeProgram.instances is programs and programs == programs_before, (
            f"{owner}: FakeProgram.instances was not restored"
        )
        assert FakeCachedProgram.instances is cached and cached == cached_before, (
            f"{owner}: FakeCachedProgram.instances was not restored"
        )
        assert (
            FakeDecimatedProgram.instances is decimated
            and decimated == decimated_before
        ), f"{owner}: FakeDecimatedProgram.instances was not restored"
        assert FlakyBuildProgram.instances is flaky and flaky == flaky_before, (
            f"{owner}: FlakyBuildProgram.instances was not restored"
        )
        assert (
            AlwaysFailingProgram.instances is failing and failing == failing_before
        ), f"{owner}: AlwaysFailingProgram.instances was not restored"
        assert (
            InterruptingProgram.instances is interrupting
            and interrupting == interrupting_before
        ), f"{owner}: InterruptingProgram.instances was not restored"
        assert StopBeforeDataProgram.external_stop is external_stop_before, (
            f"{owner}: StopBeforeDataProgram.external_stop was not restored"
        )
