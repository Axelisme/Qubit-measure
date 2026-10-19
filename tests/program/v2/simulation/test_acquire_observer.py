"""Acquire completion observers run once and cannot break acquisition admission."""

import warnings

import pytest
from zcu_tools.core.simulation import AcquireRequest, EnvironmentSnapshot, RoundSignals
from zcu_tools.program.v2.simulation import SimulationBinding


def test_completion_observer_sees_all_rounds_and_cannot_mask_execution_failure():
    observed = []

    def finish(rounds, wall_s):
        observed.append((rounds, wall_s))
        raise OSError("observer cannot write")

    def execute(request):
        if request.index == 1:
            raise ValueError("original execution failure")
        return RoundSignals(())

    try:
        from zcu_tools.program.v2.simulation import SimulationCallbacks

        binding = SimulationBinding(
            lambda: EnvironmentSnapshot("synthetic", "test", ()),
            execute,
            backend_version="test",
            seed=1,
            callbacks=SimulationCallbacks(on_end=finish),
        )
    except ImportError as error:
        raise AssertionError("binding must notify acquire completion") from error
    sequence = binding.begin()
    request = AcquireRequest("decimated", (1,), 0, (), 1, sequence, binding.rng_version)
    def acquire():
        try:
            binding.sample_round(request, 0)
            binding.sample_round(request, 1)
        finally:
            binding.end()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="original execution failure"):
            acquire()
    assert len(observed) == 1
    assert len(observed[0][0]) == 2
    assert type(observed[0][1]) is float
    binding.begin()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        binding.end()
    assert len(observed) == 2
    assert observed[1][0] == ()
