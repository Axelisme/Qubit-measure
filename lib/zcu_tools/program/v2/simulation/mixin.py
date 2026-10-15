"""Single synchronous simulation entry ahead of the original acquisition mixins."""

from zcu_tools.program.acquisition import ImproveAcquireMixin

from .soc import SimulationSoc


class SimulationAcquireMixin(ImproveAcquireMixin):
    """Delegate rounds/hooks/Stop/classification/averaging to the normal pipeline.

    Bound simulations reject stepped ownership and reentry before changing hooks
    or acquire_params. No binding leaves the original hardware path unchanged.
    """

    _simulation_active: bool = False

    def acquire(self, soc, *args, **kwargs):
        if not isinstance(soc, SimulationSoc):
            return super().acquire(soc, *args, **kwargs)
        with self._simulation_call(soc, "integrated", kwargs):
            return super().acquire(soc, *args, **kwargs)

    def acquire_decimated(self, soc, *args, **kwargs):
        if not isinstance(soc, SimulationSoc):
            return super().acquire_decimated(soc, *args, **kwargs)
        with self._simulation_call(soc, "decimated", kwargs):
            return super().acquire_decimated(soc, *args, **kwargs)

    def _simulation_call(self, soc, mode, kwargs):
        from contextlib import contextmanager

        from .frontend import build_request

        @contextmanager
        def call():
            if kwargs.get("step_rounds", False):
                raise ValueError("simulation does not support step_rounds=True")
            if getattr(self, "_simulation_active", False):
                raise RuntimeError("simulation acquisition is already active (reentry)")
            sequence = soc.binding.begin()
            self._simulation_active = True
            try:
                request = build_request(self, mode, soc.binding, sequence)
                soc.install_request(request)
                yield
            finally:
                soc.release_request()
                self._simulation_active = False
                soc.binding.end()

        return call()
