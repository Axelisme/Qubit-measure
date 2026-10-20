"""Single synchronous simulation entry ahead of the original acquisition mixins."""

from collections.abc import Callable
from inspect import Signature, signature
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from qick.qick_asm import AcquireMixin

from zcu_tools.core.simulation import RoundStatistics
from zcu_tools.program.acquisition import CancelFlagProtocol, ImproveAcquireMixin
from zcu_tools.progress_bar import make_pbar

from .soc import SimulationSoc

if TYPE_CHECKING:
    from ..modular import ModularProgramV2


# One unbound callable and its metadata, never an acquiring program instance.
_acquire_signature: tuple[Callable[..., object], Signature] | None = None


class SimulationAcquireMixin(ImproveAcquireMixin):
    """Delegate rounds/hooks/Stop/classification/averaging to the normal pipeline.

    Bound simulations reject stepped ownership and reentry before changing hooks
    or acquire_params. No binding leaves the original hardware path unchanged.
    """

    _simulation_active: bool = False
    _simulation_raw: tuple[RoundStatistics, int, int] | None = None

    def _acquire_configured(self, soc, *args, **kwargs) -> list[NDArray[np.float64]]:
        if not isinstance(soc, SimulationSoc):
            self._simulation_raw = None
            return super()._acquire_configured(soc, *args, **kwargs)
        # Bind against QICK so its positional API and invalid-argument errors
        # remain intact; processing modifiers have already populated extra_args.
        global _acquire_signature
        acquire = AcquireMixin.acquire
        cached = _acquire_signature
        if cached is None or cached[0] is not acquire:
            cached = (acquire, signature(acquire))
            _acquire_signature = cached
        bound = cached[1].bind(self, soc, *args, **kwargs)
        options = dict(bound.arguments)
        del options["self"], options["soc"]
        return self._acquire_integrated(soc, **options)

    def _acquire_integrated(
        self, soc: SimulationSoc, **options
    ) -> list[NDArray[np.float64]]:
        # options has already been bound to QICK's acquisition signature.
        rounds = options.get("rounds", 1)
        params = {
            "type": "accumulated",
            "soc": soc,
            "start_src": options.get("start_src", "internal"),
            "rounds_remaining": rounds,
            "remove_offset": options.get("remove_offset", True),
            "hidereps": True,
            "threshold": options.get("threshold"),
            "angle": options.get("angle"),
        }
        if options.get("extra_args") is not None:
            params.update(options["extra_args"])
        self.acquire_params = params
        if (
            self.counter_addr is None
            or self.loop_dims is None
            or self.avg_level is None
        ):
            raise RuntimeError("data dimensions must be defined before acquire")
        self.acc_buf = None
        self._simulation_raw = None
        self.rounds_buf = []
        self.stats = []
        program = self._program()
        # QICK compiles before loading envelopes. Simulation consumes the same
        # compiled buffers directly and has no DAC/weight transport to upload.
        program.config_all(soc, load_envelopes=False, load_mem=False)
        if options.get("load_envelopes", True):
            self._validate_simulation_uploads()
        self.rounds_pbar = make_pbar(
            total=rounds, disable=not options.get("progress", True), leave=False
        )
        try:
            self.prepare_round()
            while self.finish_round():
                self.prepare_round()
            result = self.finish_acquire()
            if result is None:
                raise RuntimeError("simulation acquisition returned no result")
            return result
        finally:
            self._close_rounds_pbar()

    def _validate_simulation_uploads(self) -> None:
        program = self._program()
        for envelopes in program.envelopes:
            contents = envelopes["envs"]
            if not isinstance(contents, dict):
                raise ValueError("compiled generator envelope table is missing")
            for pulse in contents.values():
                if pulse["data"].dtype != np.int16:
                    raise AssertionError("envelope data must have dtype int16")
        for channel, config in program.ro_chs.items():
            if program.soccfg["readouts"][channel]["has_weights"]:
                weights = config["weights"]
                if weights is not None and weights.dtype != np.int16:
                    raise AssertionError("readout weights must have dtype int16")

    def prepare_round(self) -> None:
        """Prepare simulated integrated rounds without QICK's raw-buffer clearing.

        All hardware and decimated calls retain the original QICK preparation.
        """
        params = self.acquire_params
        if (
            not isinstance(params["soc"], SimulationSoc)
            or params["type"] != "accumulated"
        ):
            super().prepare_round()
            return
        soc = params["soc"]
        soc.prepare_round()
        soc.start_src(params["start_src"])

    def _finish_accumulated_round(self, cancel_flag: CancelFlagProtocol) -> bool:
        params = self.acquire_params
        if not isinstance(params["soc"], SimulationSoc):
            return super()._finish_accumulated_round(cancel_flag)
        soc = params["soc"]
        result = soc.run_round()
        if cancel_flag.is_set():
            soc.cleanup_round()
            self._close_rounds_pbar()
            return False
        if not isinstance(result, RoundStatistics):
            raise ValueError("integrated acquisition requires RoundStatistics")
        self._consume_integrated_result(soc, result)
        soc.cleanup_round()
        self.rounds_pbar.update()
        remaining = params["rounds_remaining"]
        if not isinstance(remaining, int):
            raise ValueError("rounds_remaining must be an integer")
        params["rounds_remaining"] = remaining - 1
        if remaining <= 1:
            self._close_rounds_pbar()
            return False
        return True

    def _program(self) -> "ModularProgramV2":
        from ..modular import ModularProgramV2

        if not isinstance(self, ModularProgramV2):
            raise ValueError("simulation requires a ModularProgramV2 frontend")
        return self

    def _consume_integrated_result(
        self, soc: SimulationSoc, result: RoundStatistics
    ) -> None:
        assert self.acquire_params is not None
        assert self.rounds_buf is not None
        assert self.avg_level is not None
        params = self.acquire_params
        prog = self._program()
        assert self.loop_dims is not None
        request = soc.binding.last_request
        if request is None:
            raise RuntimeError("simulation round requires installed request")
        self._simulation_raw = (result, request.reps, request.avg_level)
        needs_raw = any(
            params.get(key) is not None
            for key in ("threshold", "ge_radius", "trackers")
        )
        if not needs_raw:
            self.acc_buf = None
            self.rounds_buf.append(self._average_statistics(result, request.reps))
            return
        raw = result.materialize_raw(request.reps, request.avg_level)
        self.acc_buf = [raw[ch] for ch in prog.ro_chs]
        self.rounds_buf.append(prog._process_accumulated(self.acc_buf))

    def _average_statistics(
        self, result: RoundStatistics, reps: int
    ) -> list[NDArray[np.float64]]:
        assert self.acquire_params is not None
        by_channel = {ch.channel: ch for ch in result.channels}
        averaged = []
        for ch, ro in self._program().ro_chs.items():
            avg = by_channel[ch].sums / reps
            if not ro["edge_counting"]:
                avg /= ro["length"]
                if self.acquire_params["remove_offset"]:
                    avg -= self._ro_offset(ch, ro.get("ro_config"))
            averaged.append(np.moveaxis(avg, -2, 0))
        return averaged

    def get_raw(self) -> list[NDArray[np.int64]] | None:
        """Return last-round int64 QICK sums, materializing compact shots once.

        Deferred shots use the retained round identity, never a new backend call.
        Hardware and decimated data use QICK buffers.
        """
        if self.acc_buf is None and self._simulation_raw is not None:
            result, reps, avg_level = self._simulation_raw
            raw = result.materialize_raw(reps, avg_level)
            self.acc_buf = [raw[ch] for ch in self._program().ro_chs]
        return super().get_raw()

    def acquire(self, soc, *args, **kwargs):
        if not isinstance(soc, SimulationSoc):
            return super().acquire(soc, *args, **kwargs)
        with self._simulation_call(soc, "integrated", kwargs):
            return super().acquire(soc, *args, **kwargs)

    def acquire_decimated(self, soc, *args, **kwargs):
        if not isinstance(soc, SimulationSoc):
            return super().acquire_decimated(soc, *args, **kwargs)
        with self._simulation_call(soc, "decimated", kwargs):
            self._simulation_raw = None
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
