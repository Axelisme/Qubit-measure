from __future__ import annotations

import logging

from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.program.base import ImproveAcquireMixin

from .ir import IRCompileMixin
from .macro import ImproveAsmV2
from .modules.registry import PulseRegistry

logger = logging.getLogger(__name__)


class ProgramV2Cfg(ConfigBase):
    reps: int = 1
    rounds: int = 1
    initial_delay: float = 1.0
    relax_delay: float = 1.0


class MyProgramV2(  # type: ignore[reportIncompatibleMethodOverride]
    ImproveAcquireMixin, ImproveAsmV2, AveragerProgramV2, IRCompileMixin
):
    def __init__(self, soccfg: QickConfig, cfg: ProgramV2Cfg, **kwargs) -> None:

        # v2 program need to pass reps and final_delay to init
        self.cfg_model = cfg
        self.pulse_registry = PulseRegistry()

        super().__init__(
            soccfg,
            cfg=None,
            reps=cfg.reps,
            initial_delay=cfg.initial_delay,
            final_wait=0.0,
            final_delay=cfg.relax_delay,
            **kwargs,
        )

        logger.debug("ASM:\n%s", self.asm())

    def _initialize(self, cfg: ProgramV2Cfg) -> None: ...

    def _body(self, cfg: ProgramV2Cfg) -> None: ...

    def compile(self) -> None:
        super().compile()

        pmem_len = len(self.binprog["pmem"]) if self.binprog["pmem"] is not None else 0
        wmem_len = len(self.binprog["wmem"]) if self.binprog["wmem"] is not None else 0
        dmem_len = len(self.binprog["dmem"]) if self.binprog["dmem"] is not None else 0
        pmem_cap = self.tproccfg["pmem_size"]
        wmem_cap = self.tproccfg["wmem_size"]
        dmem_cap = self.tproccfg["dmem_size"]

        logger.debug(
            f"{self.__class__.__name__}.compile: "
            f"pmem={pmem_len}/{pmem_cap} ({(100 * pmem_len / pmem_cap if pmem_cap else 0):.1f}%), "
            f"wmem={wmem_len}/{wmem_cap} ({(100 * wmem_len / wmem_cap if wmem_cap else 0):.1f}%), "
            f"dmem={dmem_len}/{dmem_cap} ({(100 * dmem_len / dmem_cap if dmem_cap else 0):.1f}%)",
        )

    def acquire(self, soc, *args, **kwargs):
        logger.debug(
            "MyProgramV2.acquire: reps=%s, rounds=%s",
            self.cfg_model.reps,
            self.cfg_model.rounds,
        )

        # Sim dispatch (mocksim P1-5): a MockQickSoc carrying SimParams routes
        # through the SimEngine.  We *inject* the engine onto the soc (no eager
        # compute); the real round loop then runs unchanged and the soc's
        # poll_data computes each round lazily off the engine, so round_hook /
        # _process_accumulated / _summarize_accumulated / get_raw are all reused.
        # ``stop_checkers`` stay owned by the real round loop's finish_round(),
        # matching hardware's round-boundary stop semantics. With no SimParams this
        # branch is skipped entirely and behaviour is identical to the prior real
        # path (D1).
        sim = getattr(soc, "_sim_params", None)
        if sim is not None:
            self._attach_sim_engine(
                soc,
                sim,
            )

        return super().acquire(soc, *args, rounds=self.cfg_model.rounds, **kwargs)

    def acquire_decimated(self, soc, *args, **kwargs):
        logger.debug(
            "MyProgramV2.acquire_decimated: reps=%s, rounds=%s",
            self.cfg_model.reps,
            self.cfg_model.rounds,
        )

        # Sim dispatch (mocksim D2): a MockQickSoc carrying SimParams routes the
        # decimated (time-domain / lookback) path through the SimEngine too.  We
        # inject the engine onto the soc (no eager compute); the real decimated
        # round loop then runs unchanged and the soc's get_decimated renders each
        # round's trace lazily off the engine (model A, timeFly-shifted readout
        # envelope × steady mixed S21).  With no SimParams this branch is skipped
        # and behaviour is identical to the prior real path (D1).
        sim = getattr(soc, "_sim_params", None)
        if sim is not None:
            self._attach_sim_engine(
                soc,
                sim,
            )

        return super().acquire_decimated(
            soc, *args, rounds=self.cfg_model.rounds, **kwargs
        )

    def _attach_sim_engine(
        self,
        soc,
        sim,
    ) -> None:
        """Build the SimEngine and inject it onto the mock soc for poll-time compute.

        Ensures the program is compiled (the engine reads loop_dims / ro_chs /
        the module tree), constructs the engine, and hands it to the mock soc;
        no rounds are computed here — poll_data drives compute lazily.  Lives
        here (not in acquire) to keep the dispatch branch readable; engine
        construction raises (does not swallow) so unsupported experiment
        structures fail fast instead of degrading to noise.
        """

        from .sim.engine import SimEngine

        if self.loop_dims is None or self.avg_level is None:
            self.compile()

        soc.set_sim_engine(SimEngine(self, sim))
