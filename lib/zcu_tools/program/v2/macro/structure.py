"""Native QICK operations annotated without replacing their control shell."""

from qick.asm_v2 import CloseLoop, OpenLoop, Pulse, QickRawParam

from .wmem import PatchWmemFromDmem


class SourcePulse(Pulse):
    """Native Pulse with a diagnostic source_name around its full expansion.

    All normal QICK Pulse fields retain their meaning. source_name is the
    explicit tag or native:pulse:<ordinal>, allocated by the program at call
    time. Inheriting Pulse preserves native timing and Timeline adapters.
    """

    def translate(self, prog) -> None:
        prog.record_meta(type="SOURCE_START", name=self.source_name)
        super().translate(prog)
        prog.record_meta(type="SOURCE_END", name=self.source_name)


class MarkedOpenLoop(OpenLoop):
    """Keep QICK loop discovery and emit pre-optimization native loop records.

    Inherited fields name and n are QICK loop name and positive trip count.
    native=True means QICK instructions own the loop shell; the optimizer must
    not rebuild it from these records. Counter references resolve at translate.
    LOOP_START sweeps maps resolved register addresses and WMEM word/field
    indexes (freq, phase, env, gain, length, conf = 0..5) to integer start/step
    words; step is per iteration of this loop. Definitions come from resolved
    QICK sweeps, not their instruction expansions. tables maps WMEM word/field
    indexes to rotated exec-after DMEM word offsets supplied by table macros.
    Conflicting table targets raise ValueError before recording the marker.
    """

    def translate(self, prog) -> None:
        tables: dict[int, dict[int, int]] = {}
        for name, _count, _before, after in prog.loops:
            if name != self.name:
                continue
            for macro in after.macro_list if after is not None else ():
                if isinstance(macro, PatchWmemFromDmem):
                    for address, fields in macro.table_definition(prog).items():
                        if address in tables:
                            raise ValueError(
                                "conflicting table hooks for one WMEM entry"
                            )
                        tables[address] = fields
        prog.record_meta(
            type="LOOP_START",
            name=self.name,
            info={
                "n": self.n,
                "range_hint": None,
                "native": True,
                "tables": tables,
                "sweeps": {
                    "registers": {
                        reg.full_addr(): {
                            "start": int(reg.init.start),
                            "step": int(steps[self.name]["step"]),
                        }
                        for reg in prog.reg_dict.values()
                        if isinstance(reg.init, QickRawParam)
                        and (steps := reg.init.steps) is not None
                        and self.name in steps
                    },
                    "waves": {
                        int(prog.wave2idx[wave.name]): {
                            ("freq", "phase", "env", "gain", "length", "conf").index(
                                sweep.par
                            ): {
                                "start": int(sweep.start),
                                "step": int(sweep.steps[self.name]["step"]),
                            }
                            for sweep in wave.sweeps()
                            if self.name in sweep.steps
                        }
                        for wave in prog.waves
                        if any(self.name in sweep.steps for sweep in wave.sweeps())
                    },
                },
            },
            regs={"counter_reg": self.name},
        )
        super().translate(prog)
        prog.record_meta(type="LOOP_BODY_START", name=self.name, info={"native": True})


class MarkedCloseLoop(CloseLoop):
    """Close the current QICK loop, retaining sweep updates and counter resets.

    All native loop records carry native=True. BODY_END precedes QICK sweep
    updates and the backedge; END follows the complete native closing shell.
    """

    def translate(self, prog) -> None:
        name, _count = prog.loop_stack[-1]
        prog.record_meta(type="LOOP_BODY_END", name=name, info={"native": True})
        super().translate(prog)
        prog.record_meta(type="LOOP_END", name=name, info={"native": True})
