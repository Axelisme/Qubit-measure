"""Native QICK operations annotated without replacing their control shell."""

from qick.asm_v2 import (
    AsmInst,
    CloseLoop,
    OpenLoop,
    Pulse,
    QickRawParam,
    ReadDmem,
    ReadWmem,
    WriteReg,
    WriteWmem,
)

from ..table_param import TableRawParam

_WAVE_FIELDS = ("freq", "phase", "env", "gain", "length", "conf")


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
    indexes to rotated DMEM word offsets discovered from TableRawParam fields.
    The same definitions drive metadata and native closing-shell updates.
    """

    def preprocess(self, prog) -> None:
        super().preprocess(prog)
        self.tables: dict[int, dict[int, int]] = {}
        for wave in prog.waves:
            fields: dict[int, int] = {}
            for field, par in enumerate(_WAVE_FIELDS):
                raw = getattr(wave, par)
                if isinstance(raw, TableRawParam) and raw.loop == self.name:
                    raw.to_steps(prog.loop_dict)
                    rotated = raw.values[1:] + raw.values[:1]
                    fields[field] = prog.add_dmem(
                        [(word + 2**31) % 2**32 - 2**31 for word in rotated]
                    )
            if fields:
                self.tables[int(prog.wave2idx[wave.name])] = fields
        prog.loop_wave_tables[self.name] = self.tables
        if self.tables:
            prog.add_reg(name="scratch", allow_reuse=True)

    def translate(self, prog) -> None:
        prog.record_meta(
            type="LOOP_START",
            name=self.name,
            info={
                "n": self.n,
                "range_hint": None,
                "native": True,
                "tables": self.tables,
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
        counter = prog.reg_dict[name].full_addr()
        for address, fields in prog.loop_wave_tables[name].items():
            wave_name = prog.waves[address].name
            for field, offset in fields.items():
                scratch = prog.reg_dict["scratch"].full_addr()
                AsmInst(
                    inst={
                        "CMD": "REG_WR",
                        "DST": scratch,
                        "SRC": "op",
                        "OP": f"{counter} + #{offset}",
                    },
                    addr_inc=1,
                ).translate(prog)
                # The address is dead after the read, so reuse QICK scratch for the value.
                ReadDmem(dst="scratch", addr="scratch").translate(prog)
                ReadWmem(name=wave_name).translate(prog)
                WriteReg(dst="w_" + _WAVE_FIELDS[field], src="scratch").translate(prog)
                WriteWmem(name=wave_name).translate(prog)
        super().translate(prog)
        prog.record_meta(type="LOOP_END", name=name, info={"native": True})
