"""Native QICK operations annotated without replacing their control shell."""

from qick.asm_v2 import CloseLoop, OpenLoop, Pulse


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
    """

    def translate(self, prog) -> None:
        prog.record_meta(
            type="LOOP_START",
            name=self.name,
            info={"n": self.n, "range_hint": None, "native": True},
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
