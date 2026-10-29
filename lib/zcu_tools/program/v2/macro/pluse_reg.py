from __future__ import annotations

from numbers import Integral
from typing import Self

from qick.asm_v2 import AsmInst, Macro, QickParam, TimedMacro

from .write_reg import WriteRegOp


class PulseByReg(TimedMacro):
    """Play one or more pre-addressed wmem entries back-to-back at the same TIME.

    Emits exactly ``len(addr_regs)`` ``WPORT_WR`` instructions with no other
    instructions interleaved, so the hardware can chain a multi-wave pulse
    (e.g. flat_top: ramp_up / flat / ramp_down) without a gap. All address
    arithmetic must run before this macro translates.
    """

    # fields: ch (int), addr_regs (list[str]), t (float | QickParam)
    flat_top_pulse: bool = False

    @classmethod
    def from_base(
        cls,
        prog,
        ch: int,
        base: str,
        t: float | QickParam = 0.0,
        *,
        flat_top_pulse: bool = False,
    ) -> Self:
        """Describe base-address playback, leaving address storage to the caller.

        base names a wmem-index register. flat_top_pulse selects the
        three-entry ramp/flat/ramp layout; unsupported int4 raises before any
        scratch reservation. Call prepare_addresses before translation.
        """
        source = cls(ch=ch, t=t, addr_regs=[base], flat_top_pulse=flat_top_pulse)
        source._check_layout(prog)
        return source

    @property
    def address_scratch_count(self) -> int:
        """Return how many distinct scratch registers address preparation needs."""
        return 2 if self.flat_top_pulse else 0

    def prepare_addresses(self, prog, registers: tuple[str, ...]) -> list[Macro]:
        """Prepare this source's actual addresses in caller-owned storage.

        registers must contain address_scratch_count names. Return operations
        that must execute last before this source consumes the registers. This
        step leaves register resolution deferred, as native macros require.
        Audited callers must call validate_address_storage after allocation and
        own write provenance and lifetime through translation. Native int4
        flat-top layouts raise NotImplementedError.
        """
        self._check_layout(prog)
        if len(registers) != self.address_scratch_count:
            raise ValueError(
                "pulse address preparation requires matching scratch storage"
            )
        base = self.addr_regs[0]
        self.addr_regs = [base, *registers]
        return [
            WriteRegOp(dst=register, lhs=base, op="+", rhs=offset)
            for offset, register in enumerate(registers, start=1)
        ]

    def validate_address_storage(self, prog) -> None:
        """Reject prepared address storage that aliases live operands.

        Call after allocation and time conversion. Derived addresses must be
        distinct from each other, the base and any register TIME operand.
        Native callers retain their original deferred-resolution contract.
        """
        occupied = {prog._get_reg(self.addr_regs[0])}
        time = self.t_regs.get("t")
        if isinstance(time, str):
            occupied.add(prog._get_reg(time))
        physical = [prog._get_reg(register) for register in self.addr_regs[1:]]
        if len(set(physical)) != len(physical) or any(
            register in occupied for register in physical
        ):
            raise ValueError(
                "pulse address storage aliases base, TIME or another address"
            )

    def _check_layout(self, prog) -> None:
        if self.flat_top_pulse:
            gen_type = prog.soccfg["gens"][self.ch].get("type")
            if gen_type in ("axis_sg_int4_v1", "axis_sg_int4_v2"):
                raise NotImplementedError(
                    "pulse_by_reg(flat_top_pulse=True) is not supported on "
                    f"generator type {gen_type!r}: int4 flat_top uses a "
                    f"4-entry wmem stride, not 3. ch={self.ch}"
                )

    def preprocess(self, prog):
        self.convert_time(prog, self.t, "t")

    # tolerated: QICK Macro.expand is inferred as None, but translate iterates the returned macro list.
    def expand(self, prog):  # pyright: ignore[reportIncompatibleMethodOverride]
        tproc_ch = prog.soccfg["gens"][self.ch]["tproc_ch"]
        t_reg = self.t_regs["t"]

        insts = []
        imm_time = isinstance(t_reg, Integral)
        if not imm_time:
            insts.append(self.set_timereg(prog, "t"))

        for reg_name in self.addr_regs:
            inst = {
                "CMD": "WPORT_WR",
                "DST": str(tproc_ch),
                "SRC": "wmem",
                "ADDR": f"&{prog._get_reg(reg_name)}",
            }
            if imm_time:
                inst["TIME"] = "@" + str(t_reg)
            insts.append(AsmInst(inst=inst, addr_inc=1))
        return insts
