from __future__ import annotations

from ..instructions import GenericInst, TimeInst
from ..node import IRNode, RootNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import walk_instructions


class TimingSanityPass(AbsPipeLinePass):
    """Validate conservative timing invariants for TIME instructions."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        for inst in walk_instructions(ir):
            if isinstance(inst, TimeInst):
                self._validate_time(inst)
            elif isinstance(inst, GenericInst) and inst.cmd == "TIME":
                self._validate_time_from_generic(inst)
        return ir

    def _validate_time(self, inst: TimeInst) -> None:
        c_op = inst.c_op
        if c_op is not None and c_op not in {"rst", "updt", "set_ref", "inc_ref"}:
            raise ValueError(f"Unsupported TIME C_OP: {c_op}")

        lit = inst.lit
        if not isinstance(lit, str) or not lit.startswith("#"):
            return

        try:
            value = int(lit[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid TIME literal: {lit}") from exc

        if value < 0:
            raise ValueError(f"TIME literal must be non-negative: {lit}")

    def _validate_time_from_generic(self, inst: GenericInst) -> None:
        c_op = inst.args.get("C_OP")
        if c_op is not None and c_op not in {"rst", "updt", "set_ref", "inc_ref"}:
            raise ValueError(f"Unsupported TIME C_OP: {c_op}")

        lit = inst.args.get("LIT")
        if not isinstance(lit, str) or not lit.startswith("#"):
            return

        try:
            value = int(lit[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid TIME literal: {lit}") from exc

        if value < 0:
            raise ValueError(f"TIME literal must be non-negative: {lit}")
