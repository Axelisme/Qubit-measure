"""Captured source queries for context-dependent Form definitions."""

from dataclasses import dataclass

from zcu_tools.gui.session.parameters import ParameterContext


@dataclass(frozen=True)
class FormContext:
    """Read-only reference existence for one Open/Reset factory invocation.

    parameters is the fixed point, alias and library projection, or None when
    no parameter source is available. It is borrowed read-only, never refreshed
    or mutated. No adapter identity or live session state is retained.
    """

    parameters: ParameterContext | None

    def has(self, ref: str) -> bool:
        """Report whether a captured library root or module slot exists.

        ref is a direct library key such as Q1.pulses.pi or a point module slot
        such as $qubit.module.drive. Missing sources, roots, slots and unbound
        aliases return False. Invalid syntax and source faults raise. This
        query does not materialize modules, read assets or consult live aliases.
        """
        return self.parameters is not None and self.parameters.has_library_key(ref)
