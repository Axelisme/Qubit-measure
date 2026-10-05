"""Error raised when the Bloch lowering cannot represent a module."""


class UnsupportedModuleError(ValueError):
    """Raised when a module cannot be faithfully lowered to a Bloch timeline.

    Fast-fail (per CLAUDE.md): register-driven constructs whose values cannot be
    recovered statically, and any unknown module type raise rather than being
    silently approximated.  Deterministic ``Branch`` (selected by a registered
    sweep-loop counter) *is* lowered, and so are ``Repeat`` and ``ComputedPulse``
    whose register values come from a LoadValue table; measurement-conditional
    branches, nested branches, and readout inside a branch or a repeat fast-fail.
    """
