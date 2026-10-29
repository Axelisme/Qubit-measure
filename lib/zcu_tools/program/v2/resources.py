"""Data-memory allocation and readout-route ownership shared by v2 programs."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import QickProgramV2


class ProgramResourcesMixin(QickProgramV2):
    """Add ordered int32 dmem allocation and protected logical readout routes.

    Used by both native timeline and module programs. Other QICK declaration
    and compilation behavior is unchanged. Arguments pass to QickProgramV2.
    """

    def __init__(self, *args, **kwargs):
        self._reset_resources()
        super().__init__(*args, **kwargs)

    def _reset_resources(self) -> None:
        self._dmem_buffer: list[int] = []
        self._protected_gen_routes: dict[int, str] = {}
        self._protected_ro_routes: dict[int, str] = {}

    def protect_readout_routes(
        self, owner: str, gen_ch: int, ro_chs: Sequence[int]
    ) -> None:
        """Protect successfully declared generator/ADC channels from redeclaration.

        Call after a module's declarations succeed, including in nested module
        init. owner is its diagnostic name; gen_ch and ro_chs are explicit
        logical channel indices already declared on this program. Future
        declare_gen/readout calls for these routes raise ValueError naming the
        owner and channel. Unprotected routes retain QICK's overwrite semantics.
        Missing declarations or already protected routes raise ValueError before
        changing any protection. No hardware lookup or route inference occurs.
        """
        if gen_ch not in self.gen_chs:
            raise ValueError(f"{owner}: generator route {gen_ch} is not declared")
        if gen_ch in self._protected_gen_routes:
            raise ValueError(
                f"{owner}: generator route {gen_ch} already protected by "
                f"{self._protected_gen_routes[gen_ch]}"
            )
        for ch in ro_chs:
            if ch not in self.ro_chs:
                raise ValueError(f"{owner}: readout route {ch} is not declared")
            if ch in self._protected_ro_routes:
                raise ValueError(
                    f"{owner}: readout route {ch} already protected by "
                    f"{self._protected_ro_routes[ch]}"
                )
        self._protected_gen_routes[gen_ch] = owner
        self._protected_ro_routes.update((ch, owner) for ch in ro_chs)

    def declare_gen(self, ch: int, *args, **kwargs) -> None:
        """Delegate QICK declaration arguments unchanged unless ch is protected.

        A protected generator raises ValueError with its owner and channel;
        unprotected generators retain the inherited QICK declaration contract.
        """
        if ch in self._protected_gen_routes:
            raise ValueError(
                f"{self._protected_gen_routes[ch]}: generator route {ch} is protected"
            )
        super().declare_gen(ch, *args, **kwargs)

    def declare_readout(self, ch: int, *args, **kwargs) -> None:
        """Delegate QICK declaration arguments unchanged unless ch is protected.

        A protected ADC raises ValueError with its owner and channel;
        unprotected ADCs retain the inherited QICK declaration contract.
        """
        if ch in self._protected_ro_routes:
            raise ValueError(
                f"{self._protected_ro_routes[ch]}: readout route {ch} is protected"
            )
        super().declare_readout(ch, *args, **kwargs)

    def add_dmem(self, values: Sequence[int]) -> int:
        """Append int32 words and return their zero-based start in shared dmem.

        Allocation follows call order; compilation converts all words to int32.
        Invalid or out-of-range values fail at NumPy compilation, as in QICK.
        """
        offset = len(self._dmem_buffer)
        self._dmem_buffer.extend(values)
        return offset

    def add_timing_dmem(self, values: Sequence[int], *, owner: str) -> int:
        """Append exact cycle words, or fail before exceeding hardware capacity.

        owner identifies the emission requesting the table. Existing allocations
        count against the same QICK dmem_size word limit. No words are appended
        on failure; the error names owner, requested size and available capacity.
        """
        capacity = self.tproccfg["dmem_size"]
        available = capacity - len(self._dmem_buffer)
        if len(values) > available:
            raise ValueError(
                f"{owner}: timing table requires {len(values)} dmem words; "
                f"{available} available of {capacity}"
            )
        return self.add_dmem(values)

    # tolerated: QICK infers None from its empty base implementation; compile_datamem documents ndarray or None.
    def compile_datamem(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> NDArray[np.int32] | None:
        """Return all allocated words as int32, or None for an empty buffer."""
        if len(self._dmem_buffer) == 0:
            return None

        return np.array(self._dmem_buffer, dtype=np.int32)
